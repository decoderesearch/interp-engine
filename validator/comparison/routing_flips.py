"""Did the two engines send each token to the same experts?

    python -m comparison.routing_flips [--dumps dumps] [--reference eager] [--engine vllm]

An MoE model's feed-forward is a *discontinuous* function of its router logits: the top-k is a
selection, so two engines whose logits agree to a cosine of 0.99999 can still route a token whose
top two experts are nearly tied to different experts entirely, and then that token's `mlp_out` is
some other expert's output rather than a rounding of the same one. Nothing in the comparison table
can tell that apart from ordinary kernel numerics — `mlp_out` is one number per point, and a single
flipped token out of fourteen drags the whole layer's cosine down while every other token is fine.

This reads the `router_logits` both engines already captured, takes the top-k of each as *integers*,
and reports the tokens where the two selections differ alongside that token's own `mlp_out`
agreement. Indices rather than weights on purpose: which experts win is convention-independent,
while the weights are not (Mixtral softmaxes before selecting, gpt-oss after, DeepSeek-V3 scores
with a sigmoid inside expert groups), so comparing weights would need a per-family rule and
comparing indices needs none.

Read-only over an existing dumps tree; captures nothing and loads no model beyond its config, which
is where the top-k comes from.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _top_k(config) -> int | None:
    """How many experts fire per token, under whichever name this family spells it."""
    text = getattr(config, "text_config", config)
    for field in ("num_experts_per_tok", "experts_per_token", "moe_topk"):
        value = getattr(text, field, None)
        if value:
            return int(value)
    return None


def _models(dumps: str, engine: str) -> set[str]:
    root = os.path.join(dumps, engine)
    return {
        os.path.relpath(path, root)[: -len(".npz")]
        for path in glob.glob(os.path.join(root, "**", "*.npz"), recursive=True)
    }


def _per_token_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per row, which is what makes one bad token distinguishable from a bad layer."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    norms = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.where(norms > 0, (a * b).sum(-1) / np.where(norms > 0, norms, 1.0), 1.0)


def compare(reference: dict[str, np.ndarray], engine: dict[str, np.ndarray], top_k: int) -> list[dict]:
    """One row per layer whose ``router_logits`` both engines captured."""
    rows: list[dict] = []
    keys = [k for k in reference if k.startswith("router_logits.") and k in engine]
    for key in sorted(keys, key=lambda k: int(k.split(".", 1)[1])):
        layer = key.split(".", 1)[1]
        a, b = reference[key], engine[key]
        if a.shape != b.shape:
            rows.append({"layer": layer, "shapes": (a.shape, b.shape)})
            continue
        chosen_a = np.argsort(-a.astype(np.float64), axis=-1)[:, :top_k]
        chosen_b = np.argsort(-b.astype(np.float64), axis=-1)[:, :top_k]
        flipped = np.array([set(x) != set(y) for x, y in zip(chosen_a, chosen_b, strict=True)])
        row = {"layer": layer, "tokens": int(flipped.size), "flipped": int(flipped.sum())}
        mlp_out = f"mlp_out.{layer}"
        if mlp_out in reference and mlp_out in engine and reference[mlp_out].shape == engine[mlp_out].shape:
            cosine = _per_token_cosine(reference[mlp_out], engine[mlp_out])
            row["cos_flipped"] = [round(float(c), 4) for c in cosine[flipped]]
            row["cos_agreed_min"] = round(float(cosine[~flipped].min()), 5) if (~flipped).any() else None
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="dumps")
    ap.add_argument("--reference", default="eager")
    ap.add_argument("--engine", default="vllm")
    args = ap.parse_args()

    from transformers import AutoConfig

    shared = sorted(_models(args.dumps, args.reference) & _models(args.dumps, args.engine))
    if not shared:
        print(f"no models captured by both {args.reference} and {args.engine} under {args.dumps}")
        return 2
    for hf_id in shared:
        with np.load(os.path.join(args.dumps, args.reference, f"{hf_id}.npz")) as dump:
            reference = {key: dump[key] for key in dump}
        if not any(key.startswith("router_logits.") for key in reference):
            continue
        with np.load(os.path.join(args.dumps, args.engine, f"{hf_id}.npz")) as dump:
            engine = {key: dump[key] for key in dump}
        top_k = _top_k(AutoConfig.from_pretrained(hf_id, trust_remote_code=True))
        if top_k is None:
            print(f"{hf_id}: config states no top-k, so no selection to compare")
            continue
        print(f"{hf_id} (top_k={top_k})")
        for row in compare(reference, engine, top_k):
            if "shapes" in row:
                print(f"  layer {row['layer']}: not comparable, {row['shapes'][0]} vs {row['shapes'][1]}")
                continue
            line = f"  layer {row['layer']}: {row['flipped']}/{row['tokens']} tokens routed differently"
            if "cos_flipped" in row:
                line += f"; mlp_out cos there {row['cos_flipped']}, elsewhere >= {row['cos_agreed_min']}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
