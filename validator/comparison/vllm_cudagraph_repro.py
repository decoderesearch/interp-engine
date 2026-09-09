"""Does vLLM's greedy output move when CUDA graphs are the *only* thing that changes?

Standalone: plain vLLM, no interp-engine, nothing from this repo. That is the point -- it is the
repro shape a vLLM maintainer can paste and run, and it is what vllm-project/vllm#55238 is missing.

The issue as filed toggles ``enforce_eager``, which disables torch.compile *and* CUDA graphs
together, so it never isolated graphs from compilation. The arms here do:

    nograph    CompilationMode.NONE + cudagraph_mode NONE    no inductor, no graphs
    graph      CompilationMode.NONE + cudagraph_mode FULL     no inductor, graphs
    nocustom   nograph + custom_ops=none                      no inductor, no graphs, native norms
    default    whatever the engine picks                      inductor + graphs + native norms

``nograph`` against ``graph`` is the comparison the title claims, and it is a true single-variable
step: everything else the engine reports -- attention backend, MoE backend, custom ops, norm kernels
-- is identical between the two. ``default`` is there because a third answer is possible (that
neither change alone moves the output and the interaction does) and because it is what everybody
actually serves, but it is three steps from ``nograph``, not one: it turns on inductor, and in doing
so flips ``custom_ops`` to ``none`` and RMSNorm from vLLM's C kernel to the native one. ``nocustom``
takes that last step by itself so a move under ``default`` can be attributed.

Controls are the two dense Gemma 4 checkpoints. They share Gemma 4's hybrid sliding/full attention
layout and carry no MoE block, so if they hold their argmax while the MoE 26B moves, "Gemma 4
numerics" is out; Qwen3-30B-A3B already covers MoE with uniform attention. If they move too, the
finding is about the family and the issue should be closed.

Severity is reported per position as the eager top1-minus-top2 margin beside the observed delta,
because magnitude alone does not predict a flip -- a wide-margin token absorbs several nats and a
near-tie flips on a fraction of one.

What it found, so a re-run has something to check itself against: on compute capability 10.x the two
graph arms diverge by up to 6.8 nats and the token moves, on 9.x and 12.x they are bit-identical, and
``VLLM_BATCH_INVARIANT=1`` removes it. Full account in ``../docs/VLLM_SM100_CUDAGRAPH.md``.

Usage, from a box with the checkpoints cached::

    python comparison/vllm_cudagraph_repro.py                      # every model, every arm
    python comparison/vllm_cudagraph_repro.py --models 26b         # just the MoE one
    python comparison/vllm_cudagraph_repro.py --capture 26b graph   # one arm (internal)
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys

PROMPT = "The capital of France is Paris, and the capital of Japan is"

MODELS = {
    "26b": "google/gemma-4-26B-A4B-it",  # MoE + hybrid sliding/full attention -- the subject
    "12b": "google/gemma-4-12B-it",  # dense, same attention layout -- control
    "31b": "google/gemma-4-31B",  # dense, same attention layout -- control
}

# `mode` is torch.compile, `cudagraph_mode` is replay, and holding the first fixed across the first
# two arms is the whole design. Both fields are 0.28.0's spelling (`level` was renamed to `mode`).
#
# `nocustom` exists because ``default`` is not one step away from ``nograph``: turning on
# VLLM_COMPILE also flips ``custom_ops`` from ``all`` to ``none``, which swaps RMSNorm from vLLM's
# C kernel to the native PyTorch one. Without this arm, a move under ``default`` cannot be pinned on
# inductor rather than on that kernel substitution, and the two have very different implications.
ARMS = {
    "nograph": {"mode": 0, "cudagraph_mode": "NONE"},
    "graph": {"mode": 0, "cudagraph_mode": "FULL"},
    "nocustom": {"mode": 0, "cudagraph_mode": "NONE", "custom_ops": ["none"]},
    "default": None,
}


def capture(model_key: str, arm: str, out_dir: str) -> None:
    """Run one (model, arm) and write its next token and prompt logprobs.

    In its own process because an engine holds the GPU for its lifetime, and because a second
    ``LLM`` in a process that has already built one is not a clean arm.

    The prompt is generated twice against the *same* engine and both results are recorded, which
    costs nothing and answers the repeatability half: if a single arm disagrees with itself, no
    comparison between arms means anything. It is the weaker version of that test -- same process,
    same batch shape, so it sees replay nondeterminism and not batch-shape-dependent kernel choice
    -- and running this script twice is the stronger one.
    """
    from vllm import LLM, SamplingParams

    kwargs = {}
    if ARMS[arm] is not None:
        kwargs["compilation_config"] = ARMS[arm]
    llm = LLM(
        model=MODELS[model_key],
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        # Off because a cached prefix is never re-forwarded, so the second pass below would read the
        # first pass's logprobs and every arm would look self-consistent for the wrong reason.
        enable_prefix_caching=False,
        **kwargs,
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20, prompt_logprobs=20)
    passes = []
    for _ in range(2):
        out = llm.generate([PROMPT], params)[0]
        step = (out.outputs[0].logprobs or [None])[0]
        passes.append(
            {
                "next_token": int(out.outputs[0].token_ids[0]),
                # The distribution at the *generated* position. Kept separately because
                # ``prompt_logprobs`` stops at the last prompt token, so without this the one
                # position where the argmax is actually free to move is the one not recorded.
                "next_logprobs": ({} if step is None else {int(k): float(v.logprob) for k, v in step.items()}),
                "prompt_logprobs": [
                    None if p is None else {int(k): float(v.logprob) for k, v in p.items()}
                    for p in (out.prompt_logprobs or [])
                ],
            }
        )
    with open(os.path.join(out_dir, f"{model_key}.{arm}.json"), "w") as f:
        json.dump({"model": MODELS[model_key], "arm": arm, "passes": passes}, f)


def _dists(p: dict) -> list[tuple[str, dict]]:
    """The pass's distributions in sequence order, with the generated position last as ``gen``."""
    rows = [(str(i), d) for i, d in enumerate(p["prompt_logprobs"]) if d is not None]
    if p.get("next_logprobs"):
        rows.append(("gen", p["next_logprobs"]))
    return rows


def _positions(a: dict, b: dict) -> list[dict]:
    """Per position: the reference's own margin, the worst shared-token delta, and the flip.

    ``margin`` is read off ``a`` alone. It is a property of the distribution at that position rather
    than of the comparison -- how much perturbation the argmax there can absorb before it moves --
    and it is what separates "this arm is further off" from "this prompt sits on ties".
    """
    rows = []
    for (label, pa), (_, pb) in zip(_dists(a), _dists(b), strict=False):
        ranked = sorted(pa.values(), reverse=True)
        shared = set(pa) & set(pb)
        rows.append(
            {
                "position": label,
                "margin": ranked[0] - ranked[1] if len(ranked) > 1 else float("inf"),
                "worst_delta": max((abs(pa[k] - pb[k]) for k in shared), default=0.0),
                "flip": max(pa, key=lambda k: pa[k]) != max(pb, key=lambda k: pb[k]),
            }
        )
    return rows


def _identical(a: dict, b: dict) -> bool:
    return (
        a["next_token"] == b["next_token"]
        and a["prompt_logprobs"] == b["prompt_logprobs"]
        and a.get("next_logprobs") == b.get("next_logprobs")
    )


def compare(model_key: str, out_dir: str) -> None:
    arms = {}
    for arm in ARMS:
        path = os.path.join(out_dir, f"{model_key}.{arm}.json")
        if os.path.exists(path):
            with open(path) as f:
                arms[arm] = json.load(f)
    if not arms:
        return
    print(f"\n=== {MODELS[model_key]}")
    for arm, data in arms.items():
        first, second = data["passes"]
        state = "bit-identical" if _identical(first, second) else "DIFFERS FROM ITSELF"
        print(f"  {arm:8s} next token {first['next_token']:6d}   same engine twice: {state}")

    for left, right in itertools.combinations(ARMS, 2):
        if left not in arms or right not in arms:
            continue
        a, b = arms[left]["passes"][0], arms[right]["passes"][0]
        rows = _positions(a, b)
        flips = [r for r in rows if r["flip"]]
        token = "same" if a["next_token"] == b["next_token"] else f"{a['next_token']} vs {b['next_token']}"
        verdict = "BIT-IDENTICAL" if _identical(a, b) else f"argmax differs at {len(flips)}/{len(rows)}"
        print(f"\n  {left} vs {right}: next token {token}, {verdict}")
        if not rows or _identical(a, b):
            continue
        print(f"    {'pos':>4s} {'margin':>9s} {'worst delta':>12s}   flip")
        for r in rows:
            print(
                f"    {r['position']:>4s} {r['margin']:9.4f} {r['worst_delta']:12.4f}   {'FLIP' if r['flip'] else ''}"
            )
        if flips:
            print(f"    widest margin that flipped: {max(f['margin'] for f in flips):.4f}")
        held = [r for r in rows if not r["flip"]]
        if held:
            print(f"    largest delta that did not flip: {max(r['worst_delta'] for r in held):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", default="/tmp/vllm_cudagraph_repro")
    ap.add_argument("--capture", nargs=2, metavar=("MODEL", "ARM"), help="run one arm in this process")
    ap.add_argument("--report", action="store_true", help="re-report from --out, run nothing")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.capture:
        capture(args.capture[0], args.capture[1], args.out)
        return
    for model_key in [] if args.report else args.models:
        for arm in args.arms:
            print(f"[repro] {MODELS[model_key]} {arm}", flush=True)
            # `check=False`: one arm that will not boot (an OOM on the 31B, a config a build refuses)
            # should cost its own row and not the other eight. `compare` reports what is on disk.
            subprocess.run(
                [sys.executable, __file__, "--capture", model_key, arm, "--out", args.out],
                check=False,
            )
    for model_key in args.models:
        compare(model_key, args.out)


if __name__ == "__main__":
    main()
