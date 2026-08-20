"""Tokenize each model's prompt ONCE and write shared input ids, so every engine consumes
identical tokens and any activation diff reflects model numerics, not tokenizer differences.

Run in any env with `transformers` (the driver/aggregation env):

    python -m comparison.tokenize_inputs --dumps <dir> [--models openai-community/gpt2 Qwen/Qwen3-1.7B]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from comparison.dumpio import InputSpec, write_inputs
from comparison.spec import MODELS, PROMPT, layers_for, load_sweep


def _n_layers(config) -> int:
    for attr in ("num_hidden_layers", "n_layer"):
        v = getattr(config, attr, None)
        if v:
            return int(v)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for attr in ("num_hidden_layers", "n_layer"):
            v = getattr(text_config, attr, None)
            if v:
                return int(v)
    raise ValueError("Could not determine num_hidden_layers from config")


def _attends(config) -> Callable[[int], bool]:
    """Whether a layer computes softmax attention, for the one place that already reads the config.

    Both of this module's callers want it: `layers_for` to make sure the layer plan includes a layer
    that attends, and `linear_attn_layers` to record the ones that don't so the aggregator can exclude
    `attn_out` there without re-downloading a config for every model in the sweep.
    """
    from interp_engine import facts

    resolved = facts.resolve_facts(config)
    return lambda layer: not resolved.is_linear_attention_layer(layer)


def tokenize_hf(hf_id: str, dumps: str) -> InputSpec:
    """Tokenize PROMPT for any HF id (including one not in the sweep list — used by check_model.py)
    and write the shared inputs file."""
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    # Model's own default special-token handling; this is the single tokenization all engines
    # are fed verbatim (each engine is invoked so it does NOT re-add specials).
    input_ids = [int(t) for t in tokenizer(PROMPT, add_special_tokens=True)["input_ids"]]
    n_layers = _n_layers(config)
    attends = _attends(config)
    layers = layers_for(n_layers, attends=attends)
    spec = InputSpec(
        hf_id=hf_id,
        input_ids=input_ids,
        n_layers=n_layers,
        layers=layers,
        linear_attn_layers=[layer for layer in layers if not attends(layer)],
    )
    write_inputs(dumps, spec)
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument(
        "--models", nargs="*", default=None, help="HF repo ids; default = the core 4 (or all of --models-json)"
    )
    ap.add_argument(
        "--models-json",
        default=None,
        help="path to a JSON list of HF repo ids (default comparison/sweep_models.json) for the broad sweep",
    )
    args = ap.parse_args()

    ids = args.models or ([m.hf_id for m in MODELS] if not args.models_json else list(load_sweep(args.models_json)))
    for hf_id in ids:
        try:
            spec = tokenize_hf(hf_id, args.dumps)
            print(f"[tokenize] {hf_id}: {len(spec.input_ids)} tokens, layers={spec.layers}")
        except Exception as exc:  # noqa: BLE001
            print(f"[tokenize] {hf_id}: FAILED ({type(exc).__name__}: {str(exc)[:160]})")


if __name__ == "__main__":
    main()
