"""Per-model behavioral expectations, driven by model_expectations.yaml.

Every model-specific number lives in the YAML, so covering a new model is a block there
rather than a new test function here. What this module owns is the *shape* of the questions:
does the read-out still recover a token the model obviously predicts, are its probabilities
still probabilities, is generation still deterministic, do attention rows still sum to 1.

Reference-free by design -- no TransformerLens, no second environment, no SAE. That is what
lets the whole file run in the plain CPU CI job.

What these catch that the structural tests do not: `assert_logit_lens_self_consistent`
proves the read-out agrees with the model's own forward pass, which stays true when both are
wrong together (a mis-resolved final_norm, say, that the true forward also routes through).
Naming the token the model must predict fails on that.

Each row runs twice, once per device: `<model>-cpu` at the harness dtype and `<model>-cuda` in
bf16. The CUDA half is what the GPU job selects, and it is not redundant -- it is where device
placement and bf16 rounding through the read-out show up, in the configuration the inference
app actually serves.

    pytest tests/test_model_expectations.py -m "not gpu and not xl"   # what the CPU job runs
    pytest tests/test_model_expectations.py -m "gpu and not xl"       # what the GPU job runs
    pytest tests/test_model_expectations.py -m xl                     # multi-GB rows, big box
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from harness import MODELS, ModelSpec, evict_models, load_model, parity_required

from interp_engine import EagerModel, generate_stream, layer_logits, run_with_cache

_YAML = Path(__file__).parent / "model_expectations.yaml"


@dataclass(frozen=True)
class Expectation:
    """One row of model_expectations.yaml, with the file-wide bindings already merged in.

    Bound to a device, because each row is run twice: once on CPU at the harness dtype and once
    on CUDA in bf16. Same assertions, two numeric regimes.
    """

    key: str
    spec: ModelSpec
    values: dict[str, Any]
    device: str = "cpu"

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


def _load_rows() -> list[Expectation]:
    """Merge each model block over the file-wide bindings.

    A key that names a row of the harness matrix reuses that ModelSpec, so the CI model list
    stays single-sourced in harness.py. A key that does not (the `manual` tier, whose weights
    CI must never download) builds its spec from the row's own fields.
    """
    doc = yaml.safe_load(_YAML.read_text(encoding="utf-8"))
    bindings = doc.get("bindings", {})

    rows: list[Expectation] = []
    for key, overrides in doc["models"].items():
        values = {**bindings, **(overrides or {})}
        spec = MODELS.get(key)
        if spec is None:
            hf_id = values.get("hfId")
            if not hf_id:
                raise ValueError(f"model_expectations.yaml: '{key}' is not in the harness matrix, so it needs an hfId")
            spec = ModelSpec(
                key=key,
                model_id=hf_id,
                dtype=values.get("dtype", "auto"),
                is_chat=bool(values.get("chat", False)),
                is_gated=bool(values.get("gated", False)),
            )
        rows.append(Expectation(key=key, spec=spec, values=values))
    return rows


ROWS = _load_rows()


def _params() -> list[Any]:
    """One pytest.param per (row, device), carrying the marks its tier, repo and device imply.

    Marks go on the param rather than the module -- mirroring harness.spec_params -- so a gated
    or multi-GB model deselects on its own without taking the rest of the matrix with it.

    A `ci` row is emitted twice. The `cuda` half carries @pytest.mark.gpu, which is what routes
    it to the GPU job's `-m "gpu and not xl"` and keeps it out of the CPU job's `-m "not gpu"`.
    Without it this whole file would be CPU-only, and the numbers below would never be checked
    in the dtype and on the device the inference app actually serves.

    A `manual` row is emitted once, on CUDA only. Its weights are multi-GB, nothing serves them
    on CPU, and the dispatch job that runs them is a GPU box -- a CPU variant would only offer
    the operator a way to OOM it loading a 20B model in fp32.
    """
    params = []
    for row in ROWS:
        manual = row.get("tier") == "manual"
        base_marks = []
        if row.spec.is_gated:
            base_marks.append(pytest.mark.gated)
        if manual:
            base_marks.append(pytest.mark.xl)
        for device in ("cuda",) if manual else ("cpu", "cuda"):
            marks = list(base_marks)
            if device == "cuda":
                marks.append(pytest.mark.gpu)
                marks.append(
                    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/bf16 coverage requires a GPU")
                )
            params.append(pytest.param(replace(row, device=device), id=f"{row.key}-{device}", marks=marks))
    return params


PARAMS = _params()


@pytest.fixture(scope="module", autouse=True)
def _release_vram() -> Iterator[None]:
    """Drop the cached models when the module finishes, so nothing holds VRAM after it."""
    yield
    evict_models()


def _model(row: Expectation) -> EagerModel:
    """The row's model on its bound device, session-cached by the harness so each loads once.

    CUDA rows are pinned to bf16 rather than the spec dtype, matching test_small_models_gpu.py
    and the inference app: serving fp32 on a GPU is not a configuration anything runs, so
    checking these expectations there would spend the runner's minutes on a fiction.

    `required` under IE_REQUIRE_PARITY (set by CI): otherwise a cold HF cache would turn every
    assertion here into a skip and the job would report green having exercised nothing.
    """
    dtype = "bfloat16" if row.device == "cuda" else None
    return load_model(
        row.spec,
        device=row.device,
        dtype=dtype,
        attn_implementation="eager",
        required=parity_required(),
    )


def _prompt_ids(model: EagerModel, row: Expectation) -> torch.Tensor:
    text = row["readoutPrompt"]
    return model.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)


def _topk_tokens(model: EagerModel, logits: torch.Tensor, k: int) -> list[str]:
    """Top-k tokens of a `[vocab]` logit row, whitespace-stripped for comparison.

    Stripping means a row states `dog` and matches whether the model's convention is ` dog`
    or `dog`, so an expectation does not silently depend on a tokenizer's leading-space rule.
    """
    top = torch.topk(logits.float(), k)
    return [model.tokenizer.decode([i]).strip() for i in top.indices.tolist()]


def _layer_readout(model: EagerModel, row: Expectation) -> dict[int, torch.Tensor]:
    """Last-position read-out logits at every layer, through the same call the server uses.

    `layer_logits` defaults its softcap from the arch, which is load-bearing on gemma-2: the
    real lm_head does not apply the final-logit softcap, so a read-out that skips it returns
    plausible-looking wrong tokens.
    """
    ids = _prompt_ids(model, row)
    layers = list(range(model.n_layers))
    out = layer_logits(model, ids, {"logit_lens": layers})["logit_lens"]
    return {layer: logits[-1] for layer, logits in out.items()}


# --- read-out ---------------------------------------------------------------


@pytest.mark.parametrize("row", PARAMS)
def test_readout_recovers_the_answer_at_the_final_layer(row: Expectation):
    """The token the model obviously predicts must survive the read-out.

    This is the assertion the whole file exists for. An unembed applied at the wrong layer, in
    the wrong dtype, or transposed still returns a well-formed distribution over plausible
    tokens; the answer disappearing is the only signal.
    """
    model = _model(row)
    final = model.n_layers - 1
    tokens = _topk_tokens(model, _layer_readout(model, row)[final], row["readoutTopK"])
    assert row["readoutAnswer"] in tokens, (
        f"{row.key}: {row['readoutAnswer']!r} missing from the final-layer top-{row['readoutTopK']} "
        f"for {row['readoutPrompt']!r}; got {tokens}"
    )


@pytest.mark.parametrize("row", PARAMS)
def test_readout_recovers_the_answer_across_layers(row: Expectation):
    """The answer must surface at several layers, not only the last one.

    At the final layer the residual is already what lm_head was trained on, so a read-out that
    is subtly wrong can still look right there. Requiring earlier layers to agree is what makes
    this a test of the intermediate decode rather than of the final forward pass.
    """
    model = _model(row)
    readout = _layer_readout(model, row)
    hits = [
        layer
        for layer, logits in readout.items()
        if row["readoutAnswer"] in _topk_tokens(model, logits, row["readoutTopK"])
    ]
    assert len(hits) >= row["minReadoutLayers"], (
        f"{row.key}: {row['readoutAnswer']!r} reached the top-{row['readoutTopK']} at {len(hits)} layers "
        f"({hits}), expected at least {row['minReadoutLayers']}"
    )


@pytest.mark.parametrize("row", PARAMS)
def test_readout_probabilities_are_probabilities(row: Expectation):
    """Finite, in [0, 1], summing to 1 at every layer.

    Values outside the range mean a softmax was skipped or applied along the wrong axis; NaNs
    render as blank cells in the UI rather than as an error, so nothing else would report them.
    """
    model = _model(row)
    for layer, logits in _layer_readout(model, row).items():
        probs = logits.float().softmax(-1)
        assert torch.isfinite(probs).all(), f"{row.key}: non-finite probability at layer {layer}"
        assert probs.min() >= 0.0 and probs.max() <= 1.0, f"{row.key}: probability outside [0, 1] at layer {layer}"
        assert probs.sum().item() == pytest.approx(1.0, abs=1e-3), (
            f"{row.key}: probabilities do not sum to 1 at layer {layer}"
        )


# --- generation -------------------------------------------------------------


@pytest.mark.parametrize("row", PARAMS)
def test_greedy_generation_is_deterministic(row: Expectation):
    """Two greedy runs of the same prompt must produce the same bytes.

    The one generative property every model here holds regardless of quality: the small models
    continue this prompt badly (gpt2 repeats ' cat', gemma-3-270m-it emits backslashes), so only a
    row that measured something meaningful sets generationPattern.

    This is the only test here that pays for a forward *per token*, which makes it the one where a
    slow layer type is felt: on the hybrid trunk every linear-attention layer falls back to a
    pure-PyTorch chunked recurrence ("the fast path is not available"), since those kernels are
    Triton and no CPU run will have them. That fallback is what makes it the slowest row in the
    CPU suite at ~15s -- and it is affordable at all only because the spec loads that checkpoint in
    fp32; see `harness.QWEN_THINKING` for why bf16 on a CPU is 7.7x slower, not faster.
    """
    model = _model(row)
    ids = _prompt_ids(model, row)
    n = row["generationTokens"]

    def run() -> str:
        return "".join(step.token_str for step in generate_stream(model, ids, max_tokens=n, temperature=0.0))

    first = run()
    assert first, f"{row.key}: greedy generation produced nothing"
    assert first == run(), f"{row.key}: greedy generation is not deterministic ({first!r} then {run()!r})"

    pattern = row.get("generationPattern")
    if pattern is not None:
        assert re.search(pattern, first), f"{row.key}: greedy output {first!r} does not match {pattern!r}"


# --- attention --------------------------------------------------------------


@pytest.mark.parametrize("row", PARAMS)
def test_attention_rows_are_distributions(row: Expectation):
    """Non-negative, and summing to 1 unless the model has attention sinks.

    `attnLayer` must name a softmax-attention layer. On a hybrid trunk most layers are linear
    attention and produce no probabilities at all, so capture raises rather than handing back
    a neighbouring layer's rows -- the failure this would otherwise hide.
    """
    model = _model(row)
    layer = row["attnLayer"]
    assert not model.arch.is_linear_attention_layer(layer), (
        f"{row.key}: attnLayer {layer} is a linear-attention layer; pick one of {model.arch.softmax_attention_layers()}"
    )

    attn = run_with_cache(model, _prompt_ids(model, row), [("attn_probs", layer)]).get("attn_probs", layer)
    assert attn.shape[1] == model.n_heads, f"{row.key}: expected {model.n_heads} heads, got {attn.shape[1]}"
    assert (attn >= -1e-6).all(), f"{row.key}: negative attention weight"

    row_sums = attn.float().sum(dim=-1)
    if row.get("attnRowsSumToOne", True):
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=row["rowSumEpsilon"]), (
            f"{row.key}: attention rows do not sum to 1 (min {row_sums.min():.4f}, max {row_sums.max():.4f})"
        )
    else:
        # Sink models: the sink sits in the softmax denominator, so rows over real tokens sum to
        # LESS than 1 and must never be renormalized. Two halves, because either alone is weak.
        # The upper bound catches a renormalize; on its own it would also pass for a model with
        # no sinks at all. So the sink is also required to be demonstrably absorbing mass.
        assert row_sums.max() <= 1.0 + row["rowSumEpsilon"], f"{row.key}: sink-model rows sum above 1"
        floor = row["attnSinkRowSumBelow"]
        assert row_sums.min() < floor, (
            f"{row.key}: no attention row sums below {floor} (min {row_sums.min():.6f}), so the sink "
            f"is absorbing nothing -- marked attnRowsSumToOne: false but behaving like a model without sinks"
        )

    # Row 0 is a single column (only the first key is visible to the first query), so it is
    # exactly 1 without sinks and strictly below 1 with them.
    first_row = attn[0, :, 0, 0].float()
    assert first_row.min() >= row["attnRow0Min"] - 1e-6, f"{row.key}: attention row 0 below {row['attnRow0Min']}"
    assert first_row.max() <= row["attnRow0Max"] + 1e-6, f"{row.key}: attention row 0 above {row['attnRow0Max']}"


# --- the data file itself ---------------------------------------------------


def test_every_matrix_model_has_expectations():
    """The YAML and the harness matrix must not drift apart.

    Without this, adding a model to harness.py silently buys it no behavioral coverage -- CI
    would download the weights, run the structural tests, and report green.
    """
    missing = sorted(set(MODELS) - {row.key for row in ROWS})
    assert not missing, f"models in harness.MODELS with no model_expectations.yaml row: {missing}"


def test_manual_rows_are_not_in_the_ci_matrix():
    """A `manual` row naming a matrix model would be marked xl and so run nowhere."""
    for row in ROWS:
        if row.get("tier") == "manual":
            assert row.key not in MODELS, f"{row.key} is in the CI matrix but marked tier: manual"
