"""The eager backend on a Google Cloud TPU, through TorchTPU's native ``tpu`` device.

Two halves. The first is weight-free and runs everywhere: the fp32-matmul warning's decision and the
device-agnostic cache release. The second is ``@pytest.mark.tpu`` and self-skips on any machine
where TorchTPU has not registered a ``tpu`` accelerator, the way the ``gpu`` tier skips without CUDA.
Run it on a TPU VM with ``make test-tpu``; docs/TPU.md has the whole route there.

What the device half pins, on gpt2:

- placement: weights, captures and lens read-outs stay on ``tpu`` rather than round-tripping
  through the host, which would break a downstream SAE matmul the same way it would on CUDA;
- numerics: under ``torch.set_float32_matmul_precision("highest")`` a TPU float32 capture agrees
  with a CPU float32 capture of the same prompt. TorchTPU's default ("medium") does not and is not
  expected to, which is what the load warning is for;
- the served configuration: bf16 capture, attention, logit lens and steering through generation.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from harness import GPT2, assert_logit_lens_self_consistent, evict_models, load_model, parity_required

from interp_engine import AddSpec, EagerModel, LayerSteeringSpec, SteeringSpec, capture, select_backend, steer
from interp_engine import model as model_module
from interp_engine.select import _tpu_available

PROMPT = "The capital of France is"
LAYER = 1


# --- weight-free: the fp32 matmul warning -------------------------------------------------------


def _device(kind: str) -> torch.device:
    # The real ``torch.device("tpu")`` only parses once TorchTPU has registered the backend, so the
    # decision is exercised with a stand-in carrying the one attribute it reads.
    return cast(torch.device, SimpleNamespace(type=kind))


@pytest.fixture
def matmul_precision() -> Iterator[None]:
    before = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(before)


@pytest.mark.parametrize("precision", ["medium", "high"])
def test_fp32_on_tpu_below_highest_warns(caplog, matmul_precision, precision):  # noqa: ARG001
    torch.set_float32_matmul_precision(precision)
    with caplog.at_level(logging.WARNING, logger="interp_engine.model"):
        model_module._warn_tpu_fp32_matmul("m", _device("tpu"), torch.float32)
    assert "set_float32_matmul_precision('highest')" in caplog.text


@pytest.mark.parametrize(
    ("kind", "dtype", "precision"),
    [
        ("tpu", torch.float32, "highest"),  # the fix applied
        ("tpu", torch.bfloat16, "medium"),  # bf16 asked for bf16
        ("cuda", torch.float32, "medium"),  # another device's precision is not this warning's business
        ("cpu", torch.float32, "medium"),
    ],
)
def test_fp32_matmul_warning_stays_quiet_otherwise(caplog, matmul_precision, kind, dtype, precision):  # noqa: ARG001
    torch.set_float32_matmul_precision(precision)
    with caplog.at_level(logging.WARNING, logger="interp_engine.model"):
        model_module._warn_tpu_fp32_matmul("m", _device(kind), dtype)
    assert caplog.text == ""


def test_empty_device_cache_runs_on_any_machine():
    model_module._empty_device_cache()


# --- on a TPU -----------------------------------------------------------------------------------

_no_tpu = pytest.mark.skipif(not _tpu_available(), reason="TPU coverage requires TorchTPU and a Cloud TPU")


@pytest.fixture(scope="module")
def _release_tpu() -> Iterator[None]:
    yield
    evict_models()


def _tpu_model(dtype: str) -> EagerModel:
    return load_model(GPT2, device="tpu", dtype=dtype, required=parity_required())


def _ids(model: EagerModel) -> torch.Tensor:
    return model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.float().cpu(), want.float().cpu()
    return ((got - want).norm() / want.norm()).item()


@pytest.mark.tpu
@_no_tpu
def test_auto_selection_picks_the_tpu():
    sel = select_backend(
        GPT2.model_id, requested_device=None, requested_dtype="auto", force_backend=None, vllm_available=False
    )
    assert (sel.backend, sel.device) == ("eager", "tpu")


@pytest.mark.usefixtures("_release_tpu")
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.tpu
@_no_tpu
def test_loads_and_captures_on_the_tpu(dtype: str):
    model = _tpu_model(dtype)
    assert model.device.type == "tpu"
    assert model.dtype == getattr(torch, dtype)
    ids = _ids(model)
    seq = ids.shape[1]
    cache = capture(model, ids, [("resid_post", LAYER), ("mlp_in", LAYER), ("z", LAYER), ("attn_out", LAYER)])
    resid = cache.get("resid_post", LAYER)
    assert resid.shape == (1, seq, model.d_model)
    assert resid.device.type == "tpu", "a capture round-tripped through the host"
    assert torch.isfinite(resid.float().cpu()).all()
    z = cache.get("z", LAYER)
    assert z.shape == (1, seq, model.n_heads * model.head_dim)
    with torch.no_grad():
        reconstructed = model.arch.attn_out_proj(LAYER)(z)
    atol = 1e-3 if dtype == "float32" else 2e-2
    assert torch.allclose(reconstructed.float().cpu(), cache.get("attn_out", LAYER).float().cpu(), atol=atol)


@pytest.mark.usefixtures("_release_tpu")
@pytest.mark.tpu
@_no_tpu
def test_fp32_highest_matches_cpu_fp32(matmul_precision):  # noqa: ARG001
    """The reference claim: at full fp32 matmul precision a TPU capture is a CPU capture."""
    torch.set_float32_matmul_precision("highest")
    tpu = _tpu_model("float32")
    cpu = load_model(GPT2, device="cpu", dtype="float32", required=parity_required())
    points = [("resid_post", layer) for layer in (0, cpu.n_layers // 2, cpu.n_layers - 1)] + [("attn_probs", LAYER)]
    got = capture(tpu, _ids(tpu), points)
    want = capture(cpu, _ids(cpu), points)
    for name, layer in points:
        err = _rel_err(got.get(name, layer), want.get(name, layer))
        assert err < 1e-4, f"{name}@{layer}: relative error {err:.2e} against CPU fp32"
    with torch.no_grad():
        tpu_logits = tpu.hf_model(_ids(tpu)).logits
        cpu_logits = cpu.hf_model(_ids(cpu)).logits
    assert _rel_err(tpu_logits, cpu_logits) < 1e-4
    assert torch.equal(tpu_logits.cpu().argmax(-1), cpu_logits.argmax(-1))


@pytest.mark.usefixtures("_release_tpu")
@pytest.mark.tpu
@_no_tpu
def test_bf16_logit_lens_and_attention_on_the_tpu():
    model = _tpu_model("bfloat16")
    assert_logit_lens_self_consistent(model, PROMPT)
    attn = capture(model, _ids(model), [("attn_probs", LAYER)]).get("attn_probs", LAYER)
    assert attn.device.type == "tpu"
    rowsum = attn.float().cpu().sum(dim=-1)
    assert torch.allclose(rowsum, torch.ones_like(rowsum), atol=2e-2)


@pytest.mark.usefixtures("_release_tpu")
@pytest.mark.tpu
@_no_tpu
def test_steering_reaches_generation_on_the_tpu():
    model = _tpu_model("bfloat16")
    ids = model.to_tokens(PROMPT)[0].tolist()
    vector = torch.ones(model.d_model)
    spec = SteeringSpec(layers={6: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=40.0)])})
    plain = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    with steer(model, spec):
        steered = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    after = asyncio.run(model.generate_text(ids, max_tokens=4, temperature=0.0))
    assert steered != plain
    assert after == plain
