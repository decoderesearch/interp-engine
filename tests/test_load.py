"""Unit tests for the ``load_model`` factory's routing.

The backends are patched out so the routing (which class, which kwargs) is exercised
without downloading weights or needing CUDA/vLLM.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from transformers import CLIPConfig, LlamaConfig

from interp_engine import load, model, vllm_backend
from interp_engine.select import BackendSelection


class _Spy:
    """Records the args it was constructed with, standing in for a backend class."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs


def _load(hf_model_id: str = "openai-community/gpt2", **kwargs: Any) -> _Spy:
    """Call ``load_model`` with both backends replaced by spies, on a box that has vLLM.

    Both ``vllm_installed`` references are patched because they answer for different callers: the
    one in ``load`` feeds the auto ladder, while ``require_vllm`` reads the one in ``vllm_backend``,
    where it is defined. Patching only the first would refuse every ``backend="vllm"`` case below on
    a checkout without the extra -- which is most of them, and every CPU dev box.
    """
    with (
        patch.object(load, "EagerModel", _Spy),
        patch.object(load, "VLLMModel", _Spy),
        patch.object(load, "vllm_installed", return_value=True),
        patch.object(vllm_backend, "vllm_installed", return_value=True),
    ):
        return load.load_model(hf_model_id, **kwargs)  # type: ignore[return-value]


# --- validation --------------------------------------------------------------


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        _load(backend="sglang")


def test_vllm_without_vllm_installed_raises_runtime_error():
    with (
        patch.object(vllm_backend, "vllm_installed", return_value=False),
        pytest.raises(RuntimeError, match="vLLM is not installed"),
    ):
        load.load_model("openai-community/gpt2", backend="vllm")


def test_vllm_unavailable_message_points_at_the_eager_escape_hatch():
    """The error has to tell a macOS user what to do instead, not just what failed."""
    with (
        patch.object(vllm_backend, "vllm_installed", return_value=False),
        pytest.raises(RuntimeError) as excinfo,
    ):
        load.load_model("openai-community/gpt2", backend="vllm")
    msg = str(excinfo.value)
    assert "interp-engine[vllm]" in msg
    assert "backend='eager'" in msg


def test_the_refusal_names_the_model_it_was_asked_for():
    """``backend='vllm'`` is usually set far from the call, so the message has to say which load
    it stopped -- a server loading several models would otherwise report only that one of them
    wanted vLLM."""
    with (
        patch.object(vllm_backend, "vllm_installed", return_value=False),
        pytest.raises(RuntimeError, match="openai-community/gpt2"),
    ):
        load.load_model("openai-community/gpt2", backend="vllm")


def test_constructing_the_backend_directly_refuses_the_same_way():
    """The guard is in the constructor as well, because ``load_model`` is not the only way in.

    Without it the absence surfaces from inside ``_ensure_engine`` on the first ``await``, as a bare
    ``ModuleNotFoundError`` several frames deep on a background loop thread -- and only after the
    constructor has downloaded a tokenizer and a config to get there.
    """
    with (
        patch.object(vllm_backend, "vllm_installed", return_value=False),
        pytest.raises(RuntimeError) as excinfo,
    ):
        vllm_backend.VLLMModel("openai-community/gpt2")
    msg = str(excinfo.value)
    assert "vLLM is not installed" in msg
    assert "interp-engine[vllm]" in msg
    assert "backend='eager'" in msg


# --- explicit backends -------------------------------------------------------


def test_explicit_eager_does_not_consult_the_selector():
    with patch.object(load, "select_backend") as sel:
        m = _load(backend="eager", device="cpu")
    sel.assert_not_called()
    assert m.kwargs["device"] == "cpu"


def test_explicit_vllm_does_not_consult_the_selector():
    with patch.object(load, "select_backend") as sel:
        m = _load(backend="vllm")
    sel.assert_not_called()
    assert m.kwargs["tensor_parallel_size"] == 1


# --- auto routing ------------------------------------------------------------


def _selection(*, use_vllm: bool, device: str = "cuda", dtype: str = "auto"):
    return BackendSelection(use_vllm=use_vllm, device=device, dtype=dtype, reason="test")


def test_auto_routes_to_vllm_when_the_ladder_says_so():
    with patch.object(load, "select_backend", return_value=_selection(use_vllm=True)):
        m = _load()
    assert m.args == ("openai-community/gpt2",)
    assert "device" not in m.kwargs  # vLLM always initializes on CUDA


def test_auto_routes_to_eager_when_the_ladder_says_so():
    with patch.object(load, "select_backend", return_value=_selection(use_vllm=False, device="mps", dtype="float16")):
        m = _load()
    assert m.kwargs["device"] == "mps"
    assert m.kwargs["dtype"] == "float16"


def test_auto_forwards_the_caller_device_into_the_ladder():
    with patch.object(load, "select_backend", return_value=_selection(use_vllm=False)) as sel:
        _load(device="cpu", dtype="bfloat16")
    assert sel.call_args.kwargs["requested_device"] == "cpu"
    assert sel.call_args.kwargs["requested_dtype"] == "bfloat16"
    # load_model never force-routes; backend="auto" means the ladder decides freely.
    assert sel.call_args.kwargs["force_backend"] is None


# --- multi-GPU ---------------------------------------------------------------


def test_num_gpus_becomes_tensor_parallel_size_on_vllm():
    m = _load(backend="vllm", num_gpus=4)
    assert m.kwargs["tensor_parallel_size"] == 4


def test_num_gpus_becomes_device_map_auto_on_eager():
    m = _load(backend="eager", device="cuda", num_gpus=4)
    assert m.kwargs["device_map"] == "auto"
    # Must be None, or the .to(device) in EagerModel would fight accelerate's placement.
    assert m.kwargs["device"] is None


def test_single_gpu_eager_keeps_the_explicit_device_and_no_device_map():
    m = _load(backend="eager", device="cuda", num_gpus=1)
    assert m.kwargs["device_map"] is None
    assert m.kwargs["device"] == "cuda"


def test_num_gpus_is_floored_at_one():
    m = _load(backend="eager", num_gpus=0)
    assert m.kwargs["device_map"] is None


# --- kwargs passthrough ------------------------------------------------------


def test_eager_defaults_to_eager_attention_for_attn_probs():
    m = _load(backend="eager")
    assert m.kwargs["attn_implementation"] == "eager"


def test_caller_can_override_eager_attention():
    m = _load(backend="eager", attn_implementation="sdpa")
    assert m.kwargs["attn_implementation"] == "sdpa"


def test_caller_supplied_device_map_wins_over_num_gpus():
    m = _load(backend="eager", num_gpus=4, device_map={"": 0})
    assert m.kwargs["device_map"] == {"": 0}


def test_backend_kwargs_reach_the_vllm_constructor():
    m = _load(backend="vllm", gpu_memory_utilization=0.5, enforce_eager=False)
    assert m.kwargs["gpu_memory_utilization"] == 0.5
    assert m.kwargs["enforce_eager"] is False
    assert m.kwargs["static_points"] is None


def test_static_points_reach_the_vllm_constructor():
    from interp_engine.address import Address

    points = [Address("resid_post", 0)]
    m = _load(backend="vllm-static", static_points=points)
    assert m.kwargs["static_points"] is points
    assert m.kwargs["static_writes"] is None
    assert m.kwargs["enforce_eager"] is False


def test_the_static_backend_defaults_its_tap_set_to_auto():
    """Naming the backend is the whole request; the set it implies is the useful default."""
    m = _load(backend="vllm-static")
    assert m.kwargs["static_points"] == "auto"
    assert m.kwargs["enforce_eager"] is False


@pytest.mark.parametrize("backend", ["eager", "vllm", "vllm-generate", "auto"])
def test_static_points_off_the_static_backend_is_refused(backend):
    """A tap set is how you ask for `vllm-static`, so it never silently turns graphs on."""
    with pytest.raises(ValueError, match="backend='vllm-static'"):
        _load(backend=backend, static_points="auto")


def test_static_writes_off_the_static_backend_names_itself():
    with pytest.raises(ValueError, match="static_writes="):
        _load(backend="vllm", static_writes=[])


def test_a_static_backend_declaring_no_taps_is_refused():
    """`static_points=[]` is generation-only wearing the static engine's name."""
    with pytest.raises(ValueError, match="vllm-generate"):
        _load(backend="vllm-static", static_points=[])


def test_reads_without_write_buffers_is_still_a_static_engine():
    """`static_writes=[]` narrows `auto` rather than emptying it, so it must not be refused."""
    m = _load(backend="vllm-static", static_points="auto", static_writes=[])
    assert m.kwargs["static_points"] == "auto"
    assert m.kwargs["static_writes"] == []


def test_the_generate_backend_declares_the_empty_set():
    """The empty set is what tells the backend graphs-with-no-wraps, and keeps inductor on."""
    m = _load(backend="vllm-generate")
    assert m.kwargs["static_points"] == []
    assert m.kwargs["static_writes"] is None
    assert m.kwargs["enforce_eager"] is False


@pytest.mark.parametrize("backend", ["vllm-static", "vllm-generate"])
def test_enforce_eager_true_against_a_graph_backend_is_refused(backend):
    with pytest.raises(ValueError, match="enforce_eager"):
        _load(backend=backend, enforce_eager=True)


def test_trust_remote_code_reaches_both_backends():
    assert _load(backend="eager", trust_remote_code=False).kwargs["trust_remote_code"] is False
    assert _load(backend="vllm", trust_remote_code=False).kwargs["trust_remote_code"] is False


def test_the_default_reaches_eager_unresolved_and_vllm_as_true():
    """vLLM never runs a checkpoint's *transformers* modeling code — its loader resolves against its
    own tree — so native-vs-bundled is the eager loader's question, and only it gets the unresolved
    ``None``. Handing vLLM anything but ``True`` here would change which checkpoints it accepts."""
    assert _load(backend="eager").kwargs["trust_remote_code"] is None
    assert _load(backend="vllm").kwargs["trust_remote_code"] is True


# --- native code vs. the copy bundled with the checkpoint --------------------


def _resolved(config: Any) -> bool:
    """``resolve_trust_remote_code`` for a checkpoint whose no-remote-code config load returns
    ``config`` — or raises it, which is what transformers does for a config class it lacks."""
    outcome = {"side_effect": config} if isinstance(config, Exception) else {"return_value": config}
    with patch.object(model.AutoConfig, "from_pretrained", **outcome):
        return model.resolve_trust_remote_code("some-org/some-model", None)


def test_a_checkpoint_transformers_has_a_class_for_is_loaded_natively():
    """When both implementations exist the bundled one is redundant, and it is also the older of the
    two — frozen at whatever transformers shipped with the weights, so it breaks first. All four of
    Phi-3-mini, Nemotron-3-Nano, Phi-mini-MoE and DeepSeek-V2-Lite failed to import this way, each
    differently (a removed `Cache` subclass, a hard `mamba-ssm` dep, `flash_attn`, a moved symbol),
    and all four have a native class that loads the same weights."""
    assert _resolved(LlamaConfig()) is False


def test_a_family_with_no_native_causal_lm_keeps_its_bundled_code():
    """`AutoConfig` succeeding is *not* the question. transformers knowing a config does not mean it
    can build a model from it, which is how Phi-mini-MoE passed a config-only probe while its only
    `PhimoeForCausalLM` was still the checkpoint's."""
    assert _resolved(CLIPConfig()) is True


def test_a_remote_only_config_keeps_its_bundled_code():
    """EXAONE: without remote code transformers cannot read even the config, so there is no native
    implementation to prefer and the checkpoint's own code is the only one there is."""
    assert _resolved(ValueError("contains custom code which must be executed")) is True


def test_an_explicit_request_wins_and_costs_no_config_read():
    """An explicit bool is an instruction rather than a hint — including the `True` that reproduces
    the old default, which is the escape hatch if a native class is ever the worse of the two."""
    with patch.object(model.AutoConfig, "from_pretrained") as from_pretrained:
        assert model.resolve_trust_remote_code("some-org/some-model", True) is True
        assert model.resolve_trust_remote_code("some-org/some-model", False) is False
    assert from_pretrained.call_count == 0


# --- where the weights are put ------------------------------------------------
#
# "Load then `.to(device)`" and "load onto device" reach the same model, but not by the same route:
# the first builds the whole checkpoint in host RAM before the card is asked for anything. A model too
# big for the card is then killed by the host OOM killer with the card still empty, so a card-sized
# overrun cannot be caught and leaves no traceback. It is worse for a quantized checkpoint, where the
# route also changes the weights: a quantizer with no kernels for the device it loads onto dequantizes
# to `dtype`, and CPU is such a device for every FP8 scheme, so DeepSeek-V4-Flash reaches ~285 GiB of
# bf16 on the way to a card that holds its 156 GiB of FP8 with room to spare. What decides it is a
# `device_map` in the load kwargs, which is what these read.


class _StopAfterKwargs(Exception):
    """Raised from the patched loader once the kwargs are in hand.

    The kwargs *are* the decision, and everything in ``__init__`` after the load (tokenizer, arch
    resolution) needs a real module tree. So this stops there rather than stubbing all of it.
    """


def _placement(*, quantized: bool, **kwargs: Any) -> dict[str, Any]:
    """The load kwargs ``EagerModel`` builds for a checkpoint that is / isn't quantized."""
    seen: dict[str, Any] = {}

    def fake_load(hf_model_id: str, load_kwargs: dict[str, Any], **_: Any):
        seen.update(load_kwargs)
        raise _StopAfterKwargs

    config = LlamaConfig()
    if quantized:
        config.quantization_config = {"quant_method": "fp8"}
    with (
        patch.object(model, "_load_hf_model", fake_load),
        patch.object(model.AutoConfig, "from_pretrained", return_value=config),
        pytest.raises(_StopAfterKwargs),
    ):
        model.EagerModel("some-org/some-model", trust_remote_code=False, **kwargs)
    return seen


def test_a_quantized_checkpoint_is_placed_at_load_time():
    assert _placement(quantized=True, device="cuda")["device_map"] == "cuda"


def test_an_unquantized_checkpoint_is_placed_at_load_time_too():
    """Not for the dtype reason above but for a host-RAM one, so it applies to every checkpoint.

    Loading on the CPU and moving afterwards asks host RAM to hold the model before the card is asked
    for anything, so float32 gemma-3-12b (45.4 GiB) was killed by the host OOM killer on a 44.4 GiB
    A40 and again on a 31.3 GiB 5090 -- both times with a measured device peak of 0 bytes, which reads
    as a card bound while bounding only host RAM.
    """
    assert _placement(quantized=False, device="cuda")["device_map"] == "cuda"


def test_a_load_that_names_no_device_asks_for_no_placement():
    """`device=None` means "wherever transformers puts it", and saying so as a `device_map` would pull
    accelerate into a load that has not asked to be placed anywhere."""
    assert "device_map" not in _placement(quantized=True, device=None)
    assert "device_map" not in _placement(quantized=False, device=None)


def test_an_explicit_device_map_is_not_second_guessed():
    """A caller placing the model itself (multi-GPU, or an offload map for a checkpoint bigger than
    the card) has said more than we know here."""
    assert _placement(quantized=True, device=None, device_map="auto")["device_map"] == "auto"


# --- a Hub kernel with no build for this GPU ----------------------------------
#
# `kernels` 0.16.1 refuses a prebuilt kernel whose declared architectures miss the current device.
# transformers' `lazy_load_kernel` does not catch that refusal, so it escapes rather than reaching
# the Triton fallback. The weights load fine -- measured on a B200, DeepSeek-V4-Flash-0731 loads in
# 22s and then dies on the first forward -- so a caller who is not warned at load meets a bare
# RuntimeError about CUDA capabilities partway through a capture, saying nothing about the model
# they asked for or what to do next. Hence a probe at load: the same call the forward makes.

_ARCH_REFUSAL = (
    "Kernel 'deep-gemm' variant 'torch214-cxx11-cu130-x86_64-linux' does not support the current "
    "device: CUDA capability 10.0 of the current device is not supported by the architectures of "
    "the build: 9.0a."
)


class _StopAtTokenizer(Exception):
    """Raised from the patched tokenizer, one step past the probe: reaching it means the probe passed."""


class _StubModel:
    """Just enough of a loaded model for `__init__` to reach the probe."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def eval(self) -> None: ...

    def requires_grad_(self, _: bool) -> None: ...


def _load_probing(refusal: str | None, *, quant: str | None, probe: Any = None) -> pytest.ExceptionInfo[Exception]:
    """Load a checkpoint of scheme `quant` while the Hub says `refusal`; return what was raised."""
    config = LlamaConfig()
    if quant is not None:
        config.quantization_config = {"quant_method": quant}

    with (
        patch.object(model, "_load_hf_model", lambda *a, **k: _StubModel(config)),
        patch.object(model, "_hub_kernel_arch_refusal", probe or (lambda: refusal)),
        patch.object(model.AutoConfig, "from_pretrained", return_value=config),
        patch.object(model.AutoTokenizer, "from_pretrained", side_effect=_StopAtTokenizer),
        pytest.raises(Exception) as caught,  # noqa: PT011 - the type under test is the assertion
    ):
        model.EagerModel("some-org/some-model", trust_remote_code=False, device="cuda")
    return caught


@pytest.fixture
def on_a_gpu(monkeypatch):
    monkeypatch.setattr(model.torch.cuda, "is_available", lambda: True)


def test_a_kernel_built_for_another_gpu_says_which_model_and_what_to_do(on_a_gpu):
    caught = _load_probing(_ARCH_REFUSAL, quant="fp8")
    assert caught.type is model.HubKernelUnsupported
    message = str(caught.value)
    assert "some-org/some-model" in message
    assert "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1" in message
    assert "grouped_mm" in message
    assert model.TRANSFORMERS_DEEPGEMM_NOTE in message
    assert _ARCH_REFUSAL in message, "the original refusal is evidence, not noise"


def test_a_gpu_the_kernel_does_target_is_not_warned_about(on_a_gpu):
    assert _load_probing(None, quant="fp8").type is _StopAtTokenizer


def test_an_unquantized_checkpoint_never_asks_the_hub(on_a_gpu):
    """Only FP8 reaches for DeepGEMM, and answering the probe pulls a kernel off the Hub."""
    asked = False

    def probe() -> str:
        nonlocal asked
        asked = True
        return _ARCH_REFUSAL

    assert _load_probing(None, quant=None, probe=probe).type is _StopAtTokenizer
    assert not asked


def test_a_caller_who_already_turned_deepgemm_off_is_not_warned_about_it(on_a_gpu, monkeypatch):
    """The refusal only arrives if something still reaches for the kernel, and nothing here does."""
    monkeypatch.setenv(model.DISABLE_DEEPGEMM_LINEAR, "1")
    config = LlamaConfig()
    config.quantization_config = {"quant_method": "fp8"}
    config._experts_implementation = "grouped_mm"

    with (
        patch.object(model, "_load_hf_model", lambda *a, **k: _StubModel(config)),
        patch.object(model, "_hub_kernel_arch_refusal", lambda: _ARCH_REFUSAL),
        patch.object(model.AutoConfig, "from_pretrained", return_value=config),
        patch.object(model.AutoTokenizer, "from_pretrained", side_effect=_StopAtTokenizer),
        pytest.raises(_StopAtTokenizer),
    ):
        model.EagerModel("some-org/some-model", trust_remote_code=False, device="cuda")


def test_half_a_workaround_is_still_warned_about(on_a_gpu, monkeypatch):
    """The linear path is off but the experts are still on DeepGEMM, so the forward still dies."""
    monkeypatch.setenv(model.DISABLE_DEEPGEMM_LINEAR, "1")
    config = LlamaConfig()
    config.quantization_config = {"quant_method": "fp8"}
    config._experts_implementation = "deepgemm"

    with (
        patch.object(model, "_load_hf_model", lambda *a, **k: _StubModel(config)),
        patch.object(model, "_hub_kernel_arch_refusal", lambda: _ARCH_REFUSAL),
        patch.object(model.AutoConfig, "from_pretrained", return_value=config),
        pytest.raises(model.HubKernelUnsupported),
    ):
        model.EagerModel("some-org/some-model", trust_remote_code=False, device="cuda")


# --- the workaround the harnesses opt into -------------------------------------


def _fallback_for(config: Any, *, refusal: str | None = _ARCH_REFUSAL) -> dict[str, Any]:
    with (
        patch.object(model.AutoConfig, "from_pretrained", return_value=config),
        patch.object(model, "_hub_kernel_arch_refusal", lambda: refusal),
    ):
        return model.deepgemm_fallback_kwargs("some-org/some-model")


def _fp8_moe_config() -> Any:
    config = LlamaConfig()
    config.quantization_config = {"quant_method": "fp8"}
    config.num_local_experts = 8
    return config


def test_an_fp8_moe_on_an_unserved_gpu_gets_both_halves(on_a_gpu, monkeypatch):
    monkeypatch.delenv(model.DISABLE_DEEPGEMM_LINEAR, raising=False)
    assert _fallback_for(_fp8_moe_config()) == {"experts_implementation": "grouped_mm"}
    assert os.environ[model.DISABLE_DEEPGEMM_LINEAR] == "1", "the linear half has no kwarg to carry it"


def test_a_dense_fp8_checkpoint_is_not_told_about_experts_it_does_not_have(on_a_gpu, monkeypatch):
    """`experts_implementation` on a dense trunk is a kwarg transformers has nowhere to put."""
    monkeypatch.delenv(model.DISABLE_DEEPGEMM_LINEAR, raising=False)
    config = LlamaConfig()
    config.quantization_config = {"quant_method": "fp8"}
    assert _fallback_for(config) == {}
    assert os.environ[model.DISABLE_DEEPGEMM_LINEAR] == "1", "the linear path is its whole exposure"


def test_a_gpu_the_kernel_does_target_is_left_alone(on_a_gpu, monkeypatch):
    monkeypatch.delenv(model.DISABLE_DEEPGEMM_LINEAR, raising=False)
    assert _fallback_for(_fp8_moe_config(), refusal=None) == {}
    assert model.DISABLE_DEEPGEMM_LINEAR not in os.environ


def test_an_unquantized_checkpoint_is_left_alone(on_a_gpu, monkeypatch):
    monkeypatch.delenv(model.DISABLE_DEEPGEMM_LINEAR, raising=False)
    assert _fallback_for(LlamaConfig()) == {}
    assert model.DISABLE_DEEPGEMM_LINEAR not in os.environ
