"""Which CUDA-graph path vLLM chose, for the two decisions that depend on the answer.

The rest of this package assumes ``enforce_eager=True``, because a replayed CUDA graph does not
re-execute the Python ``forward`` a ``register_forward_hook`` lives on. vLLM 0.26 complicates that
assumption without lifting it, in a way worth naming here because the flag reads as if it did:

``compilation/breakable_cudagraph.py`` is a graph path that is *not* built on torch.compile. It
stream-captures the real Python forward -- so hooks do fire while it records -- and vLLM auto-enables
it for architectures carrying no ``@support_torch_compile``, ``DeepseekV4ForCausalLM`` among them.
Replay still walks a recorded list of segments rather than the forward, so a hook fires once per
captured shape and never again. Trying to hold that seam open from here does not work; the attempt,
the measurements, and why runtime recapture returns zeroed activations are written up in
``plans/dsv4-capture-under-cudagraphs.md``.

So this module answers two questions and changes nothing else:

- :func:`graphs_replaying` -- is anything in this process replaying graphs? Used to refuse *writes*
  (:func:`refuse_writes_reason`), which fail silently rather than loudly on a replaying engine.
- :func:`graph_debug` -- what did vLLM actually decide? Reported through ``demux_debug``, because
  "capture came back short" and "this engine replays graphs" are the same bug seen from two ends.
"""

from __future__ import annotations

import os
from typing import Any

#: The ``vllm.compilation.breakable_cudagraph`` module, or None on a vLLM too old to have it. Cached
#: because the answer is a property of the installed wheel; whether the path is *enabled* is not
#: cached, because vLLM turns it on while building the engine config and a caller may well ask before
#: that has happened.
_MODULE: Any = None
_LOOKED = False


def _module() -> Any:
    global _MODULE, _LOOKED
    if not _LOOKED:
        _LOOKED = True
        try:
            from vllm.compilation import breakable_cudagraph as bcg  # pyright: ignore[reportMissingImports]

            _MODULE = bcg
        except Exception:  # noqa: BLE001 - an older vLLM has neither the module nor the flag
            _MODULE = None
    return _MODULE


def breakable_graphs_enabled() -> bool:
    """Whether vLLM has selected its breakable-cudagraph path in this process.

    Note what this does *not* say: ``enforce_eager=True`` still leaves the flag on -- vLLM sets it
    from the architecture and separately forces ``cudagraph_mode=NONE`` -- so a true answer here
    means "graphs would be breakable ones", not "graphs are running". Anything that changes
    behaviour on a *replaying* engine wants :func:`graphs_replaying` instead.
    """
    bcg = _module()
    return bool(bcg is not None and bcg.is_breakable_cudagraph_enabled())


def graphs_replaying() -> bool:
    """Whether anything in this process is set up to replay recorded graphs.

    Counts vLLM's own live wrappers rather than inferring from flags, because the flag says less than
    it looks like it does: an ``enforce_eager=True`` engine on DeepSeek-V4 has
    :func:`breakable_graphs_enabled` true *and* ``cudagraph_mode=NONE``, so no wrapper is built and
    hooks fire normally. Refusing steering there -- which works, and is how every steered
    DeepSeek-V4 request has run -- would be a regression dressed as a safety check.

    Counted by instance rather than located on the model runner because where the wrapper sits is
    vLLM's business and is not where its construction reads as if it were: the runner's own ``model``
    attribute is still the bare module on the runner this package sees.
    """
    bcg = _module()
    return bool(bcg is not None and len(bcg.BreakableCUDAGraphWrapper._all_instances) > 0)


def graph_debug(worker: object) -> dict[str, Any]:
    """What graph path this worker is on, for diagnosing a capture that came back short.

    ``cudagraph_mode`` is what vLLM decided and ``replays_graphs`` is what it did about it; they
    disagree in both directions, which is the entire reason both are here.
    """
    compilation = getattr(getattr(worker, "vllm_config", None), "compilation_config", None)
    return {
        "replays_graphs": graphs_replaying(),
        "breakable_enabled": breakable_graphs_enabled(),
        "cudagraph_mode": str(getattr(compilation, "cudagraph_mode", None)),
        "compilation_mode": str(getattr(compilation, "mode", None)),
    }


def refuse_writes_reason(what: str) -> str | None:
    """Why ``what`` cannot be served on this engine, or None when it can.

    ``VLLMModel`` already refuses writes on an engine built with ``enforce_eager=False``, from the
    kwargs and before the engine exists. This is the same refusal for the caller who built their own
    engine and drives the worker methods directly (:mod:`interp_engine.vllm_plugin`), where nothing
    upstream has checked -- and where the failure is the silent one: a hook that never fires writes
    nothing and says nothing, so the request comes back fluent, deterministic and un-steered.
    """
    if not graphs_replaying():
        return None
    from interp_engine.vllm_capture.static import STATIC_ENV, decode_static_env

    parsed = decode_static_env(os.environ.get(STATIC_ENV))
    if parsed is not None and parsed[1]:
        return None
    return (
        f"{what} needs to write a tensor back into the forward, and this engine replays recorded "
        "CUDA graphs instead of running it: the hook would never fire, and the request would return "
        'fluent, un-steered text with nothing to say so. Reload with backend="vllm" to steer '
        'through hooks, or backend="vllm-static" with the write sites named in static_writes= to '
        "steer at graph speed. Plain generation is unaffected."
    )
