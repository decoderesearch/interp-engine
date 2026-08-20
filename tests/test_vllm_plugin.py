"""The worker-extension plugin, and the name-based RPC that depends on it.

``VLLMModel`` invokes its worker hooks by NAME (``collective_rpc("register_capture", ...)``),
which is what lets it run without ``VLLM_ALLOW_INSECURE_SERIALIZATION``. A typo in one of
those strings is invisible until a GPU actually runs the request and vLLM reports an unknown
method, so the load-bearing test here reads every name out of ``vllm_backend.py`` and checks
it against the extension class. That runs anywhere, with or without vLLM installed.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

from interp_engine import WORKER_EXTENSION_CLS, InterpWorkerExtension, capture_engine_kwargs, vllm_backend, vllm_capture

_BACKEND_SOURCE = pathlib.Path(inspect.getfile(vllm_backend)).read_text()


def _extension_methods() -> set[str]:
    return {name for name in dir(InterpWorkerExtension) if not name.startswith("_")}


def test_worker_extension_cls_path_resolves_to_the_class() -> None:
    """vLLM resolves this string inside each worker process, so it must stay importable."""
    module_path, _, class_name = WORKER_EXTENSION_CLS.rpartition(".")
    module = __import__(module_path, fromlist=[class_name])
    assert getattr(module, class_name) is InterpWorkerExtension


def test_every_rpc_method_name_exists_on_the_extension() -> None:
    """Each ``collective_rpc("name", ...)`` in the backend must name a real method."""
    called = set(re.findall(r'collective_rpc\(\s*"([a-z_]+)"', _BACKEND_SOURCE))
    assert called, "found no name-based collective_rpc calls -- did the call style change?"
    assert called <= _extension_methods(), f"unknown RPC names: {sorted(called - _extension_methods())}"


def test_backend_passes_the_extension_to_the_engine() -> None:
    """Without this kwarg the by-name RPCs above have nothing to dispatch to."""
    assert '"worker_extension_cls": WORKER_EXTENSION_CLS' in _BACKEND_SOURCE


def test_backend_does_not_set_insecure_serialization() -> None:
    """The whole point of the extension: no pickled callables, so no opt-in to unpickling.

    Matches the flag being *set* rather than merely named, since the `worker_extension_cls` kwarg
    is commented with the reason the flag is unnecessary.
    """
    set_flag = re.search(
        r'VLLM_ALLOW_INSECURE_SERIALIZATION"\s*[,\]]|VLLM_ALLOW_INSECURE_SERIALIZATION=1"', _BACKEND_SOURCE
    )
    assert set_flag is None, f"backend still sets the flag: {set_flag.group() if set_flag else ''}"


def test_extension_methods_delegate_to_worker_functions() -> None:
    """Every public method forwards to a ``worker_*`` function of the same name.

    Keeps the two surfaces from drifting: a new worker function exposed under a renamed
    method, or a method that quietly grows its own implementation, both fail here.
    """
    for name in _extension_methods():
        func = f"worker_{name}"
        assert hasattr(vllm_capture, func), f"{name}() has no matching vllm_capture.{func}"
        body = inspect.getsource(getattr(InterpWorkerExtension, name))
        assert f"{func}(self" in body, f"{name}() does not delegate to {func}"


def test_extension_method_signatures_match_their_worker_functions() -> None:
    """Same parameters in the same order, minus the worker/self argument."""
    for name in _extension_methods():
        method_params = list(inspect.signature(getattr(InterpWorkerExtension, name)).parameters)
        worker_params = list(inspect.signature(getattr(vllm_capture, f"worker_{name}")).parameters)
        assert method_params[1:] == worker_params[1:], f"{name}: {method_params[1:]} != {worker_params[1:]}"


def test_extension_exposes_no_non_method_attributes() -> None:
    """vLLM asserts the extension shares no attribute name with its Worker class, and it
    checks single-underscore names too -- so anything but plain methods here is a hazard."""
    for name in _extension_methods():
        assert callable(getattr(InterpWorkerExtension, name)), f"{name} is not a method"
    assert not [n for n in vars(InterpWorkerExtension) if n.startswith("_") and not n.startswith("__")]


def test_capture_engine_kwargs_disables_the_two_things_that_break_capture() -> None:
    kwargs = capture_engine_kwargs()
    assert kwargs["enforce_eager"] is True, "CUDA graphs skip Python forward hooks"
    assert kwargs["enable_prefix_caching"] is False, "cached positions are never forwarded"


def test_module_docstring_example_is_syntactically_valid() -> None:
    """The plugin's docstring is the tier-1 onboarding path, so its example must parse."""
    from interp_engine import vllm_plugin

    doc = vllm_plugin.__doc__ or ""
    example = "\n".join(line[4:] for line in doc.splitlines() if line.startswith("    "))
    ast.parse(example)


# The other half of this contract -- that no method name collides with vLLM's own Worker,
# which vLLM hard-asserts at worker init -- needs vLLM importable, so it lives in
# apps/inference/tests/unit/test_vllm_worker_extension.py where the venv has it.
