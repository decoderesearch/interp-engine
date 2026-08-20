"""Interpreter-startup shim to inject neuronpedia hooks into SGLang's scheduler subprocess.

Python imports a module named ``sitecustomize`` at startup if one is on ``sys.path``. SGLang
spawns its scheduler as a fresh process that inherits the parent's ``PYTHONPATH``/env, so by
prepending *this* directory to ``PYTHONPATH`` (done in ``comparison/engines/sglang_engine.py``,
only for the capture run) the scheduler child runs this at startup and patches model loading.

Strictly gated: a no-op unless ``IE_SGLANG_CAPTURE_DIR`` (read) or ``IE_SGLANG_STEER`` (Tier-2
write) is set, so it never affects unrelated processes that happen to share the path.

Crucially the patch is installed *lazily*: SGLang spawns several subprocesses (scheduler,
detokenizer, ...) and only the model worker imports ``model_runner`` (which pulls in torch/CUDA).
Importing that eagerly here would stall the lightweight processes, so instead we wrap
``builtins.__import__`` cheaply and patch only once ``model_runner`` actually appears, then unwrap.
"""

import os

if os.environ.get("IE_SGLANG_CAPTURE_DIR") or os.environ.get("IE_SGLANG_STEER"):
    import builtins
    import sys

    _TARGET = "sglang.srt.model_executor.model_runner"
    _real_import = builtins.__import__

    def _import_hook(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        target = sys.modules.get(_TARGET)
        # Wait until the module is *fully* initialized (has ModelRunner), not just present in
        # sys.modules mid-bootstrap — otherwise the import below hits a circular import.
        if target is not None and hasattr(target, "ModelRunner"):
            builtins.__import__ = _real_import  # patch once, then get out of the hot path
            try:
                # SGLang's spawn can rebuild sys.path, so make sure this dir (holding the hook
                # module) is importable before we pull it in.
                _here = os.path.dirname(os.path.abspath(__file__))
                if _here not in sys.path:
                    sys.path.insert(0, _here)
                import sglang_hooks

                sglang_hooks.patch_model_runner()
            except Exception as exc:  # noqa: BLE001
                print(f"[sitecustomize] sglang_hooks patch failed: {exc}", file=sys.stderr, flush=True)
        return module

    builtins.__import__ = _import_hook
