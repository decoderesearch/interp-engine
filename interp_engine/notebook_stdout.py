"""Give a notebook kernel's stdout the file descriptor vLLM's engine start needs.

vLLM silences C-level output around ``torch.distributed.new_group`` by dup'ing over a
descriptor (``suppress_stdout`` in ``vllm/utils/system_utils.py``), and its EngineCore
child is forked, so the child inherits whichever ``sys.stdout`` the parent had. Under
Jupyter that is ``ipykernel.iostream.OutStream``, which writes through a ZMQ socket and
raises ``io.UnsupportedOperation`` from ``fileno()``. The child then dies before it loads
anything, and what reaches the caller is vLLM's ``Engine core initialization failed. See
root cause above.`` -- above being a traceback in another process, about a descriptor.

A descriptor is what that stream is supposed to hand back: ipykernel answers ``fileno()``
with ``_original_stdstream_copy``, a dup of the real stdout it took over, whenever the
kernel was started capturing the low-level ones. This supplies the one a kernel that was
not started that way -- Colab's -- did not keep. Colab is the case that matters, because
this repo's own notebook templates run there.

The descriptor stays on the stream rather than being restored after the engine is built.
The fork is not the only caller: vLLM suppresses stdout the same way around its stateless
process groups, from whichever process reaches one.

Forcing ``spawn`` would also give the child a stdout of its own, and is what
``VLLM_WORKER_MULTIPROC_METHOD`` is for. It is not what this does, because a spawned
child writes to the kernel's real stdout rather than to the cell -- which takes the engine
logs, and the root cause behind that message, out of the notebook with it.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def _answers_fileno(stream: object) -> bool:
    """Whether ``stream.fileno()`` returns rather than raises.

    ``io.UnsupportedOperation`` is both an ``OSError`` and a ``ValueError``, which is what
    a kernel stream raises; a detached or closed one raises ``ValueError`` on its own.
    """
    try:
        stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return False
    return True


def _process_stdout_fd() -> int | None:
    """The descriptor this process's stdout is, or None when it has none.

    ``sys.__stdout__`` is the stream Python opened at startup, which Jupyter leaves alone
    when it replaces ``sys.stdout``, so it is the first place to ask. Fd 1 covers an
    embedding that replaced both. Either is confirmed with ``fstat`` before being handed
    out, since a daemonized process may have closed it.
    """
    original = sys.__stdout__
    fd = original.fileno() if original is not None and _answers_fileno(original) else 1
    try:
        os.fstat(fd)
    except OSError:
        return None
    return fd


def ensure_stdout_descriptor() -> bool:
    """Give ``sys.stdout`` a ``fileno()`` if it has none, and say whether it needed one.

    Idempotent, and a no-op outside a notebook. A stream that already answers is left
    alone, and so is a process whose stdout is closed: there is no descriptor to offer
    there, and vLLM's refusal is then the honest outcome rather than one to work around.
    """
    stream = sys.stdout
    if stream is None or _answers_fileno(stream):
        return False
    fd = _process_stdout_fd()
    if fd is None:
        return False
    try:
        stream.fileno = lambda: fd  # type: ignore[method-assign]
    except (AttributeError, TypeError):
        # A stream that takes no new attribute, which a C-level or slotted one does not.
        # Nothing is lost: this is the refusal vLLM was about to raise anyway.
        return False
    logger.info(
        "%s has no file descriptor, which vLLM's engine start requires; answering fileno() with fd %d.",
        type(stream).__name__,
        fd,
    )
    return True
