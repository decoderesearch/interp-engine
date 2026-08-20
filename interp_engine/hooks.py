"""Low-level forward-hook substrate.

This is the single primitive underneath capture, steer, and any future probe/monitor.
Two operations, deliberately symmetric (reading a vector is the sibling of writing one):

- :meth:`HookManager.read`  — observe a tensor flowing through a module (no mutation).
- :meth:`HookManager.write` — replace a tensor flowing through a module.

Both work at a module's ``"input"`` (via ``register_forward_pre_hook``) or ``"output"``
(via ``register_forward_hook``). HF decoder layers return a tuple ``(hidden_states, ...)``;
we transparently extract/replace element 0 so callers always deal with a plain tensor.

Never imports from ``neuronpedia_inference``.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType

import torch
import torch.nn as nn


def extract_hidden(output: object, index: int = 0) -> torch.Tensor:
    """Pull a tensor out of a module output (tuple or bare tensor).

    ``index`` selects which element of a tuple output, for the modules whose *other* return values
    are themselves points: an MoE router returns ``(router_logits, weights, indices)``, all three of
    which are things a caller asks for by name. Default 0 is the hidden state, which is what every
    other point wants and the only element a bare-tensor output can offer.
    """
    if isinstance(output, tuple | list):
        if index >= len(output):
            raise ValueError(
                f"Asked for element {index} of a {type(output).__name__} of {len(output)} from "
                f"{'a module' if not output else type(output[0]).__name__}: this module does not "
                "return that value. The point's definition and this checkpoint's module disagree."
            )
        return output[index]
    if index:
        raise ValueError(
            f"Asked for element {index} of a module output that is a bare "
            f"{type(output).__name__}, not a tuple, so only element 0 exists."
        )
    return output  # type: ignore[return-value]


def parse_point(point: str) -> tuple[str, int]:
    """Split a hook side into ``(side, tuple index)``: ``"output"`` -> ``("output", 0)``,
    ``"output:2"`` -> ``("output", 2)``.

    The index rides on the side string rather than becoming a third element of
    ``resolve_point``'s return, because it is the same *kind* of information -- where on the module
    to read -- and every existing caller passes that around as one value.
    """
    side, _, index = point.partition(":")
    if side not in ("input", "output"):
        raise ValueError(f"point must be 'input' or 'output' (optionally 'output:N'), got {point!r}")
    if index and side == "input":
        raise ValueError(f"only a module's output can be indexed; got {point!r}")
    return side, int(index or 0)


def replace_hidden(output: object, new: torch.Tensor) -> object:
    """Rebuild a module output with the hidden-state tensor replaced."""
    if isinstance(output, tuple):
        return (new, *output[1:])
    if isinstance(output, list):
        return [new, *output[1:]]
    return new


def hidden_arg_index(args: tuple) -> int | None:
    """Which positional argument is the hidden state, decided by what it is rather than where it sits.

    Position cannot decide it. HF puts the hidden state first, vLLM's Llama/Qwen/Gemma signature is
    ``forward(positions, hidden_states)`` and puts it second, and vLLM's gpt-oss reverses its own
    convention back to ``forward(hidden_states, positions)``. A fixed index is therefore wrong for
    *some* family whichever index it picks, and picking wrong is silent: ``positions`` is a tensor
    too, so a capture keyed on the second argument returned ``[0, 1, 2, ...]`` for gpt-oss's
    ``attn_in`` and a steering write would have overwritten the position ids.

    The hidden state is the only argument that is floating-point and carries a model dimension, so
    that is what this looks for: rank >= 2 (``(tokens, d_model)`` on vLLM, ``(batch, seq, d_model)``
    on HF). ``positions`` is 1-D and integral, and the masks and caches that share these signatures
    either come after it or are not bare tensors. ``residual`` is shaped like the hidden state but
    always follows it, so first-match is the hidden state. Falls back to any floating-point tensor
    when nothing has rank 2, and returns None when no argument is a tensor at all.
    """
    floats = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor) and a.is_floating_point()]
    for i in floats:
        if args[i].ndim >= 2:
            return i
    return floats[0] if floats else None


def hidden_from_call(args: tuple, kwargs: dict) -> torch.Tensor | None:
    """The hidden-state argument of a submodule call, whichever way the caller passed it.

    A module's input is only ``args[0]`` when the caller happens to use positional arguments, and
    the callers do not agree. HF calls ``self.self_attn(hidden_states=..., position_embeddings=...)``
    by keyword on every modern family (gpt2, which passes it positionally, is the exception), so
    reading ``args[0]`` sees an empty tuple and silently observes nothing. When the call *is*
    positional, :func:`hidden_arg_index` decides which argument it is. Returns None when neither
    spelling yields a tensor, leaving "and therefore what?" to the caller -- a read skips, a write
    refuses.
    """
    found = kwargs.get("hidden_states")
    if isinstance(found, torch.Tensor):
        return found
    index = hidden_arg_index(args)
    return None if index is None else args[index]


def _replace_hidden_in_call(args: tuple, kwargs: dict, new: torch.Tensor) -> tuple[tuple, dict] | None:
    """Put ``new`` back where :func:`hidden_from_call` found the tensor it replaces.

    The write has to land on the argument the read came from, not on ``args[0]``: where the hidden
    state is second (vLLM's Llama signature) writing the first argument replaces ``positions``.
    """
    if isinstance(kwargs.get("hidden_states"), torch.Tensor):
        return args, {**kwargs, "hidden_states": new}
    index = hidden_arg_index(args)
    if index is None:
        return None
    return (*args[:index], new, *args[index + 1 :]), kwargs


ReadFn = Callable[[torch.Tensor], None]
WriteFn = Callable[[torch.Tensor], torch.Tensor]


class HookManager:
    """Context manager owning a set of forward hooks; removes them all on exit."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # --- reads ---------------------------------------------------------------
    def read(self, module: nn.Module, fn: ReadFn, *, point: str = "output") -> None:
        side, index = parse_point(point)
        if side == "output":

            def _hook(_m: nn.Module, _args: tuple, output: object) -> None:
                fn(extract_hidden(output, index))

            self._handles.append(module.register_forward_hook(_hook))
        else:

            def _pre(_m: nn.Module, args: tuple, kwargs: dict) -> None:
                hidden = hidden_from_call(args, kwargs)
                if hidden is not None:
                    fn(hidden)

            self._handles.append(module.register_forward_pre_hook(_pre, with_kwargs=True))

    # --- writes --------------------------------------------------------------
    def write(self, module: nn.Module, fn: WriteFn, *, point: str = "output") -> None:
        if ":" in point:
            raise ValueError(
                f"cannot write to {point!r}: a write replaces the module's hidden state (element 0). "
                "Editing another element of the output would change what the module *reported*, not "
                "what it computed -- an MoE router's expert choice is already made by then."
            )
        if point == "output":

            def _hook(_m: nn.Module, _args: tuple, output: object) -> object:
                return replace_hidden(output, fn(extract_hidden(output)))

            self._handles.append(module.register_forward_hook(_hook))
        elif point == "input":

            def _pre(_m: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
                hidden = hidden_from_call(args, kwargs)
                replaced = None if hidden is None else _replace_hidden_in_call(args, kwargs, fn(hidden))
                if replaced is None:
                    # Returning the call unchanged would steer nothing while reporting success, which
                    # is the one failure mode a steering caller cannot detect downstream.
                    raise ValueError(
                        f"Cannot steer the input of {type(_m).__name__}: it was called with no "
                        f"tensor argument (positional {[type(a).__name__ for a in args]}, keyword "
                        f"{sorted(kwargs)}), so there is no hidden state to replace."
                    )
                return replaced

            self._handles.append(module.register_forward_pre_hook(_pre, with_kwargs=True))
        else:
            raise ValueError(f"point must be 'input' or 'output', got {point!r}")

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> HookManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.remove_all()
