"""A module's input is not always ``args[0]``, and the hook substrate must not assume it is.

`register_forward_pre_hook` reports positional and keyword arguments separately, and hands the hook
an empty `args` when the caller passed everything by name. HF does exactly that for attention on
every modern family -- `self.self_attn(hidden_states=..., position_embeddings=...)` -- so a hook
reading `args[0]` observes nothing there. gpt2 passes it positionally, which is why this went
unnoticed: the CPU parity model is the one family the naive version works on.

The read side of that failure is loud (`run_with_cache` raises "Captured nothing"). The write side
was not: a steer on `attn_in` returned the call unchanged and the run completed, reporting success
having steered nothing. That asymmetry is why `write` raises here rather than skipping, and why
these tests assert on the steered *output* rather than on the hook being installed.

The eager and vLLM conventions differ in one respect only -- which positional slot holds the hidden
state, since vLLM's signature is `forward(positions, hidden_states)` -- so both go through
`hooks.hidden_from_call` with different `slot`s, and the last test here pins that the vLLM sibling
keeps its own.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from interp_engine.hooks import HookManager, hidden_from_call


class _Attn(nn.Module):
    """Stands in for an attention sublayer: doubles what it is given, ignores the rest."""

    def forward(self, hidden_states: torch.Tensor, position_embeddings: object = None) -> torch.Tensor:
        return hidden_states * 2


class _KeywordBlock(nn.Module):
    """The HF convention: everything by name (Llama, Qwen3, Gemma-3, OLMo-2, ...)."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(hidden_states=x, position_embeddings=(torch.zeros(1), torch.zeros(1)))


class _PositionalBlock(nn.Module):
    """The gpt2 convention, which the substrate has always handled."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


@pytest.fixture
def x() -> torch.Tensor:
    return torch.arange(6, dtype=torch.float32).reshape(2, 3)


# --- reading -----------------------------------------------------------------


@pytest.mark.parametrize("block_cls", [_KeywordBlock, _PositionalBlock])
def test_an_input_read_sees_the_hidden_state_however_it_was_passed(block_cls: type[nn.Module], x: torch.Tensor):
    block = block_cls()
    seen: list[torch.Tensor] = []
    with HookManager() as hm:
        hm.read(block.attn, seen.append, point="input")
        block(x)
    assert len(seen) == 1
    assert torch.equal(seen[0], x)


def test_a_read_skips_a_call_that_carries_no_tensor():
    """Not every module that resolves to a point necessarily runs with one, and a read that cannot
    find a tensor must observe nothing rather than guess at a non-activation argument. The caller
    that needed it is the one positioned to complain: `run_with_cache` raises "Captured nothing"."""
    module = nn.Identity()
    seen: list[torch.Tensor] = []
    with HookManager() as hm:
        hm.read(module, seen.append, point="input")
        module(input=torch.ones(2))  # not `hidden_states`, so not the point
    assert seen == []


# --- writing -----------------------------------------------------------------


@pytest.mark.parametrize("block_cls", [_KeywordBlock, _PositionalBlock])
def test_a_steer_on_an_input_reaches_the_module_however_it_was_called(block_cls: type[nn.Module], x: torch.Tensor):
    """The regression that matters: the module must *compute* on the replaced tensor.

    Asserting the output is what separates a working steer from one whose hook ran, returned the
    call untouched, and left the forward pass exactly as it found it.
    """
    block = block_cls()
    with HookManager() as hm:
        hm.write(block.attn, lambda t: t + 100.0, point="input")
        out = block(x)
    assert torch.equal(out, (x + 100.0) * 2)


def test_a_steer_that_cannot_find_its_tensor_says_so_instead_of_doing_nothing():
    module = nn.Identity()
    with HookManager() as hm:
        hm.write(module, lambda t: t + 1.0, point="input")
        with pytest.raises(ValueError, match="no tensor argument"):
            module(input=torch.ones(2))


# --- the two conventions -----------------------------------------------------


def test_the_hidden_state_is_found_by_what_it_is_not_by_which_argument_it_is():
    """Every argument order is live -- HF's `forward(hidden_states, ...)`, vLLM's
    `forward(positions, hidden_states)`, and gpt-oss reversing vLLM's own convention -- so a fixed
    slot is wrong for some family whichever one it picks. Rank and dtype tell them apart:
    `positions` is a 1-D index vector, the hidden state carries a model dimension."""
    positions = torch.arange(3)
    hidden = torch.ones(3, 8)

    assert hidden_from_call((hidden, positions), {}) is hidden  # HF, and vLLM's gpt-oss
    assert hidden_from_call((positions, hidden), {}) is hidden  # vLLM's Llama/Qwen/Gemma
    assert hidden_from_call((hidden,), {}) is hidden  # a lone argument, under any signature
    assert hidden_from_call((positions,), {"hidden_states": hidden}) is hidden  # keyword beats both

    # `residual` has the hidden state's shape and dtype, but always follows it.
    residual = torch.full((3, 8), 2.0)
    assert hidden_from_call((positions, hidden, residual), {}) is hidden
    assert hidden_from_call((hidden, positions, residual), {}) is hidden


def test_a_steer_writes_back_to_the_argument_it_read():
    """Writing `args[0]` unconditionally would hand vLLM's Llama signature a hidden state in place of
    its position ids -- shaped wrongly, and nothing about the failure would name steering."""
    from interp_engine.hooks import _replace_hidden_in_call

    positions, hidden = torch.arange(3), torch.ones(3, 8)
    new = torch.zeros(3, 8)

    args, kwargs = _replace_hidden_in_call((positions, hidden), {}, new)
    assert args[0] is positions and args[1] is new and kwargs == {}

    args, kwargs = _replace_hidden_in_call((hidden, positions), {}, new)
    assert args[0] is new and args[1] is positions

    args, kwargs = _replace_hidden_in_call((positions,), {"hidden_states": hidden}, new)
    assert args == (positions,) and kwargs["hidden_states"] is new

    assert _replace_hidden_in_call((), {}, new) is None


def test_the_vllm_hook_shares_the_eager_resolution():
    """The wrapper in `vllm_capture._hooks` exists for its docstring, not for a different rule: both
    engines face the same spread of argument orders, and gpt-oss proves vLLM's own is not uniform."""
    from interp_engine.vllm_capture._hooks import hidden_from_call as vllm_hidden_from_call

    positions, hidden = torch.arange(3), torch.ones(3, 8)
    assert vllm_hidden_from_call((positions, hidden), {}) is hidden
    assert vllm_hidden_from_call((hidden, positions), {}) is hidden
