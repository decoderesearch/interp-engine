"""The host half of the static write program: which rows each writer reaches, and which ops a site reads.

A FULL decode graph replays one recording for every step, so a static write has to be data the
kernel reads rather than a branch Python takes. These tests hold the tables the host fills, on CPU.
The kernel that reads them is checked against :func:`apply_ops` in ``test_static_program_gpu.py``.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine.address import Address
from interp_engine.vllm_capture.static import _Site
from interp_engine.vllm_capture.static_program import WRITE_OPS_ENV, WRITE_SLOTS_ENV, StaticWriteProgram

WIDTH = 8
MAX_N = 32


def _site(layer: int = 0, *, streams: int = 0) -> _Site:
    shape = (MAX_N, streams, WIDTH) if streams else (MAX_N, WIDTH)
    return _Site(Address("resid_post", layer), delta=torch.zeros(1, *shape[1:]), shape=shape)


def _add(coeff: float = 1.0, **extra) -> dict:
    return {"op": "additive", "vector": [1.0] * WIDTH, "coeff": coeff, **extra}


def _program(*sites: _Site, slots: int = 4, max_ops: int = 16) -> StaticWriteProgram:
    return StaticWriteProgram(sites=list(sites), device=torch.device("cpu"), max_n=MAX_N, slots=slots, max_ops=max_ops)


def _same(rid: str) -> str:
    return rid


def _slots(program: StaticWriteProgram, n: int) -> list[int]:
    return program.row_map[:n, 0].tolist()


def _globals(program: StaticWriteProgram, n: int) -> list[int]:
    return program.row_map[:n, 1].tolist()


def test_a_request_owns_exactly_its_rows_and_a_cobatched_one_is_left_alone() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=3, generated=True)
    program.fill_rows(["other", "a"], [2, 3], [0, 0], _same)
    slot = program._writers["a"].slot
    assert _slots(program, 5) == [-1, -1, slot, slot, slot]
    start, count = program.prog[0, slot].tolist()
    assert count == 1
    assert program.op_coef[start].tolist()[:4] == [1.0, 0.0, 0.0, 0.0]


def test_a_prompt_only_write_skips_decode_rows_by_position() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=4, generated=False)
    slot = program._writers["a"].slot
    program.fill_rows(["a"], [4], [0], _same)
    assert _slots(program, 4) == [slot] * 4
    program.fill_rows(["a"], [1], [4], _same)
    assert _slots(program, 1) == [-1]


def test_a_one_token_prefill_chunk_is_still_prompt() -> None:
    """A prefix-cache hit can leave one prompt row to compute. That row is prompt, not decode."""
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=6, generated=False)
    program.fill_rows(["a"], [1], [5], _same)
    assert _slots(program, 1) == [program._writers["a"].slot]


def test_skip_positions_hold_across_chunks() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(0, 5), prompt_len=8, generated=True)
    slot = program._writers["a"].slot
    program.fill_rows(["a"], [4], [0], _same)
    assert _slots(program, 4) == [-1, slot, slot, slot]
    program.fill_rows(["a"], [4], [4], _same)
    assert _slots(program, 4) == [slot, -1, slot, slot]


def test_rows_past_the_batch_are_cleared_when_the_batch_shrinks() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=6, generated=True)
    program.fill_rows(["a"], [6], [0], _same)
    program.fill_rows(["a"], [1], [6], _same)
    assert _slots(program, 6)[1:] == [-1] * 5


def test_the_resolver_maps_a_child_id_to_its_registration() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=2, generated=True)
    program.fill_rows(["a-1234abcd"], [2], [0], lambda rid: rid.split("-")[0])
    assert _slots(program, 2) == [program._writers["a"].slot] * 2


def test_the_global_write_yields_a_site_to_any_request_write() -> None:
    """The eager path applies the global write only where no request writes. The tables agree."""
    site, other = _site(0), _site(1)
    program = _program(site, other)
    program.set_global({site: [_add(2.0)], other: [_add(3.0)]}, None)
    glob = program.slots
    assert program.prog[0, glob, 1] == 1 and program.prog[1, glob, 1] == 1
    program.fill_rows(["x"], [3], [0], _same)
    assert _globals(program, 3) == [1, 1, 1]
    program.register("a", {site: [_add()]}, skip=(), prompt_len=3, generated=True)
    assert program.prog[0, glob, 1] == 0 and program.prog[1, glob, 1] == 1
    program.unregister("a")
    assert program.prog[0, glob, 1] == 1


def test_a_scoped_global_write_follows_positions() -> None:
    site = _site()
    program = _program(site)
    program.set_global({site: [_add()]}, {"steer_generated": False, "prompt_len": 3, "skip_positions": [0]})
    program.fill_rows(["x"], [3], [0], _same)
    assert _globals(program, 3) == [0, 1, 1]
    program.fill_rows(["x"], [1], [3], _same)
    assert _globals(program, 1) == [0]


def test_clearing_the_global_write_empties_its_column() -> None:
    site = _site()
    program = _program(site)
    program.set_global({site: [_add()]}, None)
    program.clear_global()
    assert program.prog[0, program.slots].tolist() == [0, 0]
    assert program.idle(site)


def test_unregistering_returns_the_slot_and_the_ops() -> None:
    site = _site()
    program = _program(site, slots=2, max_ops=4)
    for i in range(10):
        program.register(f"r{i}", {site: [_add(), _add()]}, skip=(), prompt_len=1, generated=True)
        program.unregister(f"r{i}")
    assert program.idle(site)
    assert program._free_ops == [(0, 4)]
    assert sorted(program._free_slots) == [0, 1]


def test_re_registering_replaces_rather_than_leaks() -> None:
    site = _site()
    program = _program(site, slots=1, max_ops=2)
    program.register("a", {site: [_add(), _add()]}, skip=(), prompt_len=1, generated=True)
    program.register("a", {site: [_add(), _add()]}, skip=(), prompt_len=1, generated=True)
    assert len(program._writers) == 1


def test_running_out_of_writer_slots_names_the_setting() -> None:
    site = _site()
    program = _program(site, slots=1)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=1, generated=True)
    with pytest.raises(RuntimeError, match=WRITE_SLOTS_ENV):
        program.register("b", {site: [_add()]}, skip=(), prompt_len=1, generated=True)


def test_running_out_of_ops_names_the_setting_and_keeps_the_slot() -> None:
    site = _site()
    program = _program(site, slots=2, max_ops=2)
    with pytest.raises(RuntimeError, match=WRITE_OPS_ENV):
        program.register("a", {site: [_add()] * 3}, skip=(), prompt_len=1, generated=True)
    assert sorted(program._free_slots) == [0, 1]
    assert program._writers == {}


def test_a_vector_of_the_wrong_width_is_refused_at_registration() -> None:
    site = _site()
    program = _program(site)
    with pytest.raises(ValueError, match="width"):
        program.register(
            "a", {site: [{"op": "additive", "vector": [1.0] * 3, "coeff": 1.0}]}, skip=(), prompt_len=1, generated=True
        )


def test_a_stream_is_refused_on_a_point_without_one() -> None:
    site = _site()
    program = _program(site)
    with pytest.raises(ValueError, match="no stream axis"):
        program.register("a", {site: [_add(stream=0)]}, skip=(), prompt_len=1, generated=True)


def test_a_stream_past_the_stack_is_refused() -> None:
    site = _site(streams=4)
    program = _program(site)
    with pytest.raises(ValueError, match="out of range"):
        program.register("a", {site: [_add(stream=4)]}, skip=(), prompt_len=1, generated=True)


def test_without_writers_the_map_is_cleared_once_and_then_left_alone() -> None:
    site = _site()
    program = _program(site)
    program.register("a", {site: [_add()]}, skip=(), prompt_len=4, generated=True)
    program.fill_rows(["a"], [4], [0], _same)
    program.unregister("a")
    program.fill_rows(["a"], [1], [4], _same)
    assert _slots(program, 4) == [-1] * 4
    assert program._last_rows == 0
