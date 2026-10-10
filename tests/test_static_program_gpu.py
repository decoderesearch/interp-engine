"""The static write kernel against its CPU twin, and inside a recorded CUDA graph.

The graph case is the defect this program exists for: vLLM's V2 runner records FULL decode graphs
with plain ``torch.cuda.graph`` while nothing is registered, then replays that recording on every
step. A write must still reach the rows registered after the recording, and only those rows.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine.address import Address
from interp_engine.vllm_capture.static import _Site
from interp_engine.vllm_capture.static_program import StaticWriteProgram, apply_ops, compile_op

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA"),
]
pytest.importorskip("triton")

MAX_N = 64
DEV = torch.device("cuda")


def _site(width: int, streams: int = 0) -> _Site:
    shape = (MAX_N, streams, width) if streams else (MAX_N, width)
    return _Site(Address("resid_post", 0), delta=torch.zeros(1, *shape[1:]), shape=shape)


def _specs(width: int, seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    v, t = torch.randn(width, generator=g), torch.randn(width, generator=g)
    return [
        {"op": "additive", "vector": (v * 0.1).tolist(), "coeff": 2.0},
        {"op": "orthogonal", "vector": v.tolist(), "coeff": 0.5},
        {"op": "projection_cap", "vector": t.tolist(), "min": -0.2, "max": 0.3},
        {"op": "norm_scaled_add", "vector": (v / v.norm()).tolist(), "coeff": 0.4, "max_fraction": 0.3},
        {"op": "ablate", "vector": t.tolist()},
        {"op": "swap", "vector": v.tolist(), "target": t.tolist()},
    ]


def _expected(hidden, residual, layout, writers):
    """``hidden`` after the writes, row by row, from :func:`apply_ops` in fp32."""
    out = hidden.float().clone()
    full = out + (residual.float() if residual is not None else 0)
    offset = 0
    for rid, n in layout:
        if rid in writers:
            rows = slice(offset, offset + n)
            out[rows] += apply_ops(full[rows], [compile_op(s) for s in writers[rid]])
        offset += n
    return out


def _tol(dtype: torch.dtype) -> dict:
    return {"rtol": 2e-2, "atol": 5e-2} if dtype == torch.bfloat16 else {"rtol": 1e-4, "atol": 1e-4}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("width", [64, 4096, 5000])
def test_the_kernel_matches_its_cpu_twin(dtype: torch.dtype, fused: bool, width: int) -> None:
    site = _site(width)
    program = StaticWriteProgram(sites=[site], device=DEV, max_n=MAX_N, slots=4, max_ops=64)
    writers = {"a": _specs(width, 1), "b": _specs(width, 2)[::-1]}
    for rid, specs in writers.items():
        program.register(rid, {site: specs}, skip=(), prompt_len=100, generated=True)
    layout = [("a", 5), ("x", 3), ("b", 1), ("a2", 2)]
    program.fill_rows([r for r, _ in layout], [n for _, n in layout], [0, 0, 7, 0], lambda r: r)
    n = sum(m for _, m in layout)
    torch.manual_seed(0)
    hidden = torch.randn(n, width, device=DEV, dtype=dtype)
    residual = torch.randn(n, width, device=DEV, dtype=dtype) if fused else None
    want = _expected(hidden, residual, layout, writers)
    program.launch(site, hidden, residual, n)
    torch.testing.assert_close(hidden.float(), want, **_tol(dtype))


def test_a_stream_op_writes_only_its_stream() -> None:
    width, streams = 128, 4
    site = _site(width, streams)
    program = StaticWriteProgram(sites=[site], device=DEV, max_n=MAX_N, slots=2, max_ops=8)
    specs = [{**_specs(width, 3)[2], "stream": 1}, _specs(width, 3)[0]]
    program.register("a", {site: specs}, skip=(), prompt_len=10, generated=True)
    program.fill_rows(["a"], [3], [0], lambda r: r)
    hidden = torch.randn(3, streams, width, device=DEV)
    want = hidden + apply_ops(hidden, [compile_op(s) for s in specs])
    program.launch(site, hidden, None, 3)
    torch.testing.assert_close(hidden, want, rtol=1e-4, atol=1e-4)


def test_a_strided_view_is_written_in_place() -> None:
    """The value point is a column slice of the fused qkv output: rows are not contiguous."""
    width = 96
    site = _site(width)
    program = StaticWriteProgram(sites=[site], device=DEV, max_n=MAX_N, slots=2, max_ops=8)
    specs = _specs(width, 4)
    program.register("a", {site: specs}, skip=(), prompt_len=10, generated=True)
    program.fill_rows(["a"], [4], [0], lambda r: r)
    fused = torch.randn(4, 3 * width, device=DEV)
    view = fused[:, 2 * width :]
    want = view + apply_ops(view, [compile_op(s) for s in specs])
    untouched = fused[:, : 2 * width].clone()
    program.launch(site, view, None, 4)
    torch.testing.assert_close(view, want, rtol=1e-4, atol=1e-4)
    assert torch.equal(fused[:, : 2 * width], untouched)


def test_a_graph_recorded_with_nothing_registered_replays_later_writes() -> None:
    width, n = 256, 8
    site = _site(width)
    program = StaticWriteProgram(sites=[site], device=DEV, max_n=MAX_N, slots=4, max_ops=32)
    hidden = torch.zeros(n, width, device=DEV)
    source = torch.randn(n, width, device=DEV)

    program.launch(site, hidden, None, n)  # warm the JIT before recording, as vLLM's warmup does
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hidden.copy_(source)
        program.launch(site, hidden, None, n)

    graph.replay()
    torch.testing.assert_close(hidden, source)

    writers = {"a": _specs(width, 5)[:1], "b": _specs(width, 6)}
    program.register("a", {site: writers["a"]}, skip=(), prompt_len=3, generated=False)
    program.register("b", {site: writers["b"]}, skip=(), prompt_len=2, generated=True)
    # One decode row for each: "a" is prompt-only, so only "b"'s row moves.
    program.fill_rows(["a", "b"], [1, 1], [3, 2], lambda r: r)
    graph.replay()
    want = source.clone()
    want[1:2] += apply_ops(source[1:2], [compile_op(s) for s in writers["b"]])
    torch.testing.assert_close(hidden[:2], want[:2], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(hidden[2:], source[2:])

    program.unregister("a")
    program.unregister("b")
    program.fill_rows(["a", "b"], [1, 1], [4, 3], lambda r: r)
    graph.replay()
    torch.testing.assert_close(hidden, source)
