"""Static writes as device tables a CUDA graph can replay.

vLLM's V2 runner records FULL decode graphs with plain ``torch.cuda.graph``. The static wraps run
once, at record time, with no request registered, and every decode step replays that recording. A
write decided in Python is therefore lost on decode. Here the decision is data. One Triton kernel
per write site reads three fixed-address tables:

* ``row_map``: per batch row, the writer slot that owns it (or -1), and whether the global write
  reaches it. The host refills it from the demux before each forward.
* ``prog``: per site and writer slot, a ``(start, count)`` range of ops. The last slot is the global
  write, which a site serves only while no request writes there.
* the op tables: per op, the addresses of two fp32 vectors, a stream, and six coefficients. The host
  fills them at each registration RPC.

The graph keeps the pointers; the values change between steps. The kernel takes the place of the one
``add_`` per site the constant write launched before, so decode pays no extra launch.

Every op has one form, ``delta = coef(x) * w``, with
``coef = c0 + c1*|x| + c2*(x.u) + c3*(clamp(x.u, lo, hi) - x.u)``. ``x`` is the residual the op
reads, and ops at one site compose in order. :func:`compile_op` maps each
:class:`~interp_engine.steer_specs.SteerMethod` onto it, and :func:`apply_ops` is the kernel's CPU
twin.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from interp_engine.steer_specs import SteerMethod, steer_method

logger = logging.getLogger(__name__)

#: Writers with their own ops (one per steered request) that one worker holds at once.
WRITE_SLOTS_ENV = "INTERP_ENGINE_STATIC_WRITE_SLOTS"

#: Ops that one worker holds at once, summed over every writer and site.
WRITE_OPS_ENV = "INTERP_ENGINE_STATIC_WRITE_OPS"

_DEFAULT_SLOTS = 1024
_DEFAULT_OPS = 65536
_INF = float("inf")
_MAX_BLOCK = 4096


@dataclass(frozen=True)
class WriteOp:
    """One op in the universal form. ``u`` and ``w`` are fp32 ``[width]``; ``stream`` -1 is every stream."""

    u: torch.Tensor
    w: torch.Tensor
    coef: tuple[float, float, float, float, float, float]
    stream: int = -1


def _unit(vec: torch.Tensor, eps: float) -> torch.Tensor:
    return vec / vec.norm().clamp_min(eps)


def compile_op(spec: dict) -> WriteOp:
    """``spec`` as one :class:`WriteOp`. Case for case the arithmetic of ``steering._make_steer_modifier``."""
    op = steer_method(spec.get("op", SteerMethod.ADDITIVE))
    vec = torch.tensor(spec["vector"], dtype=torch.float32).flatten()
    stream = -1 if spec.get("stream") is None else int(spec["stream"])
    free = (-_INF, _INF)
    match op:
        case SteerMethod.ADDITIVE:
            w = vec * float(spec["coeff"])
            return WriteOp(w, w, (1.0, 0.0, 0.0, 0.0, *free), stream)
        case SteerMethod.ORTHOGONAL:
            unit = _unit(vec, 1e-12)
            return WriteOp(unit, (float(spec["coeff"]) - 1.0) * unit, (0.0, 0.0, 1.0, 0.0, *free), stream)
        case SteerMethod.PROJECTION_CAP:
            unit = _unit(vec, 1e-12)
            lo = -_INF if spec.get("min") is None else float(spec["min"])
            hi = _INF if spec.get("max") is None else float(spec["max"])
            return WriteOp(unit, unit, (0.0, 0.0, 0.0, 1.0, lo, hi), stream)
        case SteerMethod.NORM_SCALED_ADD:
            # strength*|x|*v, capped at max_fraction*|x|. Both scale with |x|, so the cap is a constant.
            strength = float(spec["coeff"])
            max_fraction = float(spec.get("max_fraction", 1.0))
            ratio = abs(strength) * float(vec.norm())
            w = strength * (max_fraction / ratio if ratio > max_fraction else 1.0) * vec
            return WriteOp(w, w, (0.0, 1.0, 0.0, 0.0, *free), stream)
        case SteerMethod.ABLATE:
            unit = _unit(vec, float(spec.get("eps", 1e-12)))
            return WriteOp(unit, -unit, (0.0, 0.0, 1.0, 0.0, *free), stream)
        case SteerMethod.SWAP:
            eps = float(spec.get("eps", 1e-12))
            source = _unit(vec, eps)
            target = _unit(torch.tensor(spec["target"], dtype=torch.float32).flatten(), eps)
            return WriteOp(source, target - source, (0.0, 0.0, 1.0, 0.0, *free), stream)
    raise ValueError(f"static writes have no program for op={op!r}")


def apply_ops(x: torch.Tensor, ops: Sequence[WriteOp]) -> torch.Tensor:
    """The fp32 delta the kernel adds for ``ops`` at one site, over ``x`` of shape ``[..., width]``."""
    out = x.float()
    total = torch.zeros_like(out)
    for op in ops:
        u, w = op.u.to(out.device), op.w.to(out.device)
        c0, c1, c2, c3, lo, hi = op.coef
        proj = (out * u).sum(-1, keepdim=True)
        norm = out.norm(dim=-1, keepdim=True)
        delta = (c0 + c1 * norm + c2 * proj + c3 * (proj.clamp(lo, hi) - proj)) * w
        if op.stream >= 0:
            keep = torch.zeros(out.shape[-2], 1, dtype=out.dtype, device=out.device)
            keep[op.stream] = 1.0
            delta = delta * keep
        out = out + delta
        total = total + delta
    return total


def _active_rows(start: int, n: int, *, prompt_len: int, generated: bool, skip: frozenset[int]) -> np.ndarray:
    """Which of ``n`` rows at absolute positions ``start..start+n`` a scoped write reaches.

    Without a known prompt length, a one-row chunk counts as decode, as the eager-segment path decides.
    """
    keep = np.ones(n, dtype=bool)
    positions = np.arange(start, start + n)
    if not generated:
        keep &= positions < prompt_len if prompt_len > 0 else np.full(n, n > 1)
    if skip:
        keep &= ~np.isin(positions, np.fromiter(skip, dtype=np.int64))
    return keep


@dataclass
class _Writer:
    """One writer's ops: a request's, or the global write's. Holds the vectors the op table points at."""

    slot: int
    base: int
    total: int
    ranges: dict[int, tuple[int, int]]
    prompt_len: int
    generated: bool
    skip: frozenset[int]
    vectors: torch.Tensor | None = None
    unscoped: bool = False

    def rows(self, start: int, n: int) -> np.ndarray:
        if self.unscoped:
            return np.ones(n, dtype=bool)
        return _active_rows(start, n, prompt_len=self.prompt_len, generated=self.generated, skip=self.skip)


@dataclass
class _SiteInfo:
    width: int
    streams: int


@dataclass
class StaticWriteProgram:
    """The device tables one worker's write sites read, and the host code that fills them."""

    sites: list[Any]
    device: torch.device
    max_n: int
    slots: int = 0
    max_ops: int = 0
    _index: dict[int, int] = field(default_factory=dict)
    _info: list[_SiteInfo] = field(default_factory=list)
    _writers: dict[str, _Writer] = field(default_factory=dict)
    _global: _Writer | None = None
    _free_slots: list[int] = field(default_factory=list)
    _free_ops: list[tuple[int, int]] = field(default_factory=list)
    _req_count: list[int] = field(default_factory=list)
    _rid_cache: dict[str, _Writer | None] = field(default_factory=dict)
    _last_rows: int = 0

    def __post_init__(self) -> None:
        self.slots = self.slots or int(os.environ.get(WRITE_SLOTS_ENV, _DEFAULT_SLOTS))
        self.max_ops = self.max_ops or int(os.environ.get(WRITE_OPS_ENV, _DEFAULT_OPS))
        self._index = {id(site): i for i, site in enumerate(self.sites)}
        self._info = [
            _SiteInfo(int(site.shape[-1]), int(site.shape[1]) if len(site.shape) == 3 else 1) for site in self.sites
        ]
        self._free_slots = list(range(self.slots - 1, -1, -1))
        self._free_ops = [(0, self.max_ops)]
        self._req_count = [0] * len(self.sites)
        dev = self.device
        self.prog = torch.zeros(len(self.sites), self.slots + 1, 2, dtype=torch.int32, device=dev)
        self._prog_host = np.zeros((len(self.sites), self.slots + 1, 2), dtype=np.int32)
        self.row_map = torch.zeros(self.max_n, 2, dtype=torch.int32, device=dev)
        self.row_map[:, 0] = -1
        self._rows_host = np.zeros((self.max_n, 2), dtype=np.int32)
        self._rows_host[:, 0] = -1
        self._rows_sent = self._rows_host.copy()
        self.op_vec = torch.zeros(self.max_ops, 2, dtype=torch.int64, device=dev)
        self.op_stream = torch.full((self.max_ops,), -1, dtype=torch.int32, device=dev)
        self.op_coef = torch.zeros(self.max_ops, 6, dtype=torch.float32, device=dev)

    # --- registration (between steps, inside a collective_rpc) -------------------------------

    def has_site(self, site: Any) -> bool:
        return id(site) in self._index

    def idle(self, site: Any) -> bool:
        """True when no writer has ops at ``site``, so an eager forward may skip the launch."""
        index = self._index[id(site)]
        return self._req_count[index] == 0 and (self._global is None or index not in self._global.ranges)

    def register(
        self, rid: str, groups: dict[Any, list[dict]], *, skip: Sequence[int], prompt_len: int, generated: bool
    ) -> None:
        """Give request ``rid`` its own ops at each site in ``groups``. Replaces an earlier registration."""
        self.unregister(rid)
        if not self._free_slots:
            raise RuntimeError(
                f"static writes: all {self.slots} writer slots are in use. Raise {WRITE_SLOTS_ENV}, or "
                "steer fewer requests at once."
            )
        writer = self._build(groups, slot=self._free_slots[-1], skip=skip, prompt_len=prompt_len, generated=generated)
        self._free_slots.pop()
        self._writers[rid] = writer
        for index in writer.ranges:
            self._req_count[index] += 1
        self._rid_cache.clear()
        self._sync_prog(writer)

    def unregister(self, rid: str) -> None:
        writer = self._writers.pop(rid, None)
        if writer is None:
            return
        for index in writer.ranges:
            self._req_count[index] -= 1
        self._release(writer)
        self._free_slots.append(writer.slot)
        self._rid_cache.clear()
        self._sync_prog(writer, clear=True)

    def set_global(self, groups: dict[Any, list[dict]], scope: dict[str, Any] | None) -> None:
        """The global write: every row, or the rows ``scope`` names, at sites no request writes."""
        self.clear_global()
        if not groups:
            return
        scope = scope or {}
        writer = self._build(
            groups,
            slot=self.slots,
            skip=scope.get("skip_positions") or (),
            prompt_len=int(scope.get("prompt_len") or 0),
            generated=bool(scope.get("steer_generated", True)),
        )
        writer.unscoped = not scope
        self._global = writer
        self._sync_prog(writer)

    def clear_global(self) -> None:
        writer, self._global = self._global, None
        if writer is not None:
            self._release(writer)
            self._sync_prog(writer, clear=True)

    def _build(
        self, groups: dict[Any, list[dict]], *, slot: int, skip: Sequence[int], prompt_len: int, generated: bool
    ) -> _Writer:
        compiled: list[tuple[int, list[WriteOp]]] = []
        for site, specs in groups.items():
            index = self._index.get(id(site))
            if index is None:
                raise ValueError(f"static writes: {site.address} has no program site")
            ops = [compile_op(spec) for spec in specs]
            info = self._info[index]
            for op in ops:
                if op.u.numel() != info.width or op.w.numel() != info.width:
                    raise ValueError(
                        f"static write at {site.address}: vector width {op.w.numel()} does not match the "
                        f"activation width {info.width}"
                    )
                if op.stream >= 0 and len(site.shape) != 3:
                    raise ValueError(
                        f"stream={op.stream} was given for {site.address}, which has no stream axis: only the "
                        "residual points of a hyper-connection trunk carry one. Drop the coordinate, or steer "
                        "resid_streams."
                    )
                if op.stream >= info.streams:
                    raise ValueError(f"stream={op.stream} is out of range for {info.streams} residual streams")
            compiled.append((index, ops))
        total = sum(len(ops) for _, ops in compiled)
        base = self._alloc(total)
        flat: list[torch.Tensor] = []
        offsets: dict[int, int] = {}
        offset = 0
        for _, ops in compiled:
            for op in ops:
                for vec in (op.u, op.w):
                    if id(vec) not in offsets:
                        offsets[id(vec)] = offset
                        flat.append(vec)
                        offset += vec.numel()
        vectors = torch.cat(flat).to(self.device) if flat else torch.zeros(1, device=self.device)
        ptr, size = vectors.data_ptr(), vectors.element_size()
        vec_rows, streams, coefs = [], [], []
        ranges: dict[int, tuple[int, int]] = {}
        cursor = base
        for index, ops in compiled:
            ranges[index] = (cursor, len(ops))
            cursor += len(ops)
            for op in ops:
                vec_rows.append((ptr + offsets[id(op.u)] * size, ptr + offsets[id(op.w)] * size))
                streams.append(op.stream)
                coefs.append(op.coef)
        if total:
            self.op_vec[base : base + total].copy_(torch.tensor(vec_rows, dtype=torch.int64))
            self.op_stream[base : base + total].copy_(torch.tensor(streams, dtype=torch.int32))
            self.op_coef[base : base + total].copy_(torch.tensor(coefs, dtype=torch.float32))
        return _Writer(
            slot=slot,
            base=base,
            total=total,
            ranges=ranges,
            prompt_len=int(prompt_len),
            generated=bool(generated),
            skip=frozenset(int(p) for p in skip),
            vectors=vectors,
        )

    def _alloc(self, total: int) -> int:
        if total == 0:
            return 0
        for i, (start, size) in enumerate(self._free_ops):
            if size >= total:
                self._free_ops[i] = (start + total, size - total)
                if self._free_ops[i][1] == 0:
                    del self._free_ops[i]
                return start
        raise RuntimeError(
            f"static writes: no room for {total} more ops in a table of {self.max_ops}. Raise {WRITE_OPS_ENV}."
        )

    def _release(self, writer: _Writer) -> None:
        if writer.total:
            self._free_ops.append((writer.base, writer.total))
            self._free_ops.sort()
            merged: list[tuple[int, int]] = []
            for start, size in self._free_ops:
                if merged and merged[-1][0] + merged[-1][1] == start:
                    merged[-1] = (merged[-1][0], merged[-1][1] + size)
                else:
                    merged.append((start, size))
            self._free_ops = merged
        writer.vectors = None

    def _sync_prog(self, writer: _Writer, *, clear: bool = False) -> None:
        """Upload ``writer``'s column of ``prog``, and the global column, which a request write can mask."""
        column = self._prog_host[:, writer.slot]
        if writer.slot < self.slots:
            column[:] = 0
            if not clear:
                for index, (start, count) in writer.ranges.items():
                    column[index] = (start, count)
            self.prog[:, writer.slot].copy_(torch.from_numpy(column.copy()))
        glob = self._prog_host[:, self.slots]
        glob[:] = 0
        if self._global is not None:
            for index, (start, count) in self._global.ranges.items():
                if self._req_count[index] == 0:
                    glob[index] = (start, count)
        self.prog[:, self.slots].copy_(torch.from_numpy(glob.copy()))

    # --- per step (inside prepare_inputs, before the forward) --------------------------------

    def fill_rows(
        self,
        req_ids: Sequence[str],
        seq_lens: Sequence[int],
        starts: Sequence[int] | None,
        resolve: Callable[[str], str],
    ) -> None:
        """Point each row of the next forward at its writer, and upload the map when it changed."""
        rows = self._rows_host
        if not self._writers and self._global is None:
            if self._last_rows:
                rows[: self._last_rows, 0] = -1
                rows[: self._last_rows, 1] = 0
                self._upload(self._last_rows)
                self._last_rows = 0
            return
        total = min(sum(int(n) for n in seq_lens), self.max_n)
        reach = max(total, self._last_rows)
        rows[:reach, 0] = -1
        rows[:reach, 1] = 0
        glob = self._global
        offset = 0
        for i, (full_id, length) in enumerate(zip(req_ids, seq_lens, strict=False)):
            end = min(offset + int(length), self.max_n)
            if end <= offset:
                break
            n = end - offset
            writer = self._writer_for(str(full_id), resolve)
            if starts is not None:
                start = int(starts[i])
            elif writer is not None:
                start = 0 if n > 1 else writer.prompt_len
            else:
                start = 0
            if writer is not None:
                rows[offset:end, 0] = np.where(writer.rows(start, n), writer.slot, -1)
            if glob is not None:
                rows[offset:end, 1] = glob.rows(start, n)
            offset = end
        self._last_rows = total
        self._upload(reach)

    def _writer_for(self, full_id: str, resolve: Callable[[str], str]) -> _Writer | None:
        if full_id not in self._rid_cache:
            if len(self._rid_cache) > 4096:
                self._rid_cache.clear()
            self._rid_cache[full_id] = self._writers.get(resolve(full_id))
        return self._rid_cache[full_id]

    def _upload(self, reach: int) -> None:
        if reach <= 0 or np.array_equal(self._rows_host[:reach], self._rows_sent[:reach]):
            return
        source = torch.from_numpy(self._rows_host[:reach].copy())
        if self.device.type == "cuda":
            source = source.pin_memory()
        self.row_map[:reach].copy_(source, non_blocking=True)
        self._rows_sent[:reach] = self._rows_host[:reach]

    # --- the forward ---------------------------------------------------------------------------

    def launch(self, site: Any, hidden: torch.Tensor, residual: torch.Tensor | None, n: int) -> None:
        """Add ``site``'s writes into the first ``n`` rows of ``hidden``, reading ``hidden + residual``."""
        if n <= 0:
            return
        index = self._index[id(site)]
        if hidden.dim() == 2:
            streams, h_col = 1, 0
        elif hidden.dim() == 3:
            streams, h_col = int(hidden.shape[1]), int(hidden.stride(1))
        else:
            raise RuntimeError(f"static write at {site.address}: cannot serve a {hidden.dim()}-D activation")
        width = int(hidden.shape[-1])
        if width != self._info[index].width:
            raise RuntimeError(
                f"static write at {site.address}: live width {width} != site width {self._info[index].width}"
            )
        res = residual if residual is not None else hidden
        if residual is not None and tuple(residual.shape[1:]) != tuple(hidden.shape[1:]):
            raise RuntimeError(
                f"static site {site.address}: residual trailing {tuple(residual.shape[1:])} "
                f"!= hidden trailing {tuple(hidden.shape[1:])}"
            )
        r_col = 0 if res.dim() == 2 else int(res.stride(1))
        block = min(_MAX_BLOCK, 1 << max(width - 1, 0).bit_length())
        _kernel()[(min(n, self.max_n), streams)](
            hidden,
            res,
            int(hidden.stride(0)),
            h_col,
            int(hidden.stride(-1)),
            int(res.stride(0)),
            r_col,
            int(res.stride(-1)),
            self.row_map,
            self.prog[index],
            self.op_vec,
            self.op_stream,
            self.op_coef,
            width,
            self.slots,
            HAS_RES=residual is not None,
            BLOCK=block,
            num_warps=4 if block <= 1024 else 8,
        )


_KERNEL: Any = None


def _kernel() -> Any:
    """The Triton kernel, built on first use: Triton is a vLLM dependency, not one of this package's."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton  # pyright: ignore[reportMissingImports]
    import triton.language as tl  # pyright: ignore[reportMissingImports]

    @triton.jit(do_not_specialize=["h_row", "h_col", "h_el", "r_row", "r_col", "r_el", "width", "n_slots"])
    def static_write(
        h_ptr,
        r_ptr,
        h_row,
        h_col,
        h_el,
        r_row,
        r_col,
        r_el,
        row_map_ptr,
        prog_ptr,
        op_vec_ptr,
        op_stream_ptr,
        op_coef_ptr,
        width,
        n_slots,
        HAS_RES: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        stream = tl.program_id(1)
        slot = tl.load(row_map_ptr + 2 * row)
        glob = tl.load(row_map_ptr + 2 * row + 1)
        mine = tl.maximum(slot, 0)
        start = tl.load(prog_ptr + 2 * mine)
        count = tl.where(slot >= 0, tl.load(prog_ptr + 2 * mine + 1), 0)
        use_glob = (count == 0) & (glob != 0)
        start = tl.where(use_glob, tl.load(prog_ptr + 2 * n_slots), start)
        count = tl.where(use_glob, tl.load(prog_ptr + 2 * n_slots + 1), count)
        if count > 0:
            h_base = h_ptr + row.to(tl.int64) * h_row + stream.to(tl.int64) * h_col
            r_base = r_ptr + row.to(tl.int64) * r_row + stream.to(tl.int64) * r_col
            offs = tl.arange(0, BLOCK)
            for k in range(start, start + count):
                s = tl.load(op_stream_ptr + k)
                u_ptr = tl.load(op_vec_ptr + 2 * k).to(tl.pointer_type(tl.float32))
                w_ptr = tl.load(op_vec_ptr + 2 * k + 1).to(tl.pointer_type(tl.float32))
                c0 = tl.load(op_coef_ptr + 6 * k)
                c1 = tl.load(op_coef_ptr + 6 * k + 1)
                c2 = tl.load(op_coef_ptr + 6 * k + 2)
                c3 = tl.load(op_coef_ptr + 6 * k + 3)
                lo = tl.load(op_coef_ptr + 6 * k + 4)
                hi = tl.load(op_coef_ptr + 6 * k + 5)
                dot = tl.zeros([BLOCK], dtype=tl.float32)
                sq = tl.zeros([BLOCK], dtype=tl.float32)
                for c in range(0, width, BLOCK):
                    col = c + offs
                    mask = col < width
                    x = tl.load(h_base + col * h_el, mask=mask, other=0.0).to(tl.float32)
                    if HAS_RES:
                        x += tl.load(r_base + col * r_el, mask=mask, other=0.0).to(tl.float32)
                    u = tl.load(u_ptr + col, mask=mask, other=0.0)
                    dot += x * u
                    sq += x * x
                proj = tl.sum(dot, axis=0)
                capped = tl.minimum(tl.maximum(proj, lo), hi)
                coef = c0 + c1 * tl.sqrt(tl.sum(sq, axis=0)) + c2 * proj + c3 * (capped - proj)
                coef = tl.where((s < 0) | (s == stream), coef, 0.0)
                for c in range(0, width, BLOCK):
                    col = c + offs
                    mask = col < width
                    h = tl.load(h_base + col * h_el, mask=mask, other=0.0)
                    w = tl.load(w_ptr + col, mask=mask, other=0.0)
                    tl.store(h_base + col * h_el, (h.to(tl.float32) + coef * w).to(h.dtype), mask=mask)
                tl.debug_barrier()

    _KERNEL = static_write
    return _KERNEL
