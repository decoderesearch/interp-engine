"""VRAM arithmetic: what a configuration costs, and the largest one that fits.

Three questions this answers, none of which the engine could answer before:

- Will ``load_model(model, backend=..., dtype=...)`` fit on this card, and where does the memory go?
- What is the largest ``max_model_len`` / ``gpu_memory_utilization`` that *will* fit?
- How much room does a caller have to leave for things the engine never sees -- preloaded SAEs, a
  Jacobian-lens matrix, another process on the same card?

**This is a calibrated lower bound, not a measurement.** vLLM does not let anyone predict its
memory: it profiles the device at startup and sizes the KV cache from what it finds. So every
number here is arithmetic plus a margin that was paid for by a real out-of-memory failure, and the
margins live in :data:`CALIBRATION` with the failure that produced each one. A configuration this
module calls a fit can still fail on a card with something else running on it; a configuration it
refuses is genuinely too big. The asymmetry is deliberate -- under-counting is the direction that
OOMs.

The load-bearing idea
---------------------
``gpu_memory_utilization`` is a fraction of the **whole card**, not of what is free. So every
allocation falls on one side of that line, and the two sides fail differently:

- **Inside the pool** (``util x total``): weights, static tap buffers, the CUDA graph pool, and the
  KV cache with whatever is left. Ask for too much here and vLLM refuses at startup, naming the
  cache it could not build.
- **Outside the pool** (``(1 - util) x total``): the CUDA context, vLLM's own overshoot past its
  budget, allocator fragmentation, and anything the caller allocates itself. Ask for too much here
  and the process dies *during* vLLM's warmup, after the KV cache size already looked fine.

The second failure is the one that surprises people, because nothing in the configuration mentions
it. It is why :class:`Reservations` exists and why it distinguishes per-rank from host-wide.

``eager`` is a different equation entirely, and weights are not what OOMs it. See
:func:`eager_activation_bytes`.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

GIB = 1024**3

#: The backends :func:`estimate` understands. Mirrors ``load.BACKENDS`` deliberately rather than
#: importing it: ``load`` pulls in ``torch`` and the model classes, and this module is meant to be
#: importable by a sizing script that will never build a model.
#: ``tests/test_memory.py`` asserts the two agree, so a new backend cannot land unpriced.
BACKENDS = ("auto", "vllm", "vllm-static", "vllm-generate", "eager")

#: The vLLM backends, which share the pool arithmetic above.
VLLM_BACKENDS = ("vllm", "vllm-static", "vllm-generate")


# --------------------------------------------------------------------------- dtypes


#: Bytes per element for a dtype spelled as a string, narrowest tag first: ``fp4`` must be tested
#: before ``fp8`` or a substring match reports the wrong width, and 4-bit is the row that matters --
#: counting a 4-bit tensor at one byte doubles it.
#:
#: This is the canonical copy. ``vllm_capture/static.py`` delegates here rather than keeping its
#: own, because a dtype table that disagrees with itself across two modules is a silent 2x.
_DTYPE_BYTES: tuple[tuple[tuple[str, ...], float], ...] = (
    (("fp4", "nvfp4", "mxfp4", "int4", "uint4", "nf4"), 0.5),
    (("fp8", "int8", "uint8", "e4m3", "e5m2"), 1.0),
    (("float32", "fp32"), 4.0),
    (("bfloat16", "float16", "bf16", "fp16", "half"), 2.0),
)


def dtype_bytes_or_none(name: Any) -> float | None:
    """Bytes per element for a dtype named as a string, or None when the name is unrecognized.

    None rather than a default, so a caller falls back to something it knows instead of a guess:
    this reads fields no schema constrains, and a wrong *narrower* answer is what OOMs a pod.
    """
    if name is None or name == "":
        return None
    text = str(name).lower()
    for tags, size in _DTYPE_BYTES:
        if any(tag in text for tag in tags):
            return size
    return None


def dtype_bytes(name: Any, *, default: float = 2.0) -> float:
    """Bytes per element, falling back to ``default`` (bf16) for an unrecognized name."""
    resolved = dtype_bytes_or_none(name)
    return default if resolved is None else resolved


# --------------------------------------------------------------------- calibration


@dataclass(frozen=True)
class Calibration:
    """One empirical constant, with the failure that paid for it.

    Kept as data rather than as bare module constants so that ``gpu-sizer/verify.py`` can print
    them beside what it measured, and so a re-derivation has somewhere to write its provenance. A
    number in here without a ``source`` is a guess wearing a suit.
    """

    value: float
    unit: str
    why: str
    source: str


CALIBRATION: dict[str, Calibration] = {
    "cuda_context_gib": Calibration(
        value=0.6,
        unit="GiB",
        why=(
            "The CUDA context plus cuBLAS/cuDNN handles in the process holding the engine. Charged "
            "INSIDE vLLM's pool: vLLM sizes its KV cache as `total x utilization - what the process is "
            "already using`, and by the time it profiles, the context is already using this. So the "
            "context comes out of the KV cache's share rather than out of the margin."
        ),
        source=(
            "Measured on gpt2/vllm/bf16, A40, utilization 0.9. Pool claimed 40.04 GiB; vLLM built a "
            "1,151,632-token cache, which at gpt2's 36,864 B/token is 39.53 GiB, beside 0.26 GiB of "
            "weights -- so vLLM charged 0.25 GiB of process overhead against its own budget, and a "
            "further 0.34 GiB spilled past it (peak 40.38). The context is therefore split, and 0.6 "
            "inside the pool covers the measured share with margin: it makes the KV-cache prediction "
            "0.9% LOW rather than 0.6% high, which is the side to be wrong on."
        ),
    ),
    "vllm_overshoot_gib": Calibration(
        value=0.6,
        unit="GiB",
        why=(
            "What vLLM allocates past its own budget during warmup -- sampling scratch over a large "
            "vocab, and the profiling run's own peak. This genuinely does land outside the pool, "
            "because the pool is fully spoken for by the time it happens."
        ),
        source=(
            "0.34 GiB measured for gpt2/vllm on an A40, the whole of that run's overflow. Held at 0.6 "
            "because the term scales with vocabulary and batch width, and gpt2's 50k vocab is the "
            "smallest of anything here."
        ),
    ),
    "frag_fraction": Calibration(
        value=0.04,
        unit="fraction of card",
        why="Caching-allocator fragmentation; a slice of the card that is free but unusable.",
        source="Neuronpedia's sae_memory.py, derived over the A40/A100 fleet.",
    ),
    "graph_pool_gib": Calibration(
        value=3.0,
        unit="GiB",
        why=(
            "The CUDA graph pool captured when graphs are on. Zero under enforce_eager, which is why "
            "backend='vllm' is cheaper than the two graph backends before a single tap is declared."
        ),
        source="`vllm_capture/static.py`'s graph_fudge, load-bearing in fit_max_num_batched_tokens.",
    ),
    "eager_workspace_gib": Calibration(
        value=0.5,
        unit="GiB",
        why=(
            "Kernel workspace and allocator slack for a plain HF forward, over and above the "
            "activation terms counted explicitly."
        ),
        source="Rounded up from small-model eager runs; the activation terms dominate above ~1k tokens.",
    ),
    "hybrid_kv_overhead": Calibration(
        value=1.10,
        unit="multiplier on KV bytes",
        why=(
            "Extra KV cost on a trunk that mixes full-attention and sliding-window layers. vLLM's "
            "hybrid allocator pages whole blocks across both layer groups, so it fits slightly fewer "
            "tokens than the flat per-layer arithmetic predicts. Applied only when `layer_types` shows "
            "a mix; a uniform trunk needs no such term."
        ),
        source=(
            "gemma-3-1b-pt on an A40 at 4k: 1,515,475 tokens predicted flat against 1,399,779 built, "
            "i.e. 8% optimistic. 1.10 turns that into 0.98x. gpt2 (1.01x) and Qwen3-4B (1.00x) are "
            "uniform trunks and confirm the flat figure needs no correction without the mix. "
            "gemma-3-12b-pt at 8k is the loose end: it built 60,714 against 41,552 predicted, 1.46x "
            "CONSERVATIVE, so one constant does not describe both sizes. Left at 1.10 rather than "
            "widened, for two reasons. The error is one-directional -- both hybrid runs under-promise "
            "capacity, and under-promising costs a context rung, while over-promising costs an OOM. "
            "And the two figures may not be comparable: vLLM's hybrid allocator splits layers into KV "
            "groups, and the token count it logs need not mean the same thing as a flat per-layer "
            "prediction on a 40-sliding/8-full trunk. Settling that needs a read of the allocator "
            "rather than a third data point."
        ),
    ),
    "max_util": Calibration(
        value=0.90,
        unit="fraction of card",
        why=(
            "vLLM's own default and this module's ceiling. Above it the margin outside the pool is "
            "thinner than the overshoot measured above, so a fit here is luck."
        ),
        source="vLLM default; Neuronpedia caps its derived utilization at the same value.",
    ),
    "min_util": Calibration(
        value=0.10,
        unit="fraction of card",
        why="Below this vLLM has no room for weights plus a usable KV cache; refuse instead.",
        source="Neuronpedia's sae_memory.py MIN_UTIL.",
    ),
}


def _cal(name: str) -> float:
    return CALIBRATION[name].value


# --------------------------------------------------------------------------- GPUs


@dataclass(frozen=True)
class GpuSpec:
    """One GPU model, as much of it as VRAM arithmetic needs.

    ``total_bytes`` is what a **CUDA process sees**, which is not what the marketing number says and
    not always what ``nvidia-smi`` prints. Two reductions apply, and both have bitten this codebase:

    - **ECC costs 6.25% of GDDR6.** An A40 is a 48 GiB board. With ECC on, ``nvidia-smi`` reports
      46068 MiB and a process gets ~44.4 GiB; with ECC off it reports 49140 MiB and a process gets
      ~47.4 GiB. A cloud provider hands you whichever the host is set to, so a table that holds the
      larger number is ~3 GiB optimistic on half a fleet -- and because utilization is a fraction of
      the whole card, all 3 GiB land on vLLM.
    - **The driver reserves ~0.5 GiB** on top, which is the gap between ``nvidia-smi``'s total and
      ``torch.cuda.get_device_properties().total_memory``.

    So ``total_bytes`` here means the *process-visible* total, the number vLLM multiplies by
    ``gpu_memory_utilization``. Where a row was measured rather than computed, ``provenance`` says so.
    """

    name: str
    total_bytes: int
    #: Compute capability as ``(major, minor)``. Decides whether a quantization scheme has hardware
    #: behind it: FP8 tensor cores need >= 8.9 (Ada/Hopper), NVFP4 needs >= 10.0 (Blackwell), and the
    #: MXFP4 Triton path needs only >= 7.5. An Ampere card running "fp8" is emulating it.
    compute_capability: tuple[int, int]
    #: Peak memory bandwidth, for the throughput estimate. Decode is bandwidth-bound, so this is a
    #: better predictor of tokens/sec than FLOPs are.
    bandwidth_gib_s: float
    #: "measured" when a real card reported ``total_bytes``, otherwise where the number came from.
    provenance: str
    #: Cloud/consumer aliases, so a caller can name the card the way their provider does.
    aliases: tuple[str, ...] = ()
    #: True when ECC is on for the measured figure, None when not known.
    ecc_enabled: bool | None = None
    notes: str = ""

    @property
    def total_gib(self) -> float:
        """Capacity in GiB, rounded to the precision the catalog actually knows.

        Rounded because the round trip does not survive otherwise: 44.4 GiB is stored as 47,681,159,987
        bytes, and dividing that back gives 44.39999999944121. Every row here is known to about 0.1 GiB
        -- see each `provenance` -- so two decimals is already more precision than exists, and the
        unrounded form only makes the value awkward to compare or print. `total_bytes` stays exact and
        remains what the arithmetic uses.
        """
        return round(self.total_bytes / GIB, 2)

    @property
    def supports_fp8(self) -> bool:
        """FP8 tensor cores, i.e. Ada/Hopper and later. Below this, FP8 weights are emulated."""
        return self.compute_capability >= (8, 9)

    @property
    def supports_fp4(self) -> bool:
        """NVFP4 tensor cores, i.e. Blackwell and later."""
        return self.compute_capability >= (10, 0)

    @property
    def supports_mxfp4_kernels(self) -> bool:
        """The Triton MXFP4 path transformers needs to avoid dequantizing to bf16."""
        return self.compute_capability >= (7, 5)

    def cannot_verify(self) -> tuple[str, ...]:
        """Quantization schemes this card cannot give ground truth for.

        Read by the verification harness so an A40 record states on its face that it proves nothing
        about FP8, rather than leaving the absence to be mistaken for a pass.
        """
        gaps: list[str] = []
        if not self.supports_fp8:
            gaps.append("fp8")
        if not self.supports_fp4:
            gaps.append("nvfp4")
        if not self.supports_mxfp4_kernels:
            gaps.append("mxfp4")
        return tuple(gaps)


def _gib(value: float) -> int:
    return int(value * GIB)


#: The cards worth sizing against, keyed by the name ``nvidia-smi`` reports.
#:
#: ``total_bytes`` is process-visible, not the marketing capacity -- see :class:`GpuSpec` for the two
#: reductions that make those differ. **A row marked ``measured`` came off a real card**; the rest are
#: computed from the board capacity minus the usual driver reservation, and a sizer should widen its
#: margin on those. This distinction is not decoration: the A40 row in the table this was seeded from
#: was 47.4 GiB for a year, which is a real A40 with ECC *off*, and was 3 GiB optimistic on every host
#: that had it on. All 3 GiB land on vLLM, because utilization is a fraction of the whole card.
GPUS: dict[str, GpuSpec] = {
    "NVIDIA A40": GpuSpec(
        name="NVIDIA A40",
        total_bytes=_gib(44.4),
        compute_capability=(8, 6),
        bandwidth_gib_s=696,
        provenance="measured (ECC on; nvidia-smi reports 46068 MiB)",
        ecc_enabled=True,
        aliases=("a40",),
        notes="48 GiB board; ECC costs 6.25% of GDDR6, so ECC off gives ~47.4 instead. Cheap and common.",
    ),
    "NVIDIA RTX A6000": GpuSpec(
        name="NVIDIA RTX A6000",
        total_bytes=_gib(47.4),
        compute_capability=(8, 6),
        bandwidth_gib_s=768,
        provenance="measured, but before the A40 ECC correction -- treat as ECC-off",
        ecc_enabled=False,
        aliases=("a6000", "rtx a6000"),
    ),
    "NVIDIA L40S": GpuSpec(
        name="NVIDIA L40S",
        total_bytes=_gib(44.3),
        compute_capability=(8, 9),
        bandwidth_gib_s=864,
        provenance="measured",
        aliases=("l40s",),
        notes="Ada, so the first card in this table with FP8 tensor cores.",
    ),
    "NVIDIA A100 80GB PCIe": GpuSpec(
        name="NVIDIA A100 80GB PCIe",
        total_bytes=_gib(79.1),
        compute_capability=(8, 0),
        bandwidth_gib_s=1935,
        provenance="measured",
        aliases=("a100", "a100-80gb", "a100 80gb"),
    ),
    "NVIDIA A100-SXM4-80GB": GpuSpec(
        name="NVIDIA A100-SXM4-80GB",
        total_bytes=_gib(79.1),
        compute_capability=(8, 0),
        bandwidth_gib_s=2039,
        provenance="measured",
        aliases=("a100-sxm", "a100 sxm"),
    ),
    "NVIDIA A100-SXM4-40GB": GpuSpec(
        name="NVIDIA A100-SXM4-40GB",
        total_bytes=_gib(39.4),
        compute_capability=(8, 0),
        bandwidth_gib_s=1555,
        provenance="spec sheet (40 GiB board) minus the usual driver reservation",
        aliases=("a100-40gb", "a100 40gb"),
        notes="The A100 Colab Pro offers.",
    ),
    "NVIDIA H100 80GB HBM3": GpuSpec(
        name="NVIDIA H100 80GB HBM3",
        total_bytes=_gib(79.1),
        compute_capability=(9, 0),
        bandwidth_gib_s=3350,
        provenance="spec sheet; same board capacity as the measured A100 80GB",
        aliases=("h100", "h100-sxm", "h100 80gb"),
        notes="First card here with fast FP8. The one to verify the FP8 rows on.",
    ),
    "NVIDIA H100 NVL": GpuSpec(
        name="NVIDIA H100 NVL",
        total_bytes=_gib(93.1),
        compute_capability=(9, 0),
        bandwidth_gib_s=3900,
        provenance="spec sheet (94 GiB board)",
        aliases=("h100-nvl", "h100 nvl"),
    ),
    "NVIDIA H200 NVL": GpuSpec(
        name="NVIDIA H200 NVL",
        total_bytes=_gib(139.0),
        compute_capability=(9, 0),
        bandwidth_gib_s=4800,
        provenance="measured",
        aliases=("h200", "h200-nvl"),
    ),
    "NVIDIA B200": GpuSpec(
        name="NVIDIA B200",
        total_bytes=_gib(178.35),
        compute_capability=(10, 0),
        bandwidth_gib_s=8000,
        provenance="measured (ECC on; nvidia-smi reports 183359 MiB, torch 191503007744 bytes)",
        ecc_enabled=True,
        aliases=("b200",),
        notes=(
            "Blackwell: NVFP4 in hardware. The engine's published throughput tables were measured here. "
            "192 GB HBM3e, not the 180 GB the marketing sheets quote. Do NOT apply the GDDR6 ECC "
            "correction to this row: HBM3e carries ECC on-die, so the 6.25% that costs an A40 3 GiB "
            "costs this card nothing, and the measured total above already has ECC enabled."
        ),
    ),
    "NVIDIA B300": GpuSpec(
        name="NVIDIA B300",
        total_bytes=_gib(286.0),
        compute_capability=(10, 0),
        bandwidth_gib_s=8000,
        provenance="spec sheet (288 GB HBM3e board); unverified",
        aliases=("b300", "blackwell ultra"),
    ),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": GpuSpec(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        total_bytes=_gib(94.5),
        compute_capability=(12, 0),
        bandwidth_gib_s=1792,
        provenance="measured",
        aliases=("rtx pro 6000", "pro 6000", "rtx6000"),
        notes="96 GiB for roughly a third of an H200's hourly cost, at ~1.8 TB/s against ~4.8.",
    ),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": GpuSpec(
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        total_bytes=_gib(94.5),
        compute_capability=(12, 0),
        bandwidth_gib_s=1792,
        provenance="inferred from the measured Server Edition; same 96 GiB board, unverified here",
        aliases=("rtx pro 6000 workstation", "pro 6000 workstation"),
        notes=(
            "Same die and capacity as the Server Edition, which is why it borrows that measurement. "
            "The two differ in cooling and power, not memory. A provider hands you whichever the host "
            "has, so a config listing one should size against the other too."
        ),
    ),
    "NVIDIA GeForce RTX 4090": GpuSpec(
        name="NVIDIA GeForce RTX 4090",
        total_bytes=_gib(23.6),
        compute_capability=(8, 9),
        bandwidth_gib_s=1008,
        provenance="measured",
        aliases=("4090", "rtx 4090"),
    ),
    "NVIDIA GeForce RTX 5090": GpuSpec(
        name="NVIDIA GeForce RTX 5090",
        total_bytes=_gib(31.5),
        compute_capability=(12, 0),
        bandwidth_gib_s=1792,
        provenance="spec sheet (32 GiB board)",
        aliases=("5090", "rtx 5090"),
    ),
    "NVIDIA RTX A5000": GpuSpec(
        name="NVIDIA RTX A5000",
        total_bytes=_gib(23.6),
        compute_capability=(8, 6),
        bandwidth_gib_s=768,
        provenance="measured",
        aliases=("a5000",),
    ),
    "NVIDIA RTX A4500": GpuSpec(
        name="NVIDIA RTX A4500",
        total_bytes=_gib(19.7),
        compute_capability=(8, 6),
        bandwidth_gib_s=640,
        provenance="measured",
        aliases=("a4500",),
    ),
    "NVIDIA RTX A4000": GpuSpec(
        name="NVIDIA RTX A4000",
        total_bytes=_gib(15.7),
        compute_capability=(8, 6),
        bandwidth_gib_s=448,
        provenance="measured",
        aliases=("a4000",),
    ),
    # The Ada workstation pair. Same usable capacity as the Ampere cards they sit beside in a fallback
    # list, which is why those lists are sized by whichever is smallest rather than by the newest.
    "NVIDIA RTX 4000 Ada Generation": GpuSpec(
        name="NVIDIA RTX 4000 Ada Generation",
        total_bytes=_gib(19.7),
        compute_capability=(8, 9),
        bandwidth_gib_s=360,
        provenance="measured",
        aliases=("rtx 4000 ada", "4000 ada"),
        notes="Ada, so FP8 tensor cores are present where the A4500 it pairs with has none.",
    ),
    "NVIDIA RTX 2000 Ada Generation": GpuSpec(
        name="NVIDIA RTX 2000 Ada Generation",
        total_bytes=_gib(15.7),
        compute_capability=(8, 9),
        bandwidth_gib_s=224,
        provenance="measured",
        aliases=("rtx 2000 ada", "2000 ada"),
        notes="The smallest card worth serving from: 16 GiB holds an 8B model at bf16 and little else.",
    ),
    "NVIDIA L4": GpuSpec(
        name="NVIDIA L4",
        total_bytes=_gib(22.0),
        compute_capability=(8, 9),
        bandwidth_gib_s=300,
        provenance="spec sheet (24 GiB board); Colab reports ~22.5 GiB",
        aliases=("l4",),
        notes="Colab's default paid GPU, and what this repo's CI GPU job runs on.",
    ),
    "Tesla T4": GpuSpec(
        name="Tesla T4",
        total_bytes=_gib(14.6),
        compute_capability=(7, 5),
        bandwidth_gib_s=320,
        provenance="spec sheet (16 GiB board); Colab reports 15360 MiB",
        aliases=("t4", "nvidia t4"),
        notes="Colab's free tier. Compute 7.5 clears the MXFP4 kernel floor but has no bf16 tensor cores.",
    ),
}


def find_gpu(name: str) -> GpuSpec | None:
    """Look up a card by ``nvidia-smi`` name or by any alias, case- and spacing-insensitively.

    Cloud providers, Colab and ``nvidia-smi`` all spell the same card differently, and a sizer that
    only accepts one spelling is a sizer people give up on.
    """
    if not name:
        return None
    if name in GPUS:
        return GPUS[name]
    wanted = " ".join(name.lower().replace("-", " ").split())
    for spec in GPUS.values():
        candidates = [spec.name, *spec.aliases]
        if any(" ".join(c.lower().replace("-", " ").split()) == wanted for c in candidates):
            return spec
    # Last resort: a unique substring match, so "gemma on an h200" finds the H200 NVL row.
    hits = [
        spec
        for spec in GPUS.values()
        if wanted in " ".join(spec.name.lower().replace("-", " ").split())
        or any(wanted in alias.lower() for alias in spec.aliases)
    ]
    return hits[0] if len(hits) == 1 else None


def local_gpu(index: int = 0) -> GpuSpec | None:
    """The card in this box, with its **measured** total rather than the catalog's.

    Prefers what the driver reports over what the table says, because the table cannot know whether
    ECC is on. When the name is unknown the row is synthesized from the measured total, with a
    compute capability read from the driver -- so an unlisted card still sizes correctly.
    """
    total = device_total_bytes(index)
    if total is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name,compute_cap,ecc.mode.current", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        name, cap, ecc = (part.strip() for part in out.stdout.strip().splitlines()[0].split(","))
        major, _, minor = cap.partition(".")
        capability = (int(major), int(minor))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None

    known = find_gpu(name)
    if known is not None:
        from dataclasses import replace

        return replace(
            known,
            total_bytes=total,
            compute_capability=capability,
            ecc_enabled=ecc.lower().startswith("enab"),
            provenance=f"measured on this host (ECC {ecc.lower()})",
        )
    return GpuSpec(
        name=name,
        total_bytes=total,
        compute_capability=capability,
        bandwidth_gib_s=0.0,
        provenance="measured on this host; not in the catalog",
        ecc_enabled=ecc.lower().startswith("enab"),
    )


def device_total_bytes(index: int = 0) -> int | None:
    """Process-visible total VRAM of a local GPU, or None when there is no CUDA device.

    Shells out to ``nvidia-smi`` rather than importing torch, deliberately and for two reasons:
    initializing CUDA in the calling process costs a context this module exists to account for, and
    a sizing script that has not loaded a model should not be the thing that allocates one.
    ``validator/comparison/sizing.py`` reads the device the same way, for the same reason.

    ``nvidia-smi``'s total is the board total after ECC, which is ~0.5 GiB above what a process
    gets; the driver reservation is subtracted here so the result matches
    ``torch.cuda.get_device_properties().total_memory``.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = out.stdout.strip().splitlines()
    if not line:
        return None
    try:
        mib = float(line[0].strip())
    except ValueError:
        return None
    driver_reservation = 0.5 * GIB
    return max(int(mib * 1024 * 1024 - driver_reservation), 0)


# ------------------------------------------------------------------- model weights


#: Safetensors dtype tags that are **containers for quantized weights** rather than weights
#: themselves. A 4-bit checkpoint stores its payload packed into bytes or int32 words, so the element
#: count in the header is a count of *containers*, not of parameters -- see :func:`logical_param_count`.
_CONTAINER_DTYPES: dict[str, float] = {
    "U8": 1.0,
    "I8": 1.0,
    "F8_E4M3": 1.0,
    "F8_E5M2": 1.0,
    "U16": 2.0,
    "I16": 2.0,
    "U32": 4.0,
    "I32": 4.0,
    "U64": 8.0,
    "I64": 8.0,
}

#: Safetensors tags for the two fp8 formats. Whether these hold weights or *scales* depends on the
#: scheme around them: in an fp8 checkpoint they are the payload, in a 4-bit one they are the
#: per-block scales sitting beside a ``U8`` payload -- see :func:`logical_param_count`.
_FP8_TAGS: frozenset[str] = frozenset({"F8_E4M3", "F8_E5M2"})

#: Safetensors tags for 64-bit integers, which are indices and bookkeeping -- a rope table, an MTP
#: map -- and never a packed weight. They are containers by width, so the byte arithmetic wants them
#: in :data:`_CONTAINER_DTYPES`, but unpacking one at a 4-bit scheme's width multiplies a few million
#: index entries by sixteen and invents parameters out of them.
_INDEX_TAGS: frozenset[str] = frozenset({"I64", "U64"})

#: Safetensors tags that are **only ever scales**, whatever the scheme around them. ``F8_E8M0`` is
#: eight bits of exponent and no mantissa, so nothing stores a weight in it; it exists to carry the
#: per-block scale of an MX or ue8m0 payload. Counting one as a parameter is how DeepSeek V4 came to
#: report 8.9e9 weights it does not have -- every tensor carrying the tag is named ``.scale``.
_SCALE_TAGS: frozenset[str] = frozenset({"F8_E8M0"})

#: Bytes per logical parameter for each quantization scheme, i.e. what one weight really occupies
#: once unpacked from its container. Keyed on substrings of ``quant_method``.
_SCHEME_WIDTH: tuple[tuple[tuple[str, ...], float], ...] = (
    (("mxfp4", "nvfp4", "fp4", "int4", "uint4", "nf4", "awq", "gptq"), 0.5),
    (("fp8", "int8", "compressed-tensors", "finegrained_fp8", "modelopt_fp8"), 1.0),
)


def _scheme_width(quant_method: str) -> float | None:
    """Bytes per logical parameter for a quantization scheme, or None when it is not recognized.

    ``awq`` and ``gptq`` are listed at 4-bit because that is what they are in practice; an 8-bit GPTQ
    checkpoint exists but is rare, and reading it as 4-bit only over-states the dequantized size,
    which is the safe direction for a warning.
    """
    text = (quant_method or "").lower()
    if not text:
        return None
    for tags, width in _SCHEME_WIDTH:
        if any(tag in text for tag in tags):
            return width
    return None


def logical_param_count(elements_by_dtype: dict[str, int], quant_method: str = "", expert_dtype: str = "") -> int:
    """Logical parameters, unpacking whatever the containers hold.

    **This is the correction that makes the dequantization warning honest.** Safetensors headers
    report a count of *stored elements*, and for a packed checkpoint that is not the parameter count:
    gpt-oss-20b reports ``{BF16: 1.80e9, U8: 1.02e10}``, which sums to 12.0e9 for a model with about
    22e9 parameters. The U8 buckets are MXFP4 blocks holding two values per byte, so each stored byte
    is two parameters, and reading the sum as a parameter count under-states the model by 45%.

    That error matters in exactly one place and it matters a lot: the size the checkpoint becomes when
    transformers cannot find its kernels and *silently* dequantizes to bf16. Priced off stored
    elements it looks like 22 GiB; priced off logical parameters it is ~41 GiB, which is the
    documented figure and the difference between a 48 GiB card and an 80 GiB one.

    ``expert_dtype`` is what a byte container holds on a **mixed-precision** checkpoint, and without
    it the unpacking silently does not happen. DeepSeek-V4-Flash declares ``quant_method: fp8`` and
    ``expert_dtype: fp4``: the fp8 half is tagged ``F8_E4M3`` and needs no unpacking, while the
    routed experts are fp4 two-to-a-byte inside an ``I8`` array, so sizing the unpacking off the
    declared scheme leaves 141.7e9 containers read as 141.7e9 parameters against a true 290.9e9.
    The narrower of the two widths is the one the byte containers are packed at.

    An unquantized checkpoint is unaffected: its buckets are all float, so containers and parameters
    are the same thing.
    """
    widths = [w for w in (_scheme_width(quant_method), _scheme_width(expert_dtype)) if w is not None]
    native = min(widths) if widths else None
    total = 0
    for tag, count in elements_by_dtype.items():
        upper = str(tag).upper()
        if upper in _SCALE_TAGS:
            continue
        container = None if upper in _INDEX_TAGS else _CONTAINER_DTYPES.get(upper)
        if container is None or native is None:
            # A float bucket, or a scheme we do not recognize: one element is one parameter.
            total += int(count)
            continue
        if native < 1.0 and upper in _FP8_TAGS:
            # A 4-bit checkpoint packs its payload into bytes or int32 words and keeps its per-block
            # *scales* in fp8 beside it. Scales are not parameters, so unpacking them two-to-a-byte
            # invents weights that do not exist: on Llama-3.3-70B-FP4 it turns 70.5e9 into 79e9.
            # Counting them one-for-one leaves them slightly over-counted rather than doubled, which
            # keeps the total a little conservative without inflating it by a whole tensor group.
            total += int(count)
            continue
        total += int(count * (container / native))
    return total


@dataclass(frozen=True)
class WeightBytes:
    """What a checkpoint's weights cost, keyed on how it will be *loaded*.

    Weight sizing is the term that has been independently re-derived three times across this
    codebase and its downstreams, in three different shapes, which is why this returns the parts
    rather than one number:

    - ``param_count`` is the **logical** parameter count, which is what a *dequantized* load costs.
    - ``on_disk_bytes`` is what the checkpoint occupies as stored, which is what a native quantized
      load costs.
    - ``elements_by_dtype`` is the raw header count, which is neither, and is kept because it is
      where to look when a total is surprising.

    Collapsing those into one number is what made the collapsed version unusable downstream, and it
    is also how the engine's own estimator came to be 2x optimistic on its default eager path -- see
    :meth:`bytes_for_load`.
    """

    param_count: int
    on_disk_bytes: int
    #: The dtype the checkpoint is stored in, e.g. ``bfloat16``. Empty when unknown.
    stored_dtype: str = ""
    #: ``quantization_config.quant_method`` when there is one: ``mxfp4``, ``fp8``,
    #: ``compressed-tensors``, ``awq``, ``gptq``. Empty on an unquantized checkpoint.
    quant_method: str = ""
    #: ``config.expert_dtype`` when an MoE checkpoint stores its routed experts narrower than the rest
    #: of itself, e.g. ``fp4`` beside a ``fp8`` scheme. Empty everywhere else.
    expert_dtype: str = ""
    #: Stored elements per safetensors dtype tag, straight from the headers. For a packed checkpoint
    #: these are containers rather than parameters; :func:`logical_param_count` unpacks them.
    elements_by_dtype: dict[str, int] = field(default_factory=dict)
    #: Which rung of the ladder produced this: ``safetensors-metadata``, ``safetensors-index``,
    #: ``file-sizes``, ``config-count`` or ``dense-guess``. The last two are lower bounds.
    source: str = ""

    @property
    def is_quantized(self) -> bool:
        return bool(self.quant_method)

    @property
    def stored_elements(self) -> int:
        """Total stored elements, i.e. what the safetensors headers report. Not the parameter count."""
        return sum(self.elements_by_dtype.values())

    def bytes_for_load(self, load_dtype: str = "auto", *, dequantizes: bool = True) -> int:
        """Bytes on the device after loading at ``load_dtype``.

        **This is the correction that matters.** The obvious implementation returns
        ``on_disk_bytes``, and it is wrong in both directions:

        - ``EagerModel``'s dtype default is ``"float32"``, not ``"auto"``. Loading a bf16 checkpoint
          eagerly at the default therefore costs ``param_count x 4`` -- double the file size. An
          estimator that returns on-disk bytes says a 12B model fits a 24 GiB card, and it does not.
        - A quantized checkpoint asked for at a float dtype **dequantizes**. gpt-oss-20b is 12.8 GiB
          of MXFP4 on disk and ~41 GiB once transformers expands it to bf16, which is the difference
          between fitting a 48 GiB card and needing an 80 GiB one. transformers also does this
          *silently*, by warning, whenever the ``kernels`` Triton path is unavailable -- so the same
          arithmetic prices the accident as well as the request.

        ``"auto"`` means "as stored", which is the only case where the file size is the answer.

        ``dequantizes`` is what the two loaders disagree about, and getting it wrong is a 3x error in
        whichever direction the caller was not expecting. To transformers, ``dtype`` is the dtype the
        *weights* are materialized in, so asking bf16 of an MXFP4 checkpoint expands it. To vLLM, the
        same argument sets the **activation** dtype and the weights stay packed -- a quantized
        checkpoint served by vLLM costs its file size whatever dtype was requested. Four live
        Neuronpedia pods pass ``--model_dtype bfloat16`` against MXFP4 and FP8 checkpoints for exactly
        that reason, and pricing them as dequantized put gpt-oss-20b at 41.18 GiB instead of 12.82,
        which reads as "does not fit an A40" for a configuration that fits it comfortably.

        Default True because it is the pessimistic reading, and because the accident this function
        exists to price -- transformers expanding a checkpoint because its kernels are missing -- is a
        transformers accident.
        """
        wanted = dtype_bytes_or_none(load_dtype)
        if load_dtype in ("auto", "", None) or wanted is None:
            return self.on_disk_bytes
        if self.is_quantized:
            if not dequantizes:
                return self.on_disk_bytes
            native = _scheme_width(self.quant_method)
            if native is not None and wanted <= native:
                # Asking for the width it is already stored at, or narrower than transformers will
                # give you: the checkpoint is served natively and the file size stands.
                return self.on_disk_bytes
            return int(self.param_count * wanted)
        return int(self.param_count * wanted)

    def quant_family(self) -> str:
        """The numeric format this checkpoint really uses: ``fp8``, ``nvfp4``, ``int4``, ``int8``, or empty.

        ``quant_method`` is not this, and the difference decides whether a given card can run the
        checkpoint at all. ``compressed-tensors`` is a *container* format that carries fp8, int8 and
        int4 alike, so ``Meta-Llama-3.1-8B-Instruct-FP8`` reports a method with no width in it. Asking
        whether the string contains "fp8" then only works because the repo happens to be *named* FP8 --
        rename it and an Ampere card would accept a checkpoint it can only emulate.

        So the declared method is used where it names a format, and the tensor dtypes settle it where
        it does not. The headers cannot be renamed.
        """
        declared = (self.quant_method or "").lower()
        # `mxfp4` is kept distinct from `nvfp4` rather than folded in with it, even though both are
        # 4-bit and both want recent tensor cores. They fail differently, and the difference is the
        # whole point of the distinction: an MXFP4 checkpoint on a card without the kernels loads
        # anyway, dequantized to bf16 at ~3x the size, so an old card CAN give ground truth for it --
        # and that record is worth having, because the silent 3x is the trap people hit. NVFP4 without
        # the hardware has no such fallback.
        for family in ("nvfp4", "mxfp4", "fp8", "int4", "int8"):
            if family in declared:
                return family
        if any(tag in declared for tag in ("fp4", "nf4", "awq", "gptq", "uint4")):
            return "int4"
        return _scheme_from_headers(self.elements_by_dtype) if declared else ""

    def dequantized_bytes(self) -> int | None:
        """What this checkpoint costs if its quantization silently falls back to bf16, or None.

        The gpt-oss / MXFP4 trap as a number rather than a warning: transformers needs ``kernels``,
        Triton >= 3.4 and compute capability >= 7.5, and if any is missing it *warns* and expands the
        weights instead of failing. Needs the parameter count to be logical rather than stored, which
        is what :func:`logical_param_count` is for.
        """
        if not self.is_quantized:
            return None
        return int(self.param_count * 2)


@contextlib.contextmanager
def _no_progress_bars() -> Iterator[None]:
    """Silence ``huggingface_hub``'s per-shard progress bars for the duration.

    Reading a few KB of safetensors headers draws one bar per shard, which is noise in a library call
    inside someone else's CLI. Whatever the caller had set is restored, and a hub version that moves
    these helpers is tolerated rather than fatal -- they are a courtesy, not a requirement.
    """
    try:
        from huggingface_hub.utils.tqdm import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:
        yield
        return
    was_disabled = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        yield
    finally:
        if not was_disabled:
            enable_progress_bars()


def _quant_method(config: Any) -> tuple[str, str]:
    """``(quant_method, stored_dtype)`` from a config, looking in both the root and the text config.

    A multimodal wrapper keeps ``quantization_config`` at the root while the dims live in
    ``text_config``, so both are searched. The routed experts' own dtype is a separate field and a
    separate reader -- see :func:`_expert_dtype`.
    """
    from interp_engine.facts import text_config

    method = ""
    for holder in (config, text_config(config)):
        q = getattr(holder, "quantization_config", None)
        if isinstance(q, dict):
            method = str(q.get("quant_method") or q.get("quantization") or "")
        elif q is not None:
            method = str(getattr(q, "quant_method", "") or getattr(q, "quantization", "") or "")
        if method:
            break
    stored = ""
    for holder in (config, text_config(config)):
        # `dtype` first: transformers v5 renamed `torch_dtype` and warns loudly on every read of the
        # old name, which turns a sizing call into a wall of deprecation notices.
        dt = getattr(holder, "dtype", None) or getattr(holder, "torch_dtype", None)
        if dt is not None:
            stored = str(dt).replace("torch.", "")
            break
    return method.lower(), stored


def _expert_dtype(config: Any) -> str:
    """The dtype of the ROUTED EXPERT weights, which need not be the checkpoint's dtype or its scheme.

    An MoE family may store its experts narrower than everything around them and say so in a field of
    its own rather than in ``quantization_config``, because the mixed precision is a property of the
    checkpoint and not of one quantization method. DeepSeek-V4-Flash is the case that matters:
    ``quant_method: fp8`` with ``expert_dtype: fp4``, and the routed experts are the large majority of
    its parameters. ``vllm_capture.static`` reads the same field for the same reason.
    """
    from interp_engine.facts import text_config

    for holder in (config, text_config(config)):
        value = getattr(holder, "expert_dtype", None)
        if value:
            return str(value).replace("torch.", "").lower()
    return ""


#: Files that name a checkpoint's quantization when ``config.json`` does not, with the key path to
#: the scheme inside each. NVIDIA's ModelOpt exports write ``hf_quant_config.json`` and leave
#: ``config.json`` with no ``quantization_config`` at all, so a reader that only knows about the
#: config concludes the checkpoint is dense -- see :func:`_hub_quant_method`.
_QUANT_SIDECARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hf_quant_config.json", ("quantization", "quant_algo")),
    ("quantization_config.json", ("quant_method",)),
)

#: Where the same files declare that the **KV cache** is quantized too, which is a separate decision
#: from the weights and is made by the checkpoint rather than by the caller. See
#: :func:`hub_kv_quant_algo` for what missing it costs.
_KV_QUANT_SIDECARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hf_quant_config.json", ("quantization", "kv_cache_quant_algo")),
)


def _sidecar_value(hf_model_id: str, token: str | None, sources: tuple[tuple[str, tuple[str, ...]], ...]) -> str:
    """First string found at any of ``sources``, or ``""``.

    A miss is cheap and silent: no sidecar means an empty string and the caller falls back to what it
    already knew.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return ""

    with _no_progress_bars():
        for filename, path in sources:
            for local_only in (True, False):
                try:
                    local = hf_hub_download(hf_model_id, filename, local_files_only=local_only, token=token)
                except Exception:
                    continue
                try:
                    with open(local, encoding="utf-8") as handle:
                        node: Any = json.load(handle)
                except (OSError, ValueError):
                    break
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                if isinstance(node, str) and node:
                    return node.lower()
                break
    return ""


def hub_kv_quant_algo(hf_model_id: str, token: str | None = None) -> str:
    """The scheme the checkpoint stores its **KV cache** in, when it declares one.

    Separate from the weight scheme because the two differ: ``nvidia/Llama-3.3-70B-Instruct-FP4``
    packs its weights at NVFP4 and its KV cache at FP8, and says so in ``hf_quant_config.json`` as
    ``quantization.kv_cache_quant_algo``. vLLM reads that and builds a one-byte cache.

    Missing it costs a factor of two in the *conservative* direction, which is why it went unnoticed:
    that checkpoint on a B200 was predicted to hold 394,295 tokens and built 784,896. Nothing OOMs,
    but the sizer reports half the concurrency the card really gives -- on exactly the checkpoints
    Blackwell is bought to run. Measured as a KV ratio of 1.99 in ``VERIFIED.md``.
    """
    return _sidecar_value(hf_model_id, token, _KV_QUANT_SIDECARS)


def _hub_quant_method(hf_model_id: str, token: str | None) -> str:
    """The quantization scheme from a sidecar file, for repos that declare it outside ``config.json``.

    ``nvidia/Llama-3.3-70B-Instruct-FP4`` is the case that forced this: its ``config.json`` has no
    ``quantization_config``, so a config-only reader sees a dense bf16 70B. Its real scheme lives in
    ``hf_quant_config.json`` as ``quantization.quant_algo = "NVFP4"``. Missing it has two costs -- the
    dequantized size comes out 43% low, and a harness deciding whether *this card* can give ground
    truth for FP4 sees no mention of FP4 and runs the spec anyway.

    A miss is cheap and silent: no sidecar means an empty string, and the caller falls through to
    reading the headers.
    """
    return _sidecar_value(hf_model_id, token, _QUANT_SIDECARS)


def _scheme_from_headers(elements_by_dtype: dict[str, int]) -> str:
    """The scheme implied by the tensor dtypes themselves, when nothing in the repo declares one.

    The last resort, and deliberately so: the safe reading of an undeclared checkpoint is that it is
    dense, and guessing "quantized" wrongly would over-state a dequantized load. But the headers are
    real evidence, and there is one shape they can only have for one reason -- most of the model's
    elements sitting in a *byte* container, which no dense checkpoint does. bf16 weights are ``BF16``;
    a stray ``U8`` buffer is a handful of elements, not two thirds of them.

    Requires the container buckets to hold a **majority** of the elements, so a small integer
    side-table cannot make a dense model look packed. Distinguishing 4-bit from 8-bit is then the
    presence of fp8 scales beside a byte payload, which is what every 4-bit export on the Hub looks
    like; both answers are labels for :func:`_scheme_width`, not claims about a specific vendor
    format, so ``"nvfp4"`` here means "packed two-to-a-byte" rather than NVIDIA's exact encoding.
    """
    total = sum(elements_by_dtype.values())
    if not total:
        return ""
    byte_packed = sum(count for tag, count in elements_by_dtype.items() if str(tag).upper() in {"U8", "I8"})
    fp8 = sum(count for tag, count in elements_by_dtype.items() if str(tag).upper() in _FP8_TAGS)
    if byte_packed > total / 2:
        return "nvfp4" if fp8 else "int4"
    if fp8 > total / 2:
        return "fp8"
    return ""


def _config_param_count(config: Any) -> int:
    """Logical parameter count from config dims, counting MoE experts. A **lower bound**.

    Counts attention, MLP (dense, routed and shared) and embeddings, so it misses norms, biases,
    quantization scales, MTP heads and attention sinks. Across the families on hand it lands between
    0.48x and 1.0x of the truth, the low end being the MoE checkpoints that carry the most of what
    this does not count. Under-counting weights is the direction that OOMs, which is why the hub is
    asked first and this is only the second rung.
    """
    from interp_engine import facts

    f = facts.resolve_facts(config)
    if not f.n_layers or not f.d_model:
        return 0
    cfg = facts.text_config(config)
    inter = int(getattr(cfg, "intermediate_size", 0) or 0) or 4 * f.d_model
    moe_inter = int(getattr(cfg, "moe_intermediate_size", 0) or 0) or inter
    q_dim = f.n_heads * f.head_dim
    kv_dim = f.n_kv_heads * f.head_dim
    attn = f.d_model * (q_dim + 2 * kv_dim) + q_dim * f.d_model
    dense_mlp = 3 * f.d_model * inter
    routed_mlp = f.n_experts * 3 * f.d_model * moe_inter
    shared_mlp = f.n_shared_experts * 3 * f.d_model * moe_inter
    n_sparse = len(f.moe_layers) if f.n_experts else 0
    # Gemma-4 keeps the dense MLP on its sparse layers and adds the experts beside it, so there the
    # dense branch is paid on every layer rather than only on the ones the experts did not replace.
    n_dense = f.n_layers if f.dense_mlp_beside_experts else f.n_layers - n_sparse
    embeddings = f.vocab_size * f.d_model * (1 if f.tied_embeddings else 2)
    return int(f.n_layers * attn + n_dense * dense_mlp + n_sparse * (shared_mlp + routed_mlp) + embeddings)


def _hub_weight_bytes(hf_model_id: str, token: str | None) -> WeightBytes | None:
    """Weights from the Hub's own metadata, without downloading a single shard.

    Two independent facts are wanted and they come from different places:

    - **Logical parameter count**, from the safetensors headers. ``huggingface_hub`` reads these with
      range requests, so the cost is a few KB per shard rather than the shard.
    - **On-disk bytes**, from the shard index's ``metadata.total_size`` when there is an index, else
      by summing the file sizes the repo API reports.

    The parameter count alone is not enough, and this is worth being explicit about: for a packed
    4-bit checkpoint the per-dtype breakdown reports the blocks as ``U8``, and reading those as one
    byte each overstates gpt-oss-20b by about 9 GiB. The index's ``total_size`` is true bytes and is
    what settles it -- 12.82 GiB, matching what the checkpoint actually occupies.

    Both are also reachable **without a token even on a gated repo**, which is why the sizer can
    size a gated model for someone who has not accepted its licence. Only ``config.json`` is gated,
    and that is the dims rather than the weights.
    """
    try:
        from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
    except ImportError:
        return None

    param_count = 0
    by_dtype: dict[str, int] = {}
    source = ""
    with _no_progress_bars():
        try:
            meta = get_safetensors_metadata(hf_model_id, token=token)
            by_dtype = {str(k): int(v) for k, v in (meta.parameter_count or {}).items()}
            param_count = sum(by_dtype.values())
            source = "safetensors-metadata"
        except Exception:
            param_count = 0

    on_disk = 0
    for local_only in (True, False):
        try:
            path = hf_hub_download(
                hf_model_id, "model.safetensors.index.json", local_files_only=local_only, token=token
            )
        except Exception:
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                total = (json.load(handle).get("metadata") or {}).get("total_size")
        except (OSError, ValueError):
            total = None
        if total:
            on_disk = int(total)
            source = source or "safetensors-index"
            break

    if not on_disk:
        # No index: a single-file checkpoint, or a repo that ships none. The file sizes the repo API
        # reports are exact and need no token even when the repo is gated.
        try:
            info = HfApi().model_info(hf_model_id, files_metadata=True, token=token)
            shards = [
                sibling
                for sibling in (info.siblings or [])
                if str(sibling.rfilename).endswith(".safetensors") and "/" not in str(sibling.rfilename)
            ]
            on_disk = sum(int(sibling.size or 0) for sibling in shards)
            if on_disk:
                source = source or "file-sizes"
        except Exception:
            on_disk = 0

    if not param_count and not on_disk:
        return None
    if not on_disk and by_dtype:
        # No index and no file sizes, but the headers are enough: element counts times their own
        # widths IS the stored size, exactly.
        on_disk = int(
            sum(
                count * dtype_bytes(tag, default=_CONTAINER_DTYPES.get(str(tag).upper(), 2.0))
                for tag, count in by_dtype.items()
            )
        )
    if not param_count and on_disk:
        # Bytes but no headers. Assume the stored width so the count is at least self-consistent;
        # `stored_dtype` is filled in by the caller, which has the config.
        param_count = int(on_disk / 2)
    return WeightBytes(
        param_count=param_count,
        on_disk_bytes=on_disk,
        elements_by_dtype=by_dtype,
        source=source or "hub",
    )


def weight_bytes(
    *,
    config: Any = None,
    hf_model_id: str | None = None,
    token: str | None = None,
) -> WeightBytes:
    """Resolve a checkpoint's weights, preferring exactness over speed.

    The ladder, best first, each rung degrading to a wider safety margin rather than to a wrong
    number: the Hub's own metadata, then a config-derived parameter count, then a dense-transformer
    guess. ``source`` records which rung answered, so a caller can widen its margin when the answer
    came from arithmetic rather than from the checkpoint.

    ``token`` is only ever needed for a gated repo's ``config.json``; the weight metadata is public
    even where the weights are not.
    """
    resolved: WeightBytes | None = None
    if hf_model_id:
        resolved = _hub_weight_bytes(hf_model_id, token)

    method, stored = _quant_method(config) if config is not None else ("", "")
    experts = _expert_dtype(config) if config is not None else ""

    # Three places name the scheme, and a checkpoint may use any one of them: `config.json`, a
    # sidecar file, or nothing but its own tensor dtypes. Asked in that order, cheapest and most
    # authoritative first -- and the fall-through matters, because a checkpoint whose scheme goes
    # unrecognized is priced as though it were dense, which is 43% optimistic on a 4-bit 70B.
    if not method and hf_model_id:
        method = _hub_quant_method(hf_model_id, token)
    if not method and resolved is not None:
        method = _scheme_from_headers(resolved.elements_by_dtype)

    if resolved is not None:
        # Only now, with the scheme known, can the header's element counts be turned into parameters.
        logical = (
            logical_param_count(resolved.elements_by_dtype, method, experts)
            if resolved.elements_by_dtype
            else resolved.param_count
        )
        return WeightBytes(
            param_count=logical or resolved.param_count,
            on_disk_bytes=resolved.on_disk_bytes,
            stored_dtype=stored,
            quant_method=method,
            expert_dtype=experts,
            elements_by_dtype=resolved.elements_by_dtype,
            source=resolved.source,
        )

    if config is not None:
        counted = _config_param_count(config)
        if counted:
            width = dtype_bytes_or_none(method) or dtype_bytes(stored)
            return WeightBytes(
                param_count=counted,
                on_disk_bytes=int(counted * width),
                stored_dtype=stored,
                quant_method=method,
                expert_dtype=experts,
                source="config-count",
            )

    return WeightBytes(
        param_count=0,
        on_disk_bytes=0,
        stored_dtype=stored,
        quant_method=method,
        expert_dtype=experts,
        source="unknown",
    )


# ---------------------------------------------------------------------- model facts


@dataclass(frozen=True)
class ModelMemoryFacts:
    """Everything about a model that VRAM arithmetic needs, and nothing else.

    Deliberately a narrow projection of :class:`~interp_engine.facts.ModelFacts` plus the weight
    sizing, rather than a second model-knowledge module. Two fields are here that ``ModelFacts`` does
    not carry, because nothing but memory arithmetic wants them: the MLP widths.
    """

    model_id: str
    weights: WeightBytes
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    v_head_dim: int
    vocab_size: int
    intermediate_size: int
    #: Routed experts per sparse layer; 0 on a dense trunk. Here for the static tap widths --
    #: ``router_logits`` is as wide as the expert bank -- and because it is what says a layer's MLP is
    #: a fused kernel rather than three Linears, which decides whether ``mlp_act`` exists at all.
    n_experts: int = 0
    #: Per-layer attention kinds, when the config states them. Sliding-window layers cache far less
    #: KV than full-attention ones, and on a 5:1 trunk like gemma-3 that is most of the cache.
    layer_types: tuple[str, ...] | None = None
    sliding_window: int | None = None
    #: Parallel residual streams. >1 multiplies every static tap buffer by this.
    n_residual_streams: int = 1
    #: The model's advertised context, used as the ``max_model_len`` default.
    max_position_embeddings: int = 0
    architecture: str = ""
    #: The scheme the **checkpoint** stores its KV cache in, when it declares one -- ``"fp8"`` on
    #: NVIDIA's FP4 exports. Not the caller's ``kv_cache_dtype``: this is a property of the weights on
    #: disk, and vLLM honours it whether or not anyone asked. See :func:`hub_kv_quant_algo`.
    kv_quant_algo: str = ""

    @property
    def kv_width(self) -> int:
        """KV-cache elements per token per layer, K and V together."""
        return kv_cache_width(
            n_kv_heads=self.n_kv_heads, head_dim=self.head_dim, v_head_dim=self.v_head_dim, d_model=self.d_model
        )

    @property
    def full_attention_layers(self) -> int:
        """Layers that cache the whole context. Everything else caches a window."""
        if not self.layer_types:
            return self.n_layers
        return sum(1 for kind in self.layer_types[: self.n_layers] if "sliding" not in kind and "linear" not in kind)

    @property
    def trunk_dims_known(self) -> bool:
        """Whether the config was actually read, as opposed to defaulted to zeros.

        Weight bytes and trunk dims come from different places -- file sizes need no token, while
        ``config.json`` on a gated or private repo does -- so a repo can be sized to the byte while
        nothing at all is known about its attention. ``meta-models/Muse-Glimmer-30B`` is exactly that:
        55.46 GiB of weights resolved, every dim zero.

        This exists because the arithmetic downstream fails *quietly* in that state rather than
        loudly. Guards written to avoid dividing by zero used to turn the unknown into a very small
        number -- one layer, two elements, two bytes -- and 4 bytes per token divides into a 16 GiB
        budget as four billion tokens of context. The estimate then reported that everything fits with
        unlimited concurrency, which is the most dangerous answer available. Anything that needs the KV
        term has to check this and decline instead.
        """
        return self.n_layers > 0 and self.d_model > 0


def kv_cache_width(*, n_kv_heads: int = 0, head_dim: int = 0, v_head_dim: int = 0, d_model: int = 0) -> int:
    """KV-cache elements per token per layer, K and V together, for the whole model.

    ``2 * d_model`` -- the fallback when a caller has no head dims -- is the *pre-GQA* worst case and
    is wrong by 8x on the models where sizing is tight: Llama-3.3-70B caches 8 kv heads of 128 rather
    than 8192, and a DeepSeek MLA trunk one 512-wide latent head rather than 4096. That factor is the
    difference between fitting a default capture size and refusing to start, so pass the head dims.

    Mirrors ``vllm_capture/static.kv_cache_width`` and is checked against it in the tests; this copy
    exists so that sizing a model needs neither torch nor vLLM.
    """
    heads = max(int(n_kv_heads), 0)
    if heads <= 0 or int(head_dim) <= 0:
        return 2 * max(int(d_model), 1)
    return heads * (int(head_dim) + (int(v_head_dim) or int(head_dim)))


def kv_shards(facts: ModelMemoryFacts, num_gpus: int) -> int:
    """How many ways tensor parallelism actually divides the KV cache.

    **Not the rank count**, which is the assumption to get rid of. vLLM shards the cache by *KV
    head*, so how far it divides is a property of the attention shape rather than of the machine:
    Llama-3.3-70B's 8 KV heads go 2-per-rank at TP=4 and the cache really is a quarter on each card,
    while a DeepSeek MLA trunk caches a single 512-wide latent head, which cannot be cut at all --
    vLLM replicates it, and four cards hold four copies of the same cache.

    ``min`` covers both ends, including the case past the second one: vLLM pads a head count up to
    the rank count by duplicating heads, so 8 heads across 16 ranks still costs what 8 ranks cost
    rather than half as much.

    One when the head dims were never read, matching :func:`kv_cache_width`'s ``2 * d_model``
    fallback. That figure is a whole-model worst case with no head structure behind it, and dividing
    a worst case by a rank count would invent exactly the precision the fallback exists because we
    lack.
    """
    tp = max(int(num_gpus), 1)
    if facts.n_kv_heads <= 0 or facts.head_dim <= 0:
        return 1
    return min(tp, facts.n_kv_heads)


def kv_bytes_per_token(facts: ModelMemoryFacts, *, kv_dtype: str = "auto", model_dtype: str = "bfloat16") -> float:
    """KV bytes for one token of context, across every layer.

    The flat, model-wide figure: every layer charged for the full context. This is the **pessimistic**
    one, and it is what the fit floor uses, because a trunk whose windowing vLLM does not exploit
    still has to hold it.

    **Zero means unknown, not free.** When the config could not be read there is no honest number here,
    and returning a nominal one is worse than returning nothing: see
    :attr:`ModelMemoryFacts.trunk_dims_known`. Callers must treat 0 as "cannot size this" rather than
    dividing by it or adding it to a total.
    """
    if not facts.trunk_dims_known:
        return 0.0
    if kv_dtype not in ("auto", "", None):
        width = dtype_bytes(kv_dtype)
    else:
        # `auto` means "whatever the engine will do", and for a checkpoint that ships a quantized KV
        # cache that is the checkpoint's scheme rather than the model dtype -- vLLM honours the
        # declaration whether or not the caller mentioned it. Narrower of the two wins, since an
        # explicit request cannot widen what the weights already store.
        declared = dtype_bytes_or_none(facts.kv_quant_algo)
        model = dtype_bytes(model_dtype)
        width = min(declared, model) if declared is not None else model
    return facts.n_layers * max(facts.kv_width, 1) * width


def kv_bytes_for_context(
    facts: ModelMemoryFacts,
    max_model_len: int,
    *,
    kv_dtype: str = "auto",
    model_dtype: str = "bfloat16",
) -> float:
    """KV bytes vLLM will actually spend to serve ``max_model_len`` tokens of context.

    **Sliding-window layers get no discount here, and that is a measurement rather than a
    simplification.** The natural model says a sliding layer only ever caches its window, so a hybrid
    trunk should cost far less than a uniform one: gemma-3 runs five sliding layers to every global
    one, so at a 4k context with a 512-token window the windowed arithmetic is 3.85x cheaper than
    charging every layer for the full context. That model is wrong about vLLM.

    Measured on ``google/gemma-3-1b-pt``, A40, 4k context, utilization 0.9, with 37.58 GiB left for the
    cache after weights and context::

        charging every layer the full context   1,515,475 tokens
        crediting the 22 sliding layers            5,837,386 tokens
        what vLLM actually built                1,399,779 tokens

    So the flat figure is 8% optimistic and the windowed one is **4.2x** optimistic -- and 4.2x in the
    optimistic direction on the KV cache is a pod that accepts a workload it cannot hold. vLLM's hybrid
    allocator has to page whole blocks across both layer groups, and the capacity it reports is
    governed by the full-attention group.

    The residual 8% is charged as :data:`CALIBRATION`'s ``hybrid_kv_overhead`` on a mixed trunk, which
    brings the same case to 0.98x -- conservative. A uniform trunk needs no such term: gpt2 came in at
    1.01x and Qwen3-4B at 1.00x against the flat figure.
    """
    flat = kv_bytes_per_token(facts, kv_dtype=kv_dtype, model_dtype=model_dtype) * max_model_len
    if facts.layer_types and facts.full_attention_layers < facts.n_layers:
        return flat * _cal("hybrid_kv_overhead")
    return flat


def eager_activation_bytes(
    facts: ModelMemoryFacts,
    *,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype: str = "float32",
    n_capture_points: int = 0,
    requires_grad: bool = False,
    attn_implementation: str = "eager",
    logits_fp32: bool = True,
) -> dict[str, int]:
    """Peak activation bytes for one eager forward, per term.

    **Weights are not what OOMs the eager backend**, and this is the function that says why. Two
    terms grow with the prompt rather than with the model, and both are invisible in any
    weights-only estimate:

    - **The logits.** A forward materializes ``[batch, seq, vocab]``, and most families then upcast
      that to fp32. On a 262k-vocab model at 8k tokens that pair is ~12 GiB, against ~24 GiB of bf16
      weights for a 12B checkpoint. This is the term behind "it worked on a short prompt".

      Whether the upcast happens is a **per-family** matter, and the one measurement on hand says a
      family may skip it. Qwen3-4B at a 32,752-token prompt on an A40 peaked at 17.45 GiB of torch
      memory against 7.5 GiB of weights, so its ~10 GiB of activations is the bf16 copy alone
      (``32752 x 151936 x 2`` is 9.3 GiB) and nothing was upcast. No run here positively confirms a
      family that *does* upcast, so ``logits_fp32=True`` is a conservative default rather than a
      measured one: it makes this 1.2-1.8x pessimistic where the upcast is absent, and the cost of
      that is one rung of prompt length against an OOM for the opposite error. Pass
      ``logits_fp32=False`` when a family is known not to upcast.
    - **The attention matrix, quadratically.** ``load_model`` defaults ``attn_implementation`` to
      ``"eager"`` for the eager backend, which materializes ``[batch, heads, q, k]`` per layer plus a
      softmax temporary. At 8k tokens on 16 heads that is ~4 GiB *per layer*. ``"sdpa"`` or
      ``"flash_attention_2"`` removes the term outright, which is usually the cheapest fix available
      to a caller who has hit this.

    ``requires_grad=True`` is the third trap: the graph retains every layer's activations rather than
    one layer's, so the per-layer terms stop being transient and multiply by depth. A few hundred
    tokens will OOM a card that generates the same text fine.
    """
    width = dtype_bytes(dtype)
    tokens = max(batch_size, 1) * max(seq_len, 1)

    logits = int(tokens * max(facts.vocab_size, 1) * width)
    if logits_fp32 and width < 4:
        # The upcast holds both copies at once; `logits.float()` is a new tensor.
        logits += int(tokens * max(facts.vocab_size, 1) * 4)

    attn = 0
    if "eager" in (attn_implementation or "").lower():
        # Scores plus the softmax result, and two layers' worth: a layer's matrix is freed only once the
        # next has allocated, under the allocator states that matter here.
        #
        # A sliding-window layer attends over its window rather than the whole prompt, so its matrix is
        # `seq x window`, not `seq x seq`. That distinction is worth making even though it cuts an
        # estimate down, because on gemma-3 it is 40 layers out of 48 and squaring the wrong number at
        # 32k tokens is the difference between 128 GiB and 8 -- which turns a configuration that runs
        # fine into one this module refuses. Unlike the KV cache, where the same reasoning was measured
        # to be wrong about vLLM, this is plain PyTorch: the mask decides the shape and the shape is
        # what gets allocated.
        heads = max(facts.n_heads, 1)
        rows = max(seq_len, 1)
        full_layers = facts.full_attention_layers
        windowed_layers = max(facts.n_layers - full_layers, 0)
        window = min(int(facts.sliding_window or rows), rows)
        widest = rows if full_layers else window
        per_layer = max(batch_size, 1) * heads * rows * widest * width
        attn = int(per_layer * 2 * 2)
        if not full_layers and not windowed_layers:
            attn = int(max(batch_size, 1) * heads * rows * rows * width * 2 * 2)

    hidden = int(tokens * max(facts.d_model, 1) * width)
    mlp = int(tokens * max(facts.intermediate_size, 4 * max(facts.d_model, 1)) * width * 2)
    capture = int(max(n_capture_points, 0) * tokens * max(facts.d_model, 1) * width)

    if requires_grad:
        # Every layer's activations are retained rather than one layer's. The residual stream and the
        # MLP intermediate are the two that dominate; the attention matrix, if present, is retained
        # per layer as well, which is why gradients plus eager attention is the worst combination.
        depth = max(facts.n_layers, 1)
        hidden *= depth
        mlp *= depth
        if attn:
            attn = int(attn / 2 * depth)

    workspace = int(_cal("eager_workspace_gib") * GIB)
    return {
        "logits": logits,
        "attention": attn,
        "hidden_states": hidden,
        "mlp_intermediate": mlp,
        "capture_buffers": capture,
        "workspace": workspace,
    }


# --------------------------------------------------------------------- reservations


@dataclass(frozen=True)
class Reservations:
    """VRAM the engine must not touch, split by how it scales.

    The split is the whole point. A number that is replicated on every rank and a number that exists
    once behave completely differently at ``tensor_parallel_size > 1``, and conflating them is how a
    70B on two cards came to be sized against half the memory it actually needed.

    Everything here is charged **outside** vLLM's pool, because that is where it lands in practice:
    vLLM measures the device once at startup, and anything allocated after that -- which is the
    normal case, since a caller loads the engine before its own weights -- sits on top of the pool
    rather than inside it.
    """

    #: Bytes replicated on **every** GPU. The Jacobian lens is the case that matters: the read-out
    #: holds a ``[d_model, d_model]`` matrix per layer on each worker device, unsharded, so a 70B at
    #: TP=2 pays it twice. ``worker_set_lens_jacobians`` returns the byte count for exactly this.
    per_rank_bytes: int = 0
    #: Bytes that exist once, on the first GPU. A preloaded SAE cache in the serving process is the
    #: usual one. Charged to rank 0 only, so a multi-GPU fit is not penalized on every card.
    host_bytes: int = 0
    #: Peak bytes concurrent requests need outside the pool, over and above the two above. Scales
    #: with the caller's own per-request working set, which the engine cannot see.
    transient_bytes: int = 0
    #: Whether the caller allocates this **before** the engine starts.
    #:
    #: This changes which side of the line it lands on, and it is easy to get backwards. vLLM sizes its
    #: KV cache as ``card x utilization - what the process is already using``, so memory allocated
    #: *before* startup is charged against the pool and simply shrinks the cache. Memory allocated
    #: *after* startup sits on top of a pool vLLM has already filled, and eats the margin instead.
    #:
    #: Default False, which is both the common case and the dangerous one: a caller loads the engine
    #: and then loads their SAEs. That is the ordering where a reservation can OOM a process whose KV
    #: cache size looked perfectly reasonable at startup.
    before_engine: bool = False
    #: What the reservations are for, so an estimate can explain itself.
    note: str = ""

    def for_rank(self, rank: int = 0) -> int:
        return self.per_rank_bytes + self.transient_bytes + (self.host_bytes if rank == 0 else 0)

    @classmethod
    def for_jacobian_lens(cls, facts: ModelMemoryFacts, *, dtype: str = "float32", **kwargs: Any) -> Reservations:
        """Reservations sized for a Jacobian lens on this model.

        ``n_layers x d_model^2 x itemsize`` per rank, which is the shape the read-out actually holds.
        On Llama-3.3-70B in fp32 that is ~10 GiB a card, and it is why a utilization derived without
        this term hands vLLM memory the lens is about to want.
        """
        width = dtype_bytes(dtype)
        per_rank = int(max(facts.n_layers, 1) * max(facts.d_model, 1) ** 2 * width)
        note = f"jacobian lens: {facts.n_layers} layers x {facts.d_model}^2 x {width:g}B per rank"
        return cls(per_rank_bytes=per_rank, note=note, **kwargs)


# ------------------------------------------------------------------- workload specs


@dataclass(frozen=True)
class WorkloadSpec:
    """A configuration to price: the arguments a caller would pass, plus the load they intend.

    Field names match ``load_model`` and the vLLM engine arguments they become, so an estimate can be
    turned back into runnable code without a translation table.
    """

    backend: str = "vllm"
    #: The dtype weights are loaded at. ``"auto"`` means as stored. Note that ``EagerModel``'s own
    #: default is ``"float32"``, not ``"auto"``, which doubles a bf16 checkpoint -- a sizer should
    #: say so rather than reproduce it silently.
    dtype: str = "auto"
    #: KV cache dtype. ``"auto"`` follows the model dtype.
    kv_cache_dtype: str = "auto"
    max_model_len: int = 0
    max_num_batched_tokens: int = 0
    max_num_seqs: int = 0
    gpu_memory_utilization: float = 0.0
    num_gpus: int = 1
    #: Static tap sites, read and write counted together, for ``vllm-static``. ``"auto"`` resolves to
    #: ``resid_post`` read and write at every layer, i.e. ``2 * n_layers``. A raw count prices every
    #: site at the residual width; name the points in :attr:`static_points` to price them at theirs.
    static_sites: int = 0
    #: The static tap points to declare, at every layer, for ``vllm-static``, priced to read *and*
    #: write as ``"auto"`` does -- see :data:`_STATIC_CAPTURE_ONLY`. Empty resolves to ``"auto"``.
    #: Overridden by :attr:`static_sites` where a caller sets both, since a count is the more
    #: specific answer to "how much buffer".
    static_points: tuple[str, ...] = ()
    enforce_eager: bool | None = None
    #: Eager-only: the shape of the forward being priced.
    batch_size: int = 1
    seq_len: int = 0
    requires_grad: bool = False
    attn_implementation: str = ""
    n_capture_points: int = 0

    @property
    def is_vllm(self) -> bool:
        return self.backend in VLLM_BACKENDS

    @property
    def graphs_on(self) -> bool:
        """Whether a CUDA graph pool will be captured.

        ``vllm`` runs ``enforce_eager=True`` by default, because graph replay skips the Python
        forward the capture hooks live on. The two graph backends force it False.
        """
        if self.enforce_eager is not None:
            return not self.enforce_eager
        return self.backend in ("vllm-static", "vllm-generate")

    def resolved_static_points(self, facts: ModelMemoryFacts) -> tuple[str, ...]:
        """The points this spec will really declare, at every layer.

        A named point this trunk refuses is dropped rather than refused, because the caller who
        chose it was looking at a different model a moment ago: switching from Qwen3 to DeepSeek-V4
        should not error, it should fall back to the stack that trunk does carry.
        """
        if self.backend != "vllm-static":
            return ()
        offered = offered_static_points(facts)
        chosen = tuple(point for point in self.static_points if point in offered)
        return chosen or (default_static_point(facts),)

    def resolved_static_sites(self, facts: ModelMemoryFacts) -> int:
        """Static buffers this spec will actually allocate, reads and writes together."""
        if self.backend != "vllm-static":
            return 0
        if self.static_sites:
            return self.static_sites
        points = self.resolved_static_points(facts)
        return max(facts.n_layers, 1) * sum(static_point_buffers(point) for point in points)

    def static_elements(self, facts: ModelMemoryFacts) -> int:
        """Elements per token across every static buffer, which is what the bytes scale with.

        Not ``sites * width``: a named set has no single width, and averaging one is how a set
        carrying ``mlp_act`` comes to look as cheap as a set of residual taps.
        """
        if self.backend != "vllm-static":
            return 0
        if self.static_sites:
            return self.static_sites * _static_width(facts)
        points = self.resolved_static_points(facts)
        return max(facts.n_layers, 1) * sum(static_point_elements(point, facts) for point in points)

    def with_defaults(self, facts: ModelMemoryFacts) -> WorkloadSpec:
        """Fill in the engine's own defaults, so a spec prices what would really happen.

        The vLLM engine arguments are left at zero on the eager backend rather than defaulted, so that
        a spec turned back into code does not carry settings that backend has never heard of.
        """
        from dataclasses import replace

        max_len = self.max_model_len or facts.max_position_embeddings or 4096
        if self.backend == "eager":
            return replace(
                self,
                max_model_len=max_len,
                # On eager there is no paged cache, so `max_model_len` is simply the longest prompt --
                # and the prompt is what the quadratic attention term and the logits scale with. Pricing
                # a 512-token forward for a caller who said 8192 would miss the entire risk.
                seq_len=self.seq_len or max_len,
                dtype=self.dtype or "float32",
                attn_implementation=self.attn_implementation or "eager",
            )
        return replace(
            self,
            max_model_len=max_len,
            max_num_batched_tokens=self.max_num_batched_tokens or (8192 if self.backend == "vllm-static" else 2048),
            max_num_seqs=self.max_num_seqs or 256,
            gpu_memory_utilization=self.gpu_memory_utilization or 0.9,
            dtype=self.dtype or "auto",
        )


# ----------------------------------------------------------------------- estimates


@dataclass(frozen=True)
class MemoryTerm:
    """One line of the budget."""

    name: str
    bytes: int
    #: ``"pool"`` for what vLLM accounts for itself, ``"outside"`` for what it does not, ``"eager"``
    #: for the single-process backend where the distinction does not apply.
    side: str
    note: str = ""

    @property
    def gib(self) -> float:
        return self.bytes / GIB


@dataclass(frozen=True)
class MemoryEstimate:
    """What a configuration costs, whether it fits, and what to change if it does not."""

    spec: WorkloadSpec
    gpu: GpuSpec
    facts: ModelMemoryFacts
    terms: tuple[MemoryTerm, ...]
    fits: bool
    #: Bytes left over. Negative means the shortfall.
    headroom_bytes: int
    #: Context that fits in the KV cache the pool leaves, in tokens. 0 on eager.
    kv_capacity_tokens: int
    advice: tuple[str, ...] = ()
    #: ``"measured"`` when a verification record backs this exact spec, else ``"estimated"``.
    evidence: str = "estimated"
    warnings: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(term.bytes for term in self.terms)

    @property
    def pool_bytes(self) -> int:
        return sum(term.bytes for term in self.terms if term.side == "pool")

    @property
    def outside_bytes(self) -> int:
        return sum(term.bytes for term in self.terms if term.side == "outside")

    @property
    def concurrent_sequences(self) -> int:
        """How many full-length sequences the KV cache holds at once."""
        if not self.spec.max_model_len:
            return 0
        return self.kv_capacity_tokens // self.spec.max_model_len

    def term(self, name: str) -> MemoryTerm | None:
        return next((t for t in self.terms if t.name == name), None)

    def format_table(self) -> str:
        """A per-term breakdown, each side against its own budget.

        The two sides are shown separately on purpose. A single total against the card would hide
        which constraint is binding, and they are fixed by different settings: the pool by
        ``max_model_len`` and the tap set, everything outside it by ``gpu_memory_utilization``.
        """
        rows: list[str] = []
        util = self.spec.gpu_memory_utilization
        eager = self.spec.backend == "eager"

        def block(side: str, title: str, budget: float | None) -> None:
            terms = [t for t in self.terms if t.side == side]
            if not terms:
                return
            used = sum(t.bytes for t in terms) / GIB
            head = f"{title}: {used:.2f} GiB"
            if budget is not None:
                head += f" of {budget:.2f} GiB available ({budget - used:+.2f})"
            rows.append(f"  {head}")
            for term in sorted(terms, key=lambda t: -t.bytes):
                rows.append(f"    {term.name:<22} {term.gib:>7.2f}  {term.note}")

        if eager:
            block("eager", f"on the card (of {self.gpu.total_gib:.2f} GiB)", self.gpu.total_gib)
        else:
            block("outside", f"outside vLLM's pool (1 - {util:g})", (1 - util) * self.gpu.total_gib)
            rows.append("")
            block("pool", f"inside vLLM's pool ({util:g} x card)", util * self.gpu.total_gib)

        rows.append("")
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        rows.append(f"  {verdict}, headroom {self.headroom_bytes / GIB:+.2f} GiB")
        if self.kv_capacity_tokens:
            rows.append(
                f"  KV cache holds ~{self.kv_capacity_tokens:,} tokens "
                f"= {self.concurrent_sequences} sequences of {self.spec.max_model_len:,}"
            )
        return "\n".join(rows)


def static_buffer_bytes(n_sites: int, max_n: int, width: int, *, element_bytes: int = 2) -> int:
    """Static tap buffer bytes: one ``max_n``-row buffer of ``width`` per site.

    Does **not** shard with tensor parallelism -- a ``d_model``-wide tap is replicated on every rank --
    which is why a static set costs the same on eight cards as on one.
    """
    return int(n_sites) * int(max_n) * int(width) * int(element_bytes)


def static_tap_bytes(elements_per_token: int, max_n: int, *, element_bytes: int = 2) -> int:
    """Static tap buffer bytes from a per-token element count summed over every buffer.

    The general form of :func:`static_buffer_bytes`, which assumes one width for every site. A named
    point set does not have one: ``mlp_act`` is ``intermediate_size`` wide where ``resid_post`` is
    ``d_model``, so the sum is taken before the multiply rather than after.
    """
    return int(elements_per_token) * int(max_n) * int(element_bytes)


def _static_width(facts: ModelMemoryFacts) -> int:
    """Elements per token per static site, for a set declared as a raw count rather than by name.

    A hyper-connection trunk carries several parallel residual streams and ``"auto"`` declares the
    whole stack, so the buffer is that many times wider -- four times, on the DeepSeek-V4 block.
    """
    return max(facts.d_model, 1) * max(facts.n_residual_streams, 1)


#: The static tap points this sizer can price, in forward order, each with the trunk dimension its
#: buffer is wide along.
#:
#: Not the whole vocabulary -- ``vllm_capture.static`` serves about twenty-five names, and the ones
#: missing here are missing because their width is not on :class:`ModelMemoryFacts`: the QK-norm
#: points need a per-head shape that depends on how the norm was written, and ``value`` is the fused
#: qkv output on most families rather than the ``d_model`` its name suggests. Pricing one of those at
#: ``d_model`` would be a guess wearing a number's clothes.
#:
#: ``streams`` and ``d_model`` differ only on a hyper-connection trunk, which is also the trunk that
#: refuses ``resid_pre`` / ``resid_post`` outright -- see :func:`offered_static_points`.
_STATIC_POINT_WIDTHS: dict[str, str] = {
    "resid_pre": "d_model",
    "resid_streams": "streams",
    "attn": "qkv",
    "z": "heads",
    "attn_out": "d_model",
    "mlp_act": "neurons",
    "router_logits": "experts",
    "mlp_out": "d_model",
    "resid_post": "d_model",
}

#: Points that allocate a read buffer and nothing else. ``attn`` is refused as a write outright -- it
#: is a copy of the kernel's q/k/v, and there is no meaning to adding to it.
#:
#: Everything else is priced with a ``delta`` beside its ``buf``, which is a choice worth naming.
#: ``"auto"`` declares a write at every read and that is what doubles the default set -- but
#: :func:`~interp_engine.vllm_capture.static.resolve_static_points` leaves an *explicit* list
#: read-only unless ``static_writes`` restates it. Pricing a named set the expensive way is
#: deliberate: a tap set is asked for in order to steer at the same addresses, and quoting the
#: read-only figure to someone who then adds a steer is the direction that OOMs.
_STATIC_CAPTURE_ONLY: frozenset[str] = frozenset({"attn"})

#: Points a hyper-connection trunk refuses, mirroring ``vllm_capture._tree._SINGLE_STREAM_POINTS``:
#: with several streams in flight there is no single residual vector for these to name.
_SINGLE_STREAM_ONLY: frozenset[str] = frozenset({"resid_pre", "resid_mid", "resid_post"})


def offered_static_points(facts: ModelMemoryFacts) -> tuple[str, ...]:
    """The points that can be declared on *this* trunk, in forward order.

    Two trunk properties gate the list, and both are refusals in the engine rather than tidying here.
    A hyper-connection block has no single residual vector, so the three names for one are out and
    ``resid_streams`` is in. And a sparse block's MLP is a fused kernel with no activation tensor to
    tap and a router that has one, so ``mlp_act`` and ``router_logits`` trade places.
    """
    streams = max(facts.n_residual_streams, 1) > 1
    moe = facts.n_experts > 0
    offered = []
    for name in _STATIC_POINT_WIDTHS:
        if name in _SINGLE_STREAM_ONLY and streams:
            continue
        if name == "resid_streams" and not streams:
            continue
        if name == "mlp_act" and moe:
            continue
        if name == "router_logits" and not moe:
            continue
        offered.append(name)
    return tuple(offered)


def default_static_point(facts: ModelMemoryFacts) -> str:
    """What ``static_points="auto"`` resolves to, which the trunk decides rather than the caller."""
    return "resid_streams" if max(facts.n_residual_streams, 1) > 1 else "resid_post"


def static_point_buffers(point: str) -> int:
    """Buffers one point allocates per layer.

    Three for ``attn``, which is not a tensor but a request for the kernel's q, k and v; two for
    everything else, a read and the write ``"auto"`` declares beside it.
    """
    return 3 if point == "attn" else 1 if point in _STATIC_CAPTURE_ONLY else 2


def static_point_elements(point: str, facts: ModelMemoryFacts) -> int:
    """Elements per token that one point costs at one layer, across every buffer it allocates.

    The widths are the engine's, from ``vllm_capture.static._buffer_shape`` and ``_activation_width``:
    the residual and sublayer points are ``d_model``, ``z`` is the o_proj's input and so head-shaped,
    ``mlp_act`` is the down_proj's input and so ``intermediate_size``, ``router_logits`` is the gate's
    output and so as wide as the expert bank, and ``attn`` is three buffers at ``n_heads`` and twice
    ``n_kv_heads`` head widths rather than one at ``d_model``.

    Pricing every point at ``d_model`` is what this exists to stop being: on Qwen3-32B ``mlp_act`` is
    5.0x a residual tap and ``attn`` is 0.6x, so a set of three is anywhere from 1.6x to 11x the
    default depending on which three.
    """
    d_model = max(facts.d_model, 1)
    head_dim = max(facts.head_dim, 1)
    kind = _STATIC_POINT_WIDTHS.get(point)
    if kind == "streams":
        width = d_model * max(facts.n_residual_streams, 1)
    elif kind == "heads":
        width = max(facts.n_heads, 1) * head_dim
    elif kind == "neurons":
        width = max(facts.intermediate_size or 4 * d_model, 1)
    elif kind == "experts":
        width = max(facts.n_experts, 1)
    elif kind == "qkv":
        # One buffer per role, so the three are summed rather than multiplied by `static_point_buffers`.
        width = (max(facts.n_heads, 1) + 2 * max(facts.n_kv_heads, 1)) * head_dim
    else:
        width = d_model
    return width if point in _STATIC_CAPTURE_ONLY else width * 2


def estimate(
    facts: ModelMemoryFacts,
    gpu: GpuSpec,
    spec: WorkloadSpec,
    reservations: Reservations | None = None,
) -> MemoryEstimate:
    """Price a configuration on a card, term by term.

    The vLLM arm splits every term across the utilization line described in the module docstring; the
    eager arm has no such line, so it charges everything against the card. Both add the same
    reservations, and both report the same shape of answer.
    """
    spec = spec.with_defaults(facts)
    res = reservations or Reservations()
    tp = max(int(spec.num_gpus), 1)
    terms: list[MemoryTerm] = []
    warnings: list[str] = []
    advice: list[str] = []

    # Every row in `gpu-sizer/VERIFIED.md` was measured on one card. The single-GPU arithmetic is
    # calibrated against hardware; how it divides across ranks is not, and the two terms it is most
    # likely to be wrong about -- the weights, which TP shards unevenly, and the cache, which it shards
    # for some attention shapes and replicates for others -- are the two largest.
    if tp > 1:
        warnings.append(
            f"{tp}-GPU figures are unverified: every configuration measured so far ran on a single card, "
            f"so tensor parallelism here is arithmetic no hardware has checked"
        )

    # Only the eager backend expands a quantized checkpoint to the requested dtype; vLLM reads the same
    # argument as an activation dtype and serves the packed weights. See `bytes_for_load`.
    weights_total = facts.weights.bytes_for_load(spec.dtype, dequantizes=spec.backend == "eager")
    if not weights_total:
        warnings.append(
            "weight bytes are unknown, so every figure below is only the non-weight terms; "
            "pass hf_model_id or a config to model_memory_facts()"
        )
    if facts.weights.is_quantized:
        dequantized = facts.weights.dequantized_bytes() or 0
        if not gpu.supports_mxfp4_kernels and "fp4" in facts.weights.quant_method:
            warnings.append(
                f"{gpu.name} is compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]}, below the 7.5 "
                f"the MXFP4 Triton path needs, so transformers will warn and dequantize: "
                f"{dequantized / GIB:.1f} GiB rather than {facts.weights.on_disk_bytes / GIB:.1f} GiB"
            )
        elif spec.backend == "eager" and dequantized > weights_total * 1.5:
            # Only on the eager arm, because the accident this prices is a transformers one: the
            # kernels are missing, so the checkpoint is expanded on the way in. vLLM never takes that
            # path -- which is why `weights_total` above is the packed size on that arm -- and a
            # figure four times the real one, attached to a configuration that cannot reach it, reads
            # as a risk rather than as the aside it would be.
            warnings.append(
                f"quantized checkpoint ({facts.weights.quant_method}): {weights_total / GIB:.1f} GiB served "
                f"natively, but ~{dequantized / GIB:.1f} GiB if the kernels are missing and transformers "
                f"dequantizes silently -- install the `quant` extra and pre-cache its Hub kernels"
            )
        if "fp8" in facts.weights.quant_method and not gpu.supports_fp8:
            warnings.append(
                f"{gpu.name} has no FP8 tensor cores, so an FP8 checkpoint runs emulated: correct, but "
                f"slower than the same weights on Ada/Hopper or newer"
            )

    if spec.backend == "eager":
        # One process, no pool: weights land wherever `device_map` puts them, and the activation peak
        # sits beside them on the same card.
        if not facts.trunk_dims_known:
            # Here the unknown term is the activation peak rather than the KV cache, and on eager that
            # is the term that decides it -- the logits alone need the vocabulary. Same refusal.
            warnings.append(
                f"cannot size the activation peak for {facts.model_id}: its config gave no dimensions, "
                f"and on eager the prompt-driven terms usually exceed the weights. Only the weights "
                f"below are real."
            )
        per_card_weights = weights_total // tp
        terms.append(
            MemoryTerm(
                "weights",
                per_card_weights,
                "eager",
                f"{facts.weights.param_count / 1e9:.1f}B params at {spec.dtype}"
                + (f", spread over {tp} GPUs" if tp > 1 else "")
                + f" [{facts.weights.source}]",
            )
        )
        activation = eager_activation_bytes(
            facts,
            batch_size=spec.batch_size,
            seq_len=spec.seq_len,
            dtype=spec.dtype if spec.dtype != "auto" else (facts.weights.stored_dtype or "bfloat16"),
            n_capture_points=spec.n_capture_points,
            requires_grad=spec.requires_grad,
            attn_implementation=spec.attn_implementation or "eager",
        )
        for name, value in activation.items():
            if value:
                terms.append(MemoryTerm(name, value, "eager", _eager_note(name, spec, facts)))
        terms.append(MemoryTerm("cuda_context", int(_cal("cuda_context_gib") * GIB), "eager", "process CUDA context"))
        if res.for_rank(0):
            terms.append(MemoryTerm("reserved", res.for_rank(0), "eager", res.note or "caller reservations"))

        total = sum(term.bytes for term in terms)
        headroom = gpu.total_bytes - total
        fits = headroom >= 0 and facts.trunk_dims_known
        if not fits and facts.trunk_dims_known:
            advice.extend(_eager_advice(activation, spec, facts))
        return MemoryEstimate(
            spec=spec,
            gpu=gpu,
            facts=facts,
            terms=tuple(terms),
            fits=fits,
            headroom_bytes=headroom,
            kv_capacity_tokens=0,
            advice=tuple(advice),
            warnings=tuple(warnings),
        )

    # --- vLLM: two sides of the utilization line ---------------------------------
    if not facts.trunk_dims_known:
        # The KV cache is the whole question on a vLLM backend -- it is what the pool spends whatever
        # the weights leave -- so with no attention dims there is nothing to weigh it against. Said out
        # loud rather than silently priced at zero, because a KV term of zero reads as "it fits".
        warnings.append(
            f"cannot size the KV cache for {facts.model_id}: its config gave no layer or head "
            f"dimensions, so only the weights below are real. A gated or private repo returns file "
            f"sizes without a token but not config.json -- pass one. Every figure that depends on "
            f"the cache is omitted rather than guessed."
        )
    context = int(_cal("cuda_context_gib") * GIB)
    overshoot = int(_cal("vllm_overshoot_gib") * GIB)
    frag = int(_cal("frag_fraction") * gpu.total_bytes)
    reserved_outside = res.for_rank(0) if not res.before_engine else 0
    reserved_inside = res.for_rank(0) if res.before_engine else 0

    terms.append(
        MemoryTerm(
            "vllm_overshoot",
            overshoot,
            "outside",
            "what vLLM allocates past its own budget during warmup",
        )
    )
    terms.append(MemoryTerm("fragmentation", frag, "outside", f"{_cal('frag_fraction'):.0%} of the card"))
    if reserved_outside:
        terms.append(
            MemoryTerm(
                "reserved",
                reserved_outside,
                "outside",
                res.note or "caller reservations, allocated after the engine (SAEs, lens, other processes)",
            )
        )

    # Inside the pool. The CUDA context is here rather than outside because vLLM's budget is measured
    # against what the process is ALREADY using -- see CALIBRATION["cuda_context_gib"].
    terms.append(MemoryTerm("cuda_context", context, "pool", "process CUDA context, charged against vLLM's budget"))
    if reserved_inside:
        terms.append(
            MemoryTerm(
                "reserved",
                reserved_inside,
                "pool",
                res.note or "caller reservations, allocated before the engine so vLLM sees them as used",
            )
        )

    per_card_weights = weights_total // tp
    terms.append(
        MemoryTerm(
            "weights",
            per_card_weights,
            "pool",
            f"{facts.weights.param_count / 1e9:.1f}B params at {spec.dtype}"
            + (f", sharded over TP={tp}" if tp > 1 else "")
            + f" [{facts.weights.source}]",
        )
    )

    sites = spec.resolved_static_sites(facts)
    elements = spec.static_elements(facts)
    if sites:
        buffers = static_tap_bytes(elements, spec.max_num_batched_tokens)
        named = spec.resolved_static_points(facts)
        terms.append(
            MemoryTerm(
                "static_buffers",
                buffers,
                "pool",
                (
                    f"{', '.join(named)} at {max(facts.n_layers, 1)} layers"
                    if named and not spec.static_sites
                    else f"{sites} sites x {_static_width(facts)} wide"
                )
                + f" = {sites} buffers x {spec.max_num_batched_tokens} rows"
                + (" (not sharded by TP)" if tp > 1 else ""),
            )
        )
    else:
        buffers = 0

    graphs = int(_cal("graph_pool_gib") * GIB) if spec.graphs_on else 0
    if graphs:
        terms.append(MemoryTerm("graph_pool", graphs, "pool", "CUDA graph capture pool"))

    shards = kv_shards(facts, tp)
    kv_floor = int(
        kv_bytes_per_token(facts, kv_dtype=spec.kv_cache_dtype, model_dtype=spec.dtype) * spec.max_model_len / shards
    )
    kv_note = f"one sequence of {spec.max_model_len} tokens, every layer at full context"
    if tp > 1:
        kv_note += f", sharded {shards} ways" if shards > 1 else ", replicated on every rank"
    terms.append(MemoryTerm("kv_cache_floor", kv_floor, "pool", kv_note))

    pool_available = int(spec.gpu_memory_utilization * gpu.total_bytes)
    outside_needed = overshoot + frag + reserved_outside
    pool_needed = context + reserved_inside + per_card_weights + buffers + graphs + kv_floor

    # Both constraints have to hold, and they fail differently -- see the module docstring.
    pool_headroom = pool_available - pool_needed
    outside_headroom = (gpu.total_bytes - pool_available) - outside_needed
    headroom = min(pool_headroom, outside_headroom)
    # An unsizable model is never reported as fitting. `kv_floor` is 0 when the dims are unknown, so
    # the arithmetic above would otherwise weigh the weights against the pool and find room to spare --
    # a confident yes built on the one term nobody could measure. A refusal is not a failure verdict
    # here; the warning above says which it is.
    fits = pool_headroom >= 0 and outside_headroom >= 0 and facts.trunk_dims_known

    kv_room = max(pool_available - context - reserved_inside - per_card_weights - buffers - graphs, 0)
    per_token = (
        kv_bytes_for_context(facts, spec.max_model_len, kv_dtype=spec.kv_cache_dtype, model_dtype=spec.dtype) / shards
    )
    kv_capacity = int(kv_room / (per_token / spec.max_model_len)) if per_token > 0 else 0

    if not fits:
        advice.extend(
            _vllm_advice(
                spec=spec,
                facts=facts,
                gpu=gpu,
                pool_headroom=pool_headroom,
                outside_headroom=outside_headroom,
                weights=per_card_weights,
                buffers=buffers,
            )
        )
    if spec.gpu_memory_utilization > _cal("max_util"):
        warnings.append(
            f"gpu_memory_utilization={spec.gpu_memory_utilization} is above vLLM's own default of "
            f"{_cal('max_util')}; the margin left outside the pool is thinner than the overshoot "
            f"measured during warmup, so a fit here is not reliable"
        )

    return MemoryEstimate(
        spec=spec,
        gpu=gpu,
        facts=facts,
        terms=tuple(terms),
        fits=fits,
        headroom_bytes=headroom,
        kv_capacity_tokens=kv_capacity,
        advice=tuple(advice),
        warnings=tuple(warnings),
    )


def _eager_note(name: str, spec: WorkloadSpec, facts: ModelMemoryFacts) -> str:
    if name == "logits":
        return f"{spec.batch_size}x{spec.seq_len} x vocab {facts.vocab_size:,}, plus the fp32 upcast"
    if name == "attention":
        return f"quadratic in the prompt: {facts.n_heads} heads x {spec.seq_len}^2, attn_implementation='eager'"
    if name == "capture_buffers":
        return f"{spec.n_capture_points} points x {spec.batch_size}x{spec.seq_len} x {facts.d_model}"
    if name == "mlp_intermediate":
        return f"intermediate {facts.intermediate_size:,}"
    if name == "hidden_states" and spec.requires_grad:
        return f"retained across all {facts.n_layers} layers (requires_grad=True)"
    return ""


def _eager_advice(activation: dict[str, int], spec: WorkloadSpec, facts: ModelMemoryFacts) -> list[str]:
    """What to change, cheapest first, naming the term each fixes."""
    out: list[str] = []
    if activation.get("attention", 0) > GIB:
        out.append(
            f"attn_implementation='sdpa' removes the {activation['attention'] / GIB:.1f} GiB attention "
            f"matrix entirely; it is quadratic in the prompt and load_model defaults to 'eager' here"
        )
    if activation.get("logits", 0) > GIB:
        halved = spec.seq_len // 2
        out.append(
            f"the logits are {activation['logits'] / GIB:.1f} GiB at {spec.seq_len} tokens over a "
            f"{facts.vocab_size:,} vocab; {halved} tokens halves it"
        )
    if spec.requires_grad:
        out.append("requires_grad=False if you are not fitting a lens: gradients retain every layer's activations")
    if spec.dtype in ("float32", "fp32"):
        out.append("dtype='bfloat16' halves the weights and every activation term (EagerModel defaults to float32)")
    out.append("backend='vllm' pages the KV cache and never materializes all-position logits")
    return out


def _vllm_advice(
    *,
    spec: WorkloadSpec,
    facts: ModelMemoryFacts,
    gpu: GpuSpec,
    pool_headroom: int,
    outside_headroom: int,
    weights: int,
    buffers: int,
) -> list[str]:
    out: list[str] = []
    if outside_headroom < 0:
        out.append(
            f"lower gpu_memory_utilization: {-outside_headroom / GIB:.1f} GiB more is needed OUTSIDE the "
            f"pool, which is where the CUDA context, the warmup overshoot and your own reservations live"
        )
    if pool_headroom < 0:
        shortfall = -pool_headroom / GIB
        if buffers and buffers > -pool_headroom:
            out.append(
                f"static buffers are {buffers / GIB:.1f} GiB: a smaller max_num_batched_tokens, or "
                f"static_writes=[] to drop the write half, recovers most of {shortfall:.1f} GiB"
            )
        if spec.max_model_len > 2048:
            # Per card, like every other figure in this list: the reader is looking at one card's
            # shortfall, so an unsharded saving would not be the saving they are being offered.
            freed = (
                spec.max_model_len
                // 2
                * kv_bytes_per_token(facts, kv_dtype=spec.kv_cache_dtype, model_dtype=spec.dtype)
                / kv_shards(facts, spec.num_gpus)
            )
            out.append(
                f"max_model_len={spec.max_model_len} sets the KV floor; halving it frees about {freed / GIB:.1f} GiB"
            )
        if spec.dtype in ("float32", "fp32", "auto") and facts.weights.stored_dtype in ("float32", "fp32"):
            out.append("dtype='bfloat16' halves the weights, which are the largest term in the pool")
        if weights > gpu.total_bytes * 0.7:
            need = weights / (gpu.total_bytes * 0.6)
            out.append(
                f"the weights alone are {weights / GIB:.1f} GiB of a {gpu.total_gib:.1f} GiB card; "
                f"num_gpus={max(2, int(need) + 1)} shards them, or use a quantized checkpoint"
            )
        if spec.graphs_on:
            out.append(
                f"backend='vllm' instead of '{spec.backend}' frees the {_cal('graph_pool_gib'):.0f} GiB graph "
                f"pool and every static buffer, at 4-11x lower decode throughput"
            )
    return out


# ------------------------------------------------------------------- model resolution


def model_memory_facts(
    hf_model_id: str,
    *,
    config: Any = None,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> ModelMemoryFacts:
    """Resolve everything sizing needs for a model, without downloading weights.

    Reads ``config.json`` (a few KB) and the weight *metadata* (a few KB of safetensors headers plus
    the shard index), never a shard. A gated repo needs a token for the config but not for the weight
    metadata, so a caller with no token still gets exact weight bytes and only loses the KV-cache
    precision that the dims provide.

    ``token`` falls back to the usual environment variables, so an exported ``HF_TOKEN`` is picked up.
    """
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if config is None:
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(hf_model_id, token=token, trust_remote_code=trust_remote_code)
        except Exception:
            config = None

    weights = weight_bytes(config=config, hf_model_id=hf_model_id, token=token)

    if config is None:
        # No dims. Everything that depends on them degrades to the pre-GQA worst case, which is
        # pessimistic by up to 8x on the KV term rather than optimistic -- the right direction, but
        # worth saying out loud, which the caller does via `weights.source`.
        return ModelMemoryFacts(
            model_id=hf_model_id,
            weights=weights,
            n_layers=0,
            d_model=0,
            n_heads=0,
            n_kv_heads=0,
            head_dim=0,
            v_head_dim=0,
            vocab_size=0,
            intermediate_size=0,
        )

    from interp_engine import facts as facts_mod

    f = facts_mod.resolve_facts(config)
    cfg = facts_mod.text_config(config)
    intermediate = int(getattr(cfg, "intermediate_size", 0) or 0) or 4 * max(f.d_model, 1)
    return ModelMemoryFacts(
        model_id=hf_model_id,
        weights=weights,
        n_layers=f.n_layers,
        d_model=f.d_model,
        n_heads=f.n_heads,
        n_kv_heads=f.n_kv_heads,
        head_dim=f.head_dim,
        v_head_dim=f.v_head_dim,
        vocab_size=f.vocab_size,
        intermediate_size=intermediate,
        n_experts=f.n_experts,
        layer_types=f.layer_types,
        sliding_window=f.sliding_window,
        n_residual_streams=f.n_residual_streams,
        max_position_embeddings=int(getattr(cfg, "max_position_embeddings", 0) or 0),
        architecture=f.architecture,
        kv_quant_algo=hub_kv_quant_algo(hf_model_id, token),
    )


# ----------------------------------------------------------------------------- fit


#: Capture sizes ``fit`` steps ``max_num_batched_tokens`` down through. 1024 is the floor: below it
#: vLLM will not usefully serve, so refuse rather than recommend something that cannot work.
#: Same ladder as ``vllm_capture/static._CAPTURE_SIZES``, asserted equal in the tests.
CAPTURE_SIZES = (16384, 8192, 4096, 2048, 1024)

#: Contexts ``fit`` will step ``max_model_len`` down through when the caller did not pin one.
#:
#: Searching this is not a nicety. Lowering ``max_model_len`` is the single most effective fix for a
#: model that will not fit -- it is the only term a caller can change that shrinks the KV floor
#: linearly without touching the weights -- and a sizer that reports "nothing fits" while a 4k context
#: would have worked on a card the user already owns is worse than useless. gemma-3-12b advertises
#: 131k, which at 48 layers costs 48 GiB of KV floor: more than an A40 has, for a context almost nobody
#: runs. The same model at 8k fits that card with room to spare.
CONTEXT_LADDER = (131072, 65536, 32768, 16384, 8192, 4096, 2048)


def fit(
    facts: ModelMemoryFacts,
    gpu: GpuSpec,
    *,
    backend: str = "vllm",
    reservations: Reservations | None = None,
    max_model_len: int = 0,
    dtype: str = "auto",
    kv_cache_dtype: str = "auto",
    num_gpus: int = 1,
    static_sites: int = 0,
    static_points: Sequence[str] = (),
    max_num_batched_tokens: int = 0,
    min_kv_sequences: int = 2,
    batch_size: int = 1,
    seq_len: int = 0,
    n_capture_points: int = 0,
    requires_grad: bool = False,
    attn_implementation: str = "",
) -> tuple[WorkloadSpec, MemoryEstimate] | None:
    """The largest configuration of this shape that fits, or None when none does.

    Solves in the order the constraints actually bind:

    1. **Utilization.** Everything outside the pool is fixed by the card and the reservations, so the
       ceiling on utilization is arithmetic, not search. It is then *truncated* to two decimals rather
       than rounded, because rounding up is the direction that OOMs, and capped at vLLM's own 0.90.
    2. **Batch width.** With the utilization known, step ``max_num_batched_tokens`` down the ladder
       until the static buffers fit beside the weights, the graphs and the KV floor.
    3. **A usable cache.** A configuration that fits but holds fewer than ``min_kv_sequences``
       full-length sequences is refused: it would serve one request at a time and stall the rest, which
       reads as a hang rather than as the capacity problem it is.

    ``min_kv_sequences=2`` because one is not enough to overlap a prefill with a decode; raise it for a
    pod expected to serve concurrent traffic.
    """
    res = reservations or Reservations()

    if not facts.trunk_dims_known:
        # Short-circuit what would happen anyway. `estimate` refuses to report `fits` on a model whose
        # dims are unknown, so every rung of every ladder below would come back False; returning here
        # just saves walking them. Callers wanting to tell "cannot size" from "does not fit" should
        # read `facts.trunk_dims_known`, which is the only thing that distinguishes them.
        return None

    if backend == "eager":
        # Prompt length is to eager what context is to vLLM: the term that decides whether it fits, and
        # the one to step down. Sizing only for the model's advertised context and giving up otherwise
        # is how this came to report that gemma-3-12b runs eagerly on *no* GPU in the catalogue -- its
        # 131k advertised context puts 206 GiB of logits on the card, while the same model at a 4k
        # prompt needs about 25 GiB and fits one A40 with room to spare. The answer was not merely
        # pessimistic, it was the kind that sends someone to buy a second card they do not need.
        #
        # An explicitly requested length is still honoured exactly, on the same reasoning as the vLLM
        # branch below: quietly fitting a shorter prompt than was asked for answers a different
        # question, and the caller finds out at run time.
        pinned = seq_len or max_model_len
        advertised = facts.max_position_embeddings or 4096
        prompts = [pinned] if pinned else [advertised, *[n for n in CONTEXT_LADDER if n < advertised]]

        for prompt in prompts:
            # `sdpa` rather than the engine's `eager` default: the quadratic attention matrix is the
            # largest avoidable term on this backend, and a sizer recommending a configuration should
            # recommend the one that works. `estimate` still prices the default when asked for it.
            spec = WorkloadSpec(
                backend="eager",
                dtype=dtype,
                num_gpus=num_gpus,
                batch_size=batch_size,
                seq_len=prompt,
                n_capture_points=n_capture_points,
                requires_grad=requires_grad,
                attn_implementation=attn_implementation or "sdpa",
                max_model_len=max_model_len,
            )
            est = estimate(facts, gpu, spec, res)
            if est.fits:
                return est.spec, est
        return None

    # Only what genuinely lands outside the pool sets the ceiling on utilization. The CUDA context does
    # not: vLLM charges it against its own budget, so counting it here would lower utilization to buy
    # margin that is not needed and shrink the KV cache for nothing.
    outside = int(_cal("vllm_overshoot_gib") * GIB) + int(_cal("frag_fraction") * gpu.total_bytes)
    if not res.before_engine:
        outside += res.for_rank(0)

    ceiling = (gpu.total_bytes - outside) / gpu.total_bytes
    # Truncate, never round: one step of 0.01 is ~0.44 GiB on a 44 GiB card, and the estimate feeding
    # this has more error than that in the optimistic direction.
    util = min(_cal("max_util"), int(ceiling * 100) / 100)
    if util < _cal("min_util"):
        return None

    advertised = facts.max_position_embeddings or 4096
    if max_model_len:
        # Pinned by the caller: a sizer that quietly served less context than was asked for would be
        # answering a different question, and the caller would find out at request time.
        contexts = [max_model_len]
    else:
        contexts = [advertised, *[size for size in CONTEXT_LADDER if size < advertised]]

    asked = max_num_batched_tokens or (8192 if backend == "vllm-static" else 2048)
    batch_widths = [asked, *[size for size in CAPTURE_SIZES if size < asked]]

    # Context-major: the largest context that fits is worth more than the widest prefill batch, because
    # too small a context refuses requests outright while a narrow batch only slows prefill down.
    for context in contexts:
        for batched in batch_widths:
            spec = WorkloadSpec(
                backend=backend,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                max_model_len=context,
                # A prefill batch wider than the context is waste: nothing can fill it.
                max_num_batched_tokens=min(batched, context),
                gpu_memory_utilization=util,
                num_gpus=num_gpus,
                static_sites=static_sites,
                static_points=tuple(static_points),
            )
            est = estimate(facts, gpu, spec, res)
            if est.fits and est.concurrent_sequences >= min_kv_sequences:
                return est.spec, est
    return None


def fit_across(
    facts: ModelMemoryFacts,
    gpus: list[GpuSpec],
    *,
    backend: str = "vllm",
    max_gpus: int = 8,
    **kwargs: Any,
) -> list[tuple[GpuSpec, int, WorkloadSpec, MemoryEstimate]]:
    """Every ``(gpu, count)`` that fits, cheapest card and fewest cards first.

    Multi-GPU is tried in powers of two only, because that is what tensor parallelism requires: the
    attention heads have to divide evenly across ranks, and vLLM refuses a count that does not divide
    the KV heads.
    """
    out: list[tuple[GpuSpec, int, WorkloadSpec, MemoryEstimate]] = []
    for gpu in sorted(gpus, key=lambda g: g.total_bytes):
        count = 1
        while count <= max_gpus:
            result = fit(facts, gpu, backend=backend, num_gpus=count, **kwargs)
            if result is not None:
                spec, est = result
                out.append((gpu, count, spec, est))
                break
            count *= 2
    return out
