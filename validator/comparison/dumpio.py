"""On-disk layout for the validator (stdlib + numpy only, so every engine env can use it).

Every path is keyed by the HF repo id, so the org becomes a directory level and the tree reads like
the hub does:

<dumps>/
  inputs/<org>/<model>.json         # shared, pre-tokenized input ids + layer plan
  <engine>/<org>/<model>.npz        # captured activations, keyed by spec.dump_key(point, layer)
  <engine>/<org>/<model>.meta.json  # status / captured points / sae features / skip reason

Dumps written before the id became the key sit at the old alias paths, where nothing looks for them:
they are not read as anything, and re-capturing is the only way to refill a cell. The verdicts
themselves are not lost with them, because those live in `comparison/results/`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

# numpy is imported lazily inside the array-I/O functions so this module (and its
# `classify_failure` helper) stays importable under a bare `python3` with no deps
# (e.g. `python -m comparison.show_errors`). The annotations below are strings
# (`from __future__ import annotations`), so the type-only import stays lazy too.
if TYPE_CHECKING:
    import numpy as np


@dataclass
class InputSpec:
    hf_id: str
    input_ids: list[int]
    n_layers: int
    layers: list[int]
    # Which of `layers` are linear-attention (Qwen3.5/3.6 hybrid trunks): those layers have no
    # softmax attention module, so `attn_out` there is not a quantity the fused engines can produce
    # and is excluded from the comparison. Recorded here at tokenize time because it comes from the
    # config, which the aggregator would otherwise have to re-download to know. `None` means an
    # older inputs file that predates the field, which is not the same claim as "no such layers".
    linear_attn_layers: list[int] | None = None


@dataclass
class CaptureMeta:
    engine: str
    hf_id: str
    # "ok" | "skip" (engine can't run this checkpoint) | "error" (raised) | "crash" (killed before it
    # could record anything — see the crash-meta path in run_all_models.sh)
    status: str
    reason: str = ""
    dtype: str = ""  # dtype the engine was asked to load (native; e.g. "float32"/"bfloat16")
    device: str = ""  # "cuda" | "cpu"
    points: list[str] = field(default_factory=list)  # dump keys actually captured
    # Requested dump keys the engine did NOT return. A partial capture is a real result (a fused
    # engine has no `attn_out` on a linear-attention layer) but it must not read as a full pass, so
    # it is recorded rather than inferred from the absence of a key.
    missing_points: list[str] = field(default_factory=list)
    # Optional SAE spot-check: {sae_id: {"feature_l0": float, "top_index": int, "acts": [...]}}
    sae: dict = field(default_factory=dict)
    # When this capture ran (UTC ``YYYY-MM-DD``). The table dates each cell from its own capture rather
    # than from the last aggregation, so re-rendering the README does not restamp 38 rows with today.
    # Empty on a dump that predates the field, where the meta file's mtime is the honest fallback.
    captured_at: str = ""
    # What produced this capture: {package: {"version", "commit"?, "dirty"?}} for the engine's own
    # stack, filled by `engine_versions.engine_versions()` in the engine's process (the aggregator runs
    # in a different venv and cannot see vLLM's or SGLang's packages). Empty on a crash record, which is
    # written by a bare python3 that has none of them, and on any meta predating the field.
    versions: dict = field(default_factory=dict)


# Failure signatures that mean "this comparison engine legitimately can't run this checkpoint" (a
# bleeding-edge / multimodal / capacity / env limit of the *comparison* engine — eager, the
# reference, handles these), as opposed to a real bug. Matched case-insensitively against the full
# exception text; matches are recorded as `skip` (clean "—" cells with a readable reason) instead of
# an alarming `error`. Reference (eager) regressions and ambiguous crashes stay `error`.
UNSUPPORTED_SIGNATURES = (
    # nnsight/nnterp: multimodal checkpoints register with AutoModelForImageTextToText, so its
    # text-only LanguageModel wrapper refuses to load them.
    "automodelforimagetexttotext",
    "multimodal model so languagemodel",
    "use visionlanguagemodel",
    # nnterp's rename contract wants a `self_attn` on layer 0, which a hybrid trunk hasn't got. The
    # adapter now relaxes that for trunks `facts` recognizes as hybrid (see nnsight_engine.capture), so
    # this signature is left for a trunk it cannot classify: still an upstream naming limit, not a bug.
    "could not find self_attn module",
    # ...and the other half of that contract, which checks the *shape* a layer takes and returns. A
    # hyper-connection trunk hands its blocks a stack of `hc_mult` residual streams, so DeepSeek-V4's
    # layer input is (1, 3, 4, 4096) where nnterp asserts (1, 3, 4096) and declines the checkpoint. An
    # architecture it does not model rather than a capture that went wrong.
    "could not check the io of",
    # TransformerLens loads the checkpoint's own tensors and converts them in bfloat16, so a natively
    # fp8-quantized checkpoint (DeepSeek-V4 ships e4m3 weights with block scales) meets torch's type
    # promotion rules instead of a dequantization path, and dies before any weight is converted.
    "promotion for float8 types is not supported",
    # TransformerLens: arch not in its model registry, or a transformers-version rename its loader
    # hasn't caught up to (e.g. GPTNeoX `embed_out`), or a TL3 multimodal adapter's optional deps.
    "not found. valid official model names",
    "object has no attribute 'embed_out'",
    # ...and the same rename reaching SGLang, whose Transformers fallback backend looks the head up by
    # the old name. One transformers rename, three engines.
    "named 'model.embed_out'",
    "needs the optional torchvision",
    "multimodalarchitectureadapter",
    # transformers too old in the engine's env to recognize the architecture.
    "does not recognize this architecture",
    "does not support",
    # capacity (a bigger checkpoint than this GPU can hold) — not a correctness issue.
    "out of memory",
    "cuda out of memory",
    # legacy TransformerLens converts weights after loading them, so its host-RAM peak is ~2x the
    # checkpoint; past the container's limit it is SIGKILLed. tlens_engine refuses first, with this
    # phrase, so the cell records a readable capacity skip instead of vanishing.
    "to load+convert into hookedtransformer",
    # SGLang docker/venv env mismatch (PyTorch/CuDNN) — environmental, not a model issue.
    "cudnn compatibility",
    # SGLang's olmo2.py reads `config.rope_parameters["rope_theta"]`, from before transformers nested
    # the rope settings — so every olmo-3 checkpoint dies in `Olmo2Attention.__init__`, taking the
    # scheduler's process group with it. An upstream limit that happens to arrive as a death.
    "keyerror: 'rope_theta'",
)


def classify_failure(reason: str) -> str:
    """Map a capture failure's exception text to ``skip`` (expected/unsupported) or ``error`` (a real
    bug or ambiguous crash worth surfacing). Single source of truth for run_engine + show_errors."""
    low = (reason or "").lower()
    return "skip" if any(sig in low for sig in UNSUPPORTED_SIGNATURES) else "error"


def inputs_path(dumps: str, hf_id: str) -> str:
    return os.path.join(dumps, "inputs", f"{hf_id}.json")


def write_inputs(dumps: str, spec: InputSpec) -> None:
    path = inputs_path(dumps, spec.hf_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(spec), f)


def read_inputs(dumps: str, hf_id: str) -> InputSpec:
    with open(inputs_path(dumps, hf_id)) as f:
        return InputSpec(**json.load(f))


def _dump_paths(dumps: str, engine: str, hf_id: str) -> tuple[str, str]:
    d = os.path.join(dumps, engine)
    return os.path.join(d, f"{hf_id}.npz"), os.path.join(d, f"{hf_id}.meta.json")


def write_meta(dumps: str, meta: CaptureMeta) -> None:
    """Write only the meta, leaving any existing npz alone.

    Separate from :func:`write_capture` so a crash record can be written by a bare `python3` with no
    numpy — the process that crashed is exactly the one that had the deps.
    """
    _, meta_path = _dump_paths(dumps, meta.engine, meta.hf_id)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(asdict(meta), f, indent=2)


def with_mask_sentinel(scores: np.ndarray) -> np.ndarray:
    """A score matrix with its ``-inf`` mask fills rewritten to the sentinel dumps use.

    Pre-softmax scores carry a "attention cannot see this" fill, and the engines do not spell it the
    same way: HF's eager attention adds the *checkpoint dtype's* minimum, while a recompute from
    captured q/k leaves ``-inf``, and so does a checkpoint that masks its own compressed blocks --
    DeepSeek-V4-Flash's `compressed_sparse_attention` layers, where the last three key columns of a
    13-token prompt are staircased out. Two things downstream want one spelling: `run_engine` refuses
    a capture holding non-finite values, and a reader of the .npz should not have to know which engine
    wrote it to tell a mask from a score.

    Only ``-inf`` is rewritten. ``NaN`` is left to trip that guard, which is the point of it -- a NaN
    dump scores as a mismatch in every *other* engine, so a capture that produces one must fail rather
    than be tidied into a plausible-looking matrix. `aggregate._mask_floor` reads either spelling.
    """
    import numpy as np

    if not np.isneginf(scores).any():
        return scores
    return np.where(np.isneginf(scores), float(np.finfo(np.float32).min), scores)


def write_capture(dumps: str, meta: CaptureMeta, arrays: dict[str, np.ndarray]) -> None:
    import numpy as np

    npz_path, _ = _dump_paths(dumps, meta.engine, meta.hf_id)
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    # Store everything as float32 for stable cross-engine comparison. Array names reach
    # `savez_compressed` as keyword arguments, which a checker cannot tell apart from its own
    # `allow_pickle` flag once they arrive by unpacking, hence the widened value type. A dump key is
    # a canonical address (`resid_post.5`), so it can never actually collide with that name.
    to_write: dict[str, Any] = {k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()}
    np.savez_compressed(npz_path, **to_write)
    meta.points = sorted(arrays.keys())
    write_meta(dumps, meta)


def read_meta(dumps: str, engine: str, hf_id: str) -> CaptureMeta | None:
    _, meta_path = _dump_paths(dumps, engine, hf_id)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return CaptureMeta(**json.load(f))


def read_arrays(dumps: str, engine: str, hf_id: str) -> dict[str, np.ndarray]:
    import numpy as np

    npz_path, _ = _dump_paths(dumps, engine, hf_id)
    if not os.path.exists(npz_path):
        return {}
    with np.load(npz_path) as z:
        return {k: z[k] for k in z.files}
