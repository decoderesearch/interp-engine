"""Tests for the `gpu-sizer/` scripts' own logic, as opposed to the arithmetic they call.

`interp_engine.memory` is tested in `test_memory.py`. What is left here is the reasoning the CLI and
the harness do *around* it, and it is worth pinning because both parts of it have already been wrong
in a way that produced confident, plausible, false output:

- `evidence_for` once keyed on the backend without the settings, so a `vllm-static` crash recorded at
  16,384 batched tokens stamped **KNOWN TO FAIL** on a recommendation using 8,192 -- and the sizer
  then printed that label and "FITS, headroom +2.07 GiB" one after the other.
- `distill_error` feeds a markdown table, and a progress bar reaching a cell splits the row into
  extra columns. The failure it has to survive is the one where a killed process left nothing in its
  output *but* progress bars.

The directory has a hyphen in it, so it is not importable as a package; `_load` reads each script by
path instead. Nothing here touches a GPU, a network, or a checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from interp_engine import memory as mem

SIZER = Path(__file__).resolve().parent.parent / "gpu-sizer"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_sizer_{name}", SIZER / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fit = _load("fit")
verify = _load("verify")


def record(
    *,
    outcome: str = "pass",
    model_id: str = "test/model",
    gpu: str = "NVIDIA A40",
    backend: str = "vllm",
    dtype: str = "bfloat16",
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    seq_len: int = 0,
) -> dict:
    return {
        "model_id": model_id,
        "backend": backend,
        "outcome": outcome,
        "gpu": {"name": gpu},
        "spec": {
            "dtype": dtype,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "seq_len": seq_len,
        },
    }


def candidate(**kwargs) -> mem.WorkloadSpec:
    base = {"backend": "vllm", "dtype": "bfloat16", "max_model_len": 8192, "max_num_batched_tokens": 8192}
    return mem.WorkloadSpec(**{**base, **kwargs})


def evidence(records: list[dict], spec: mem.WorkloadSpec, backend: str = "vllm") -> str:
    return fit.evidence_for(records, "test/model", "NVIDIA A40", backend, spec)


# ------------------------------------------------------------------- what counts as evidence


def test_an_exact_match_is_reported_verbatim():
    assert evidence([record()], candidate()) == "verified"
    assert "FAIL" in evidence([record(outcome="crash")], candidate())


def test_a_pass_vouches_for_a_narrower_configuration_but_not_a_wider_one():
    """Every knob compared only ever costs memory, so a pass bounds everything below it and nothing above."""
    passed = [record(max_model_len=8192)]
    assert evidence(passed, candidate(max_model_len=4096)) == "verified at 8,192 ctx"
    assert evidence(passed, candidate(max_model_len=32768)) == "estimated"


def test_a_failure_condemns_a_wider_configuration_but_not_a_narrower_one():
    failed = [record(outcome="oom", max_model_len=8192)]
    assert evidence(failed, candidate(max_model_len=16384)) == "fails at 8,192 ctx"
    assert evidence(failed, candidate(max_model_len=4096)) == "estimated"


def test_a_failure_at_a_wider_batch_does_not_condemn_a_narrower_one():
    """The regression that made the sizer contradict itself on gemma-3-12b in consecutive lines."""
    crashed = [record(outcome="crash", backend="vllm-static", max_num_batched_tokens=16384)]
    spec = candidate(backend="vllm-static", max_num_batched_tokens=8192)
    assert evidence(crashed, spec, backend="vllm-static") == "estimated"


def test_evidence_never_crosses_card_backend_or_dtype():
    """A result is evidence for the thing that was run, and a card is part of the thing that was run."""
    runs = [record()]
    assert fit.evidence_for(runs, "test/model", "NVIDIA H100 80GB HBM3", "vllm", candidate()) == "estimated"
    assert fit.evidence_for(runs, "other/model", "NVIDIA A40", "vllm", candidate()) == "estimated"
    assert evidence(runs, candidate(), backend="vllm-static") == "estimated"
    assert evidence(runs, candidate(dtype="float16")) == "estimated"


def test_no_records_at_all_reads_as_estimated():
    assert evidence([], candidate()) == "estimated"


# ------------------------------------------------------------------- reading a failure


def test_distill_error_prefers_the_line_naming_the_exception():
    text = "Loading weights:  55%|#####     | 585/1065 [00:35<00:30, 15.66it/s]\ntorch.OutOfMemoryError: CUDA oom"
    assert verify.distill_error(text) == "torch.OutOfMemoryError: CUDA oom"


def test_distill_error_says_so_when_a_load_was_killed_without_raising():
    """All bars and no exception is not missing information: it is what an OS-level kill looks like."""
    text = "Loading weights:  77%|#######   | 820/1065 [00:41<00:12, 19.2it/s]\r"
    out = verify.distill_error(text)
    assert "killed during load" in out
    assert "77%" in out


def test_distill_error_never_returns_a_pipe():
    """A stray pipe in a markdown cell silently splits the row, which is worse than a useless cell."""
    for text in (
        "Loading weights: 77%|#######   | 820/1065 [00:41<00:12, 19.2it/s]",
        "ValueError: got |this| instead",
        "",
    ):
        assert "|" not in verify.distill_error(text)


@pytest.mark.parametrize("length", [200, 5000])
def test_distill_error_is_bounded(length: int):
    assert len(verify.distill_error("RuntimeError: " + "x" * length)) <= 130


# ------------------------------------------------------------------- reading a result


def test_margin_note_reads_a_spill_differently_on_a_run_that_lived_and_one_that_died():
    """The same ratio is a warning on a survivor and a post-mortem on a casualty."""
    lived = verify.Record(
        model_id="m", backend="vllm", outcome="pass", spec={}, gpu={}, estimate={}, measured={}, stress={}
    )
    died = verify.Record(
        model_id="m", backend="vllm", outcome="crash", spec={}, gpu={}, estimate={}, measured={}, stress={}
    )
    assert verify.margin_note(0.4, lived) == "margin held"
    assert "thinner" in verify.margin_note(1.6, lived)
    assert "died" in verify.margin_note(1.6, died)


def test_estimate_note_does_not_credit_the_estimate_for_a_process_that_died_early():
    """A run killed mid-load never reaches its peak, so a low ratio there says nothing about the estimate."""
    died = verify.Record(
        model_id="m", backend="eager", outcome="oom", spec={}, gpu={}, estimate={}, measured={}, stress={}
    )
    lived = verify.Record(
        model_id="m", backend="eager", outcome="pass", spec={}, gpu={}, estimate={}, measured={}, stress={}
    )
    assert "died before reaching" in verify.estimate_note(0.17, died)
    assert verify.estimate_note(0.84, lived) == "estimate is conservative"
    assert "OPTIMISTIC" in verify.estimate_note(1.2, lived)


# ------------------------------------------------------------------- what the report flags


def test_block_rounding_alone_does_not_flag_a_row():
    """The two B200 rows that prompted the threshold, at their measured numbers.

    Both sat a fraction under 1.00 because vLLM allocates the cache in whole blocks and the last
    partial one is rounded away -- far less than one block's worth in each case. A marker that fires
    on rows like these is a marker every reader learns to skip, which costs the one that means it.
    """
    assert verify.kv_ratio_cell(1_109_600, 1_111_293) == "1.00"  # 0.15% under
    assert verify.kv_ratio_cell(1_239_264, 1_242_299) == "1.00"  # 0.24% under


def test_an_estimate_that_overpromises_by_more_than_half_a_percent_still_flags():
    """The direction worth being strict about: cache promised that the engine could not build."""
    assert verify.kv_ratio_cell(994, 1_000) == "0.99 !"
    assert verify.kv_ratio_cell(500, 1_000) == "0.50 !"


def test_a_conservative_estimate_is_never_flagged_however_far_out_it_is():
    """Over-prediction is the safe direction, and the NVFP4 KV bug was found by reading the ratio."""
    assert verify.kv_ratio_cell(1_000, 1_000) == "1.00"
    assert verify.kv_ratio_cell(784_896, 394_295) == "1.99"


def test_a_row_missing_either_number_reports_neither_a_ratio_nor_a_verdict():
    """A run that died before vLLM logged its cache, where a ratio would be a division by nothing."""
    assert verify.kv_ratio_cell(0, 1_000) == "-"
    assert verify.kv_ratio_cell(1_000, 0) == "-"


def test_matched_expectation_treats_every_kind_of_death_as_a_failure():
    for outcome in ("oom", "refused", "crash"):
        r = verify.Record(
            model_id="m",
            backend="vllm",
            outcome=outcome,
            spec={},
            gpu={},
            estimate={},
            measured={},
            stress={},
            expected="fail",
        )
        assert r.matched_expectation
        assert not verify.Record(**{**vars(r), "expected": "pass"}).matched_expectation
