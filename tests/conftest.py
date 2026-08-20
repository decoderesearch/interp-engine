"""Shared fixtures for engine tests.

The fast suite runs on CPU against cached ``openai-community/gpt2`` (the runnable parity gate)
plus the two small instruct models in ``tests/harness.py``. Tests that need the reference
backend (TransformerLens) import-skip when it isn't available -- except under
``IE_REQUIRE_PARITY=1`` (set by CI), where a missing reference is a failure, so the golden gate
can't silently disappear into a skip.
"""

import os
import warnings

import pytest
from _pytest.terminal import TerminalReporter
from harness import GPT2, hf_token_present, load_model, parity_required

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_HF_TOKEN_ABSENT_MSG = (
    "HF_TOKEN (and HUGGING_FACE_HUB_TOKEN) is not set: gated-model tests "
    "(@pytest.mark.gated, e.g. google/gemma-3-270m-it) will be SKIPPED. "
    "Set HF_TOKEN to exercise gated models."
)


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    # Warn once at the very start of the session if the token is missing.
    if not hf_token_present():
        warnings.warn(_HF_TOKEN_ABSENT_MSG, stacklevel=1)


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    # Deterministically skip anything marked ``gated`` when no HF token is available.
    if hf_token_present():
        return
    skip_gated = pytest.mark.skip(reason="HF_TOKEN not set; skipping gated-model test")
    for item in items:
        if "gated" in item.keywords:
            item.add_marker(skip_gated)


def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: int,  # noqa: ARG001
    config: pytest.Config,  # noqa: ARG001
) -> None:
    # Warn again in the end-of-run summary so the skip isn't silently missed.
    if not hf_token_present():
        terminalreporter.write_line("")
        terminalreporter.write_line(f"WARNING: {_HF_TOKEN_ABSENT_MSG}", yellow=True, bold=True)


@pytest.fixture(scope="session")
def gpt2():
    # ``required`` under IE_REQUIRE_PARITY: a cold HF cache would otherwise turn the whole
    # golden gate into skips (and, without the guard, into a raw fixture error).
    return load_model(GPT2, device="cpu", attn_implementation="eager", required=parity_required())


@pytest.fixture(scope="session")
def tlens_gpt2():
    """Reference TransformerLens model loaded with the same `no_processing` semantics."""
    if parity_required():
        import transformer_lens as tl
    else:
        tl = pytest.importorskip("transformer_lens")
    return tl.HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu", dtype="float32")


@pytest.fixture(scope="session")
def prompt():
    return "The quick brown fox jumps over the lazy dog"
