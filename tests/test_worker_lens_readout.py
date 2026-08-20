"""The word-mask contract of :func:`worker_lens_readout`, on a stub worker.

The read-out ranks tokens with non-word ones masked out, but each position's *final* row
carries the model's real next-token prediction, so that one row keeps its true top-1 even
when the winner is punctuation. Two invariants come out of that and neither is visible
from the shape of the result:

- masking changes the *ranking* only. Probabilities stay normalised over the whole vocab,
  because ``log_z`` is taken before the mask is applied -- so a masked read-out of a token
  reports the same probability an unmasked one would.
- the preserved row is the last of each ``rows_per_group``, not the first and not every row.

Worth a stub rather than a GPU model because the arithmetic is the contract; the surface
``worker_lens_readout`` needs from a vLLM worker is small enough to stand up on CPU.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from interp_engine.vllm_capture import decode_tensor_payload, encode_tensor_payload, worker_lens_readout

VOCAB = 8
WORD_TOKENS = (0, 1, 2, 3)  # 4..7 stand in for punctuation / non-word pieces
ROWS_PER_GROUP = 3


class StubModel(torch.nn.Module):
    """The whole surface ``worker_lens_readout`` touches: a final norm and ``compute_logits``.

    No ``config`` and no ``logits_processor``, which is the "vLLM applied no scale and no
    softcap" case -- the Llama/Qwen shape, and the one the lens actually runs on.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.norm = torch.nn.Identity()
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))
        self._logits = logits

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        assert hidden.shape[0] == self._logits.shape[0]
        return self._logits


def read_out(logits: torch.Tensor, *, word_mask: torch.Tensor | None, top_n: int = 3):
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=StubModel(logits)))
    resid = torch.zeros(logits.shape[0], 1)
    out = worker_lens_readout(
        worker,
        encode_tensor_payload(resid),
        top_n,
        None,
        None if word_mask is None else encode_tensor_payload(word_mask),
        ROWS_PER_GROUP,
    )
    return decode_tensor_payload(out["top_idx"]), decode_tensor_payload(out["top_probs"])


@pytest.fixture
def logits() -> torch.Tensor:
    """Six rows / two groups, with a non-word winner on rows 0, 2, 3 and 5.

    Rows 2 and 5 are the group finals, so those two keep their non-word winner while rows
    0 and 3 lose theirs to the mask.
    """
    values = torch.full((6, VOCAB), -10.0)
    values[0, 5], values[0, 2] = 10.0, 5.0  # non-word wins, word runner-up
    values[1, 1], values[1, 0] = 9.0, 4.0  # word already wins
    values[2, 6], values[2, 3] = 12.0, 6.0  # FINAL of group 0: non-word wins
    values[3, 7], values[3, 0] = 8.0, 3.0
    values[4, 3], values[4, 2] = 7.0, 2.0
    values[5, 4], values[5, 1] = 11.0, 1.0  # FINAL of group 1: non-word wins
    return values


@pytest.fixture
def word_mask() -> torch.Tensor:
    mask = torch.zeros(VOCAB, dtype=torch.bool)
    mask[list(WORD_TOKENS)] = True
    return mask


def test_without_a_mask_every_row_reports_its_true_top_1(logits):
    top_idx, _probs = read_out(logits, word_mask=None)
    assert top_idx[:, 0].tolist() == [5, 1, 6, 7, 3, 4]


def test_the_mask_demotes_non_word_winners_except_on_each_group_final_row(logits, word_mask):
    top_idx, _probs = read_out(logits, word_mask=word_mask)
    #                            ^ rows 2 and 5 are the finals and keep 6 and 4
    assert top_idx[:, 0].tolist() == [2, 1, 6, 0, 3, 4]


def test_masking_leaves_the_runner_up_ranking_alone(logits, word_mask):
    """Below the top-1 the surviving order is still the unmasked order, word tokens only."""
    top_idx, _probs = read_out(logits, word_mask=word_mask)
    assert set(top_idx[0].tolist()) <= set(WORD_TOKENS)
    assert top_idx[0, 0].item() == 2


def test_probabilities_stay_normalised_over_the_full_vocab(logits, word_mask):
    """The mask is a ranking device; it must not renormalise the probabilities.

    Row 0's reported top token is 2 only because 5 was masked, but 5 still holds most of
    the mass -- so p(2) must be softmax over the *unmasked* row, well under 1.
    """
    _idx, probs = read_out(logits, word_mask=word_mask)
    expected = torch.softmax(logits[0], dim=-1)[2]
    assert probs[0, 0].item() == pytest.approx(expected.item(), rel=1e-5)
    assert probs[0, 0].item() < 0.01

    masked_final, unmasked_final = (
        read_out(logits, word_mask=word_mask)[1][2, 0].item(),
        read_out(logits, word_mask=None)[1][2, 0].item(),
    )
    assert masked_final == pytest.approx(unmasked_final, rel=1e-6)


def test_the_models_own_logits_are_not_written_through(logits, word_mask):
    """The mask is applied in place, so it must land on a tensor the read-out owns.

    ``logits.float()`` is a no-op returning ``logits`` itself whenever the model already
    computes in float32, and masking through that would hand the caller a tensor with half
    its vocab set to -inf.
    """
    before = logits.clone()
    read_out(logits, word_mask=word_mask)
    assert torch.equal(logits, before)


def test_a_mask_shorter_than_the_vocab_is_padded_as_non_word(logits):
    """Tokenizers under-count padded embedding tables; the extra slots are never word-like."""
    short = torch.ones(4, dtype=torch.bool)  # only tokens 0..3 described, all word-like
    top_idx, _probs = read_out(logits, word_mask=short)
    assert top_idx[:, 0].tolist() == [2, 1, 6, 0, 3, 4]


def test_a_trailing_partial_group_is_left_unpreserved(logits, word_mask):
    """Only complete groups have a final row; a ragged tail must not promote a stray row."""
    top_idx, _probs = read_out(logits[:5], word_mask=word_mask)
    assert top_idx[:, 0].tolist() == [2, 1, 6, 0, 3]


def test_every_row_is_its_own_group_when_rows_per_group_is_one(logits, word_mask):
    """Then every row is a final row, so the mask should change nothing."""
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=StubModel(logits)))
    out = worker_lens_readout(
        worker,
        encode_tensor_payload(torch.zeros(6, 1)),
        3,
        None,
        encode_tensor_payload(word_mask),
        1,
    )
    assert decode_tensor_payload(out["top_idx"])[:, 0].tolist() == [5, 1, 6, 7, 3, 4]


def test_softcap_is_applied_when_vllm_has_not_applied_one(logits):
    """Gemma-style cap: the caller passes it and the worker applies ``cap * tanh(x / cap)``."""
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=StubModel(logits)))
    out = worker_lens_readout(worker, encode_tensor_payload(torch.zeros(6, 1)), 1, 5.0, None, ROWS_PER_GROUP)
    probs = decode_tensor_payload(out["top_probs"])
    capped = 5.0 * torch.tanh(logits[0] / 5.0)
    assert probs[0, 0].item() == pytest.approx(torch.softmax(capped, dim=-1).max().item(), rel=1e-5)
    assert not math.isnan(probs[0, 0].item())


class _GraphWrapper:
    """vLLM's cudagraph wrappers, in the two respects this depends on.

    ``CUDAGraphWrapper`` and 0.26's ``BreakableCUDAGraphWrapper`` both replace
    ``model_runner.model`` with a plain object -- not an ``nn.Module`` -- that forwards attribute
    access to the module and returns it from ``unwrap()``.
    """

    def __init__(self, runnable: object) -> None:
        self.runnable = runnable

    def __getattr__(self, key: str):
        return getattr(self.__dict__["runnable"], key)

    def unwrap(self) -> object:
        return self.runnable


def test_the_read_out_works_on_a_graph_replaying_engine(logits):
    """A lens read-out needs the *weights*, not a forward hook, so a graph-replaying engine is a
    perfectly good place to run one -- and the two architectures vLLM auto-enables its breakable
    cudagraph path for (DeepSeek-V4, Qwen3.8) were the ones where this failed, with a type error
    naming a vLLM class rather than an answer. `_tree._worker_model` unwraps instead.
    """
    wrapped = SimpleNamespace(model_runner=SimpleNamespace(model=_GraphWrapper(StubModel(logits))))
    out = worker_lens_readout(wrapped, encode_tensor_payload(torch.zeros(6, 1)), 3, None, None, ROWS_PER_GROUP)

    assert decode_tensor_payload(out["top_idx"])[:, 0].tolist() == [5, 1, 6, 7, 3, 4]
