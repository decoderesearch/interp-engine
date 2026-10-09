"""A left-padded batch through ``capture`` on a real vLLM engine.

The CPU half (``tests/test_padded_positions.py``) runs the non-eager arm on an eager model seen
through the protocol. This half runs it on vLLM, where each row is its own concurrent request, and
checks the rows against eager at every unmasked position. The model is driven through the sync
facade only, so it is bound to that loop and no other.
"""

from __future__ import annotations

import pytest
import torch
from harness import require_vllm

from interp_engine import Address, capture, load_model, sync_model

require_vllm()  # skips this module without vLLM; fails under IE_REQUIRE_VLLM (set by the GPU CI job)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the vLLM backend initializes on CUDA"),
]

MODEL = "openai-community/gpt2"
PROMPTS = ["The capital of France is Paris, and the capital of Germany is", "Hello there"]
POINTS = [Address("resid_post", 2), Address("mlp_out", 7), Address("z", 5)]


@pytest.fixture(scope="module")
def padded() -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [tokenizer(p)["input_ids"] for p in PROMPTS]
    seq = max(len(r) for r in rows)
    ids = torch.full((len(rows), seq), tokenizer.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), seq), dtype=torch.long)
    for b, row in enumerate(rows):
        ids[b, seq - len(row) :] = torch.tensor(row)
        mask[b, seq - len(row) :] = 1
    return ids, mask


@pytest.fixture(scope="module")
def eager_cache(padded: tuple[torch.Tensor, torch.Tensor]) -> dict[Address, torch.Tensor]:
    ids, mask = padded
    model = load_model(MODEL, backend="eager", dtype="float32", device="cuda")
    cache = capture(model, ids, POINTS, attention_mask=mask)
    captured = {a: cache[a].float().cpu() for a in POINTS}
    del model, cache
    torch.cuda.empty_cache()
    return captured


@pytest.fixture(scope="module")
def vllm_model():
    model = load_model(MODEL, backend="vllm", dtype="float32", max_model_len=512, gpu_memory_utilization=0.2)
    sync = sync_model(model)
    sync.warmup()
    yield model
    sync.shutdown()


def test_each_padded_row_agrees_with_eager_at_every_unmasked_position(
    vllm_model, padded: tuple[torch.Tensor, torch.Tensor], eager_cache: dict[Address, torch.Tensor]
) -> None:
    """Relative, because vLLM's fused kernels are not bit-identical to eager PyTorch."""
    ids, mask = padded
    cache = capture(vllm_model, ids, POINTS, attention_mask=mask)
    keep = mask.bool()
    for address in POINTS:
        mine, theirs = cache[address].float().cpu(), eager_cache[address]
        assert mine.shape == theirs.shape, address
        scale = max(theirs[keep].abs().max().item(), 1e-6)
        assert (mine[keep] - theirs[keep]).abs().max().item() / scale < 1e-2, f"{address} disagrees"
        assert not mine[~keep].any(), f"{address}: masked positions hold zeros on vLLM"
