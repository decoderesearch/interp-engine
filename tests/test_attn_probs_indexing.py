"""`attn_probs` must come from the layer that was asked for, on a hybrid trunk too.

`output_attentions=True` returns one entry per layer that actually ran a softmax. On a plain
decoder that is every layer, so the tuple is indexable by layer number. On a hybrid
linear/full trunk it is not: Qwen3.5-0.8B marks 18 of its 24 layers `linear_attention`, so the
tuple has 6 entries -- for layers 3, 7, 11, 15, 19, 23 -- and indexing it by layer number
returns a DIFFERENT layer's attention for layers 0-5 and an IndexError beyond that.

The silent half is what makes this worth a test. Layer 0 came back with a plausible
`[1, 8, seq, seq]` whose rows summed to 1, so every shape and normalization check passed while
reading layer 3's attention. Only a comparison against the raw tuple catches it.
"""

from __future__ import annotations

import pytest
import torch
from harness import GPT2, QWEN_THINKING, load_model, parity_required

from interp_engine import run_with_cache

PROMPT = "cat dog cat dog cat"


def _ids(model):
    return model.tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)


def _raw_attentions(model):
    """The forward pass's own attentions tuple, which is the ground truth being mapped into."""
    with torch.no_grad():
        return model.hf_model(_ids(model), output_attentions=True, use_cache=False).attentions


def test_hybrid_trunk_emits_fewer_attentions_than_layers():
    """The premise. If a transformers release starts padding the tuple, the rest is moot."""
    model = load_model(QWEN_THINKING, device="cpu", attn_implementation="eager", required=parity_required())
    softmax_layers = model.arch.softmax_attention_layers()
    assert len(softmax_layers) < model.n_layers, "Qwen3.5-0.8B is supposed to be a hybrid trunk"
    assert len(_raw_attentions(model)) == len(softmax_layers)


def test_hybrid_layers_map_to_their_own_attention():
    """Each softmax layer must return its own rows, not a neighbour's.

    Every layer in **one** capture, which is the engine's own claim (any number of points, one
    forward) and not a shortcut: the per-layer assertion is identical either way, and a request per
    layer only pays for the same forward six times. That matters on this model in particular --
    18 of its 24 layers are linear-attention with no fast CPU kernel, so a forward costs ~15s
    whatever the prompt, and the loop version of this test was the second-slowest in the suite.
    """
    model = load_model(QWEN_THINKING, device="cpu", attn_implementation="eager", required=parity_required())
    raw = _raw_attentions(model)
    layers = model.arch.softmax_attention_layers()
    cache = run_with_cache(model, _ids(model), [("attn_probs", layer) for layer in layers])
    for position, layer in enumerate(layers):
        captured = cache.get("attn_probs", layer)
        assert torch.equal(captured, raw[position]), f"layer {layer} returned the wrong attention"


@pytest.mark.parametrize("layer", [0, 6, 22])
def test_linear_attention_layers_refuse_rather_than_substitute(layer: int):
    """Asking a linear layer for probabilities is an error, not a neighbouring layer's rows.

    And the error arrives *before* the forward: whether a layer runs a softmax is `layer_types`,
    known at load, so there is nothing to compute first.
    """
    model = load_model(QWEN_THINKING, device="cpu", attn_implementation="eager", required=parity_required())
    assert model.arch.is_linear_attention_layer(layer)
    with pytest.raises(ValueError, match="linear-attention layer"):
        run_with_cache(model, _ids(model), [("attn_probs", layer)])


def test_an_impossible_layer_is_refused_without_running_the_forward():
    """The fail-fast property, asserted rather than left to a stopwatch.

    A forward hook on the trunk is the cheapest witness: if it never fires, no forward ran. Pinning
    it matters because the refusal moving back after the forward would be invisible -- the same
    exception, from the same place, just seconds later and after every side effect in the pass.
    """
    model = load_model(QWEN_THINKING, device="cpu", attn_implementation="eager", required=parity_required())
    linear = next(layer for layer in range(model.n_layers) if model.arch.is_linear_attention_layer(layer))
    fired = []
    handle = model.arch.decoder_layers[0].register_forward_hook(lambda *_: fired.append(True))
    try:
        with pytest.raises(ValueError, match="linear-attention layer"):
            run_with_cache(model, _ids(model), [("attn_probs", linear)])
    finally:
        handle.remove()
    assert not fired, "the forward ran before the request was refused"


def test_plain_trunk_is_still_indexed_by_layer():
    """The mapping must be the identity where it always was -- gpt2 has no linear layers."""
    model = load_model(GPT2, device="cpu", attn_implementation="eager", required=parity_required())
    raw = _raw_attentions(model)
    assert len(raw) == model.n_layers
    for layer in range(model.n_layers):
        assert model.arch.attn_probs_index(layer) == layer
        captured = run_with_cache(model, _ids(model), [("attn_probs", layer)]).get("attn_probs", layer)
        assert torch.equal(captured, raw[layer])
