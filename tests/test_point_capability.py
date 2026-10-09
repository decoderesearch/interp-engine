"""``refuses()`` / ``serves()``: the per-point capability query, and that it cannot drift.

The query exists so that a caller holding a protocol-typed model can ask *this* model about *this*
point rather than reassembling the answer from the published tables. So the test that matters is not
that a particular point refuses -- it is that the verdict and the capture agree, point for point, on
a real checkpoint. A copy of a capability table passes a test that asserts its own contents.
"""

from __future__ import annotations

from harness import GPT2, load_model

from interp_engine import Address, capture, capture_attention
from interp_engine.points import Scope, eager_only, known_names, point_spec, tp_sharded, vllm_hookable
from interp_engine.residual_basis import ResidualBasis
from interp_engine.vllm_backend import VLLMModel


def _addresses_for(model) -> list[Address]:
    """One address per canonical point, layer 0 where the point takes a layer."""
    out = []
    for name in sorted(known_names()):
        spec = point_spec(name, model.residual_basis.n_streams)
        if spec is None:
            continue
        out.append(Address(name, 0) if spec.scope is Scope.LAYER else Address(name))
    return out


ATTENTION = ("attn_probs", "attn_scores")


def _obtain_eager(model, tokens, address: Address) -> None:
    """Fetch ``address`` the way a caller would, which for the attention pair is its own method.

    The route matters to what the agreement below proves. Probing every point with ``capture`` makes
    the attention pair agree on any backend that refuses it there -- true of all of them, since no
    backend hooks a probability matrix it never materialises -- and so passes while advertising the
    pair as unavailable on a backend that recomputes it.
    """
    if address.name in ATTENTION:
        capture_attention(model, tokens, [address.layer])
    else:
        capture(model, tokens, [address])


def test_the_walker_finds_both_kinds_of_point(gpt2) -> None:
    """Guards the loops below, which pass trivially over an empty or one-sided list.

    gpt2 is the right checkpoint for this precisely because it is old: no router, no QK-norm, no
    gated MLP or attention output, so a good half of the table refuses on it and the agreement test
    has something to disagree about.
    """
    addresses = _addresses_for(gpt2)
    assert len(addresses) >= 20
    assert sum(gpt2.serves(a) for a in addresses) >= 10, "expected gpt2 to serve the residual points"
    assert sum(not gpt2.serves(a) for a in addresses) >= 5, "expected gpt2 to lack the MoE/QK-norm points"


def test_the_verdict_matches_what_a_capture_does(gpt2, prompt) -> None:
    """The anti-drift test. Every canonical point, asked and then attempted.

    This is the property the query is for: a server deciding what to advertise, or returning a 400
    instead of a 500, must get the same answer the forward would give. A disagreement in either
    direction is a bug with a distinctive smell -- refusing a point that works reads to the caller
    as a missing feature, and promising one that does not reads as a broken backend.
    """
    tokens = gpt2.to_tokens(prompt)
    disagreed: list[str] = []
    for address in _addresses_for(gpt2):
        reason = gpt2.refuses(address)
        try:
            _obtain_eager(gpt2, tokens, address)
        except Exception as exc:  # noqa: BLE001 - any failure is a "cannot serve"
            captured, failure = False, str(exc)
        else:
            captured, failure = True, ""
        if captured and reason is not None:
            disagreed.append(f"{address}: refused but captured -- {reason}")
        elif not captured and reason is None:
            disagreed.append(f"{address}: promised but raised -- {failure}")
    assert not disagreed, "refuses() and the capture path disagree:\n" + "\n".join(disagreed)


def test_serves_is_the_inverse_of_refuses(gpt2) -> None:
    for address in _addresses_for(gpt2):
        assert gpt2.serves(address) is (gpt2.refuses(address) is None), address


def test_a_point_gpt2_has_is_served(gpt2) -> None:
    assert gpt2.refuses("resid_post", 0) is None
    assert gpt2.serves(Address("resid_post", 0))


def test_a_point_this_architecture_lacks_names_the_architecture(gpt2) -> None:
    """gpt2 has no QK-norm, and the reason has to say so rather than "unavailable".

    Which kind of "no" this is decides what the caller does next: a point the architecture does not
    have is not a backend limit and switching backend will not help.
    """
    reason = gpt2.refuses("q_norm_out", 0)
    assert reason is not None
    assert "q_norm_out" in reason or "norm" in reason.lower()


def test_the_layer_may_be_passed_either_way(gpt2) -> None:
    """``refuses("resid_post", 0)`` and ``refuses(Address("resid_post", 0))`` are one question."""
    assert gpt2.refuses("resid_post", 0) == gpt2.refuses(Address("resid_post", 0))
    assert gpt2.refuses("resid_post.0") == gpt2.refuses(Address("resid_post", 0))


def test_an_unknown_name_is_refused_rather_than_raising(gpt2) -> None:
    """A verdict, not an exception -- a caller filtering a list should not have to guard each one."""
    assert gpt2.refuses("not_a_point", 0) is not None


def test_the_attention_pair_follows_attn_implementation(prompt) -> None:
    """The one eager refusal that is about the load rather than the architecture.

    Loaded with sdpa, the pair cannot be rebuilt from a real softmax -- and the reason names
    ``attn_implementation`` so the fix is the load call rather than the backend.
    """
    sdpa = load_model(GPT2, device="cpu", attn_implementation="sdpa")
    reason = sdpa.refuses("attn_probs", 0)
    assert reason is not None and "attn_implementation" in reason
    assert sdpa.refuses("resid_post", 0) is None, "only the attention pair is affected"


def test_it_costs_no_forward(gpt2, monkeypatch) -> None:
    """Cheap enough to call per request: the query must not run the model.

    Asserted by breaking the forward. A verdict that needed one would make a server's startup
    advertisement quadratic in its point set, which is the reason this is a separate method rather
    than "try the capture and catch".
    """

    def explode(*args, **kwargs):
        raise AssertionError("refuses() ran a forward pass")

    monkeypatch.setattr(gpt2.hf_model, "forward", explode)
    for address in _addresses_for(gpt2):
        gpt2.refuses(address)


def test_the_sync_facade_forwards_both(gpt2) -> None:
    """``SyncModel`` is hand-written, so the twins are a thing a test has to check."""
    from interp_engine.sync import sync_model

    sync = sync_model(gpt2)
    assert sync.refuses("resid_post", 0) is None
    assert sync.serves("resid_post", 0)
    assert sync.refuses("q_norm_out", 0) == gpt2.refuses("q_norm_out", 0)


def _vllm(*, enforce_eager: bool = True, static_reads: tuple[Address, ...] = ()) -> VLLMModel:
    """A VLLMModel with no engine behind it.

    No ``__init__``: the real one downloads a tokenizer and builds an engine, and neither bears on a
    verdict that is defined to need neither. Same idiom as ``test_vllm_hook_availability``.
    """
    model = object.__new__(VLLMModel)
    model._engine_kwargs = {"enforce_eager": enforce_eager}
    model._residual_basis = ResidualBasis()
    model._static_reads = frozenset(static_reads)
    model.tensor_parallel_size = 1
    model.num_hidden_layers = 12  # the layer-range check reads it
    return model


def test_vllm_serves_the_hookable_points_and_refuses_the_eager_only_ones() -> None:
    """The vLLM verdict on a hooked engine is the point table, both ways round."""
    for name in sorted(vllm_hookable()):
        spec = point_spec(name, 1)
        if spec is None or spec.scope is not Scope.LAYER:
            continue
        assert _vllm().refuses(name, 0) is None, name
    for name in sorted(eager_only()):
        spec = point_spec(name, 1)
        if spec is None or spec.scope is not Scope.LAYER:
            continue
        reason = _vllm().refuses(name, 0)
        assert reason is not None and name in reason, name


def test_a_graph_engine_serves_only_what_it_declared() -> None:
    """With graphs on, a point is servable iff a static tap was baked for it.

    The refusal a caller must get *before* the request, because the one they would otherwise get is
    not an error: a hook that never fires returns fluent, unsteered text.
    """
    declared = Address("resid_post", 3)
    model = _vllm(enforce_eager=False, static_reads=(declared,))
    assert model.refuses(declared) is None
    undeclared = model.refuses("resid_post", 4)
    assert undeclared is not None and "static_points" in undeclared


def test_a_graph_engine_with_no_declarations_refuses_everything() -> None:
    model = _vllm(enforce_eager=False)
    reason = model.refuses("resid_post", 0)
    assert reason is not None


# The live vLLM half of this agreement is `test_the_verdict_matches_a_live_capture` in
# tests/test_vllm_capture_gpu.py. It cannot run here: the eager tests above initialise CUDA in this
# process, and vLLM's engine core is forked, so it comes up to `cudaErrorInitializationError`. That
# module already owns a warmed engine and the one event loop such a test needs.


def test_the_attention_pair_is_answered_by_its_own_route_on_every_backend() -> None:
    """The pair is hookable nowhere and served by ``capture_attention``, so the verdict is about it.

    Answering from the hook table instead makes every backend refuse a pair it recomputes -- a
    caller advertising from ``serves`` then hides a working feature, which is the failure this whole
    query exists to prevent, arrived at through the query itself.
    """
    hooked = _vllm()
    for name in ATTENTION:
        assert hooked.serves(Address(name, 0)), f"a hooked vLLM engine recomputes {name}"
    graph = _vllm(enforce_eager=False)
    for name in ATTENTION:
        assert "forward hooks" in (graph.refuses(Address(name, 0)) or ""), (
            f"a graph engine with no attention tap cannot recompute {name}, and should say so"
        )


def test_tensor_parallelism_is_not_a_narrowing() -> None:
    """A sharded point is gathered at collect, so a TP pod serves what a single-GPU one serves.

    Pinned as a unit test because the mistake it prevents was made downstream: a serving pod
    narrowed its advertised set by shard width, which refused points the worker gathers. No engine
    needed -- the claim is about the two tables, not about a running pod.
    """
    assert tp_sharded() & vllm_hookable(), "expected some hookable points to be sharded"
    assert "router_logits" not in tp_sharded(), "a replicated gate reaches rank 0 whole"
