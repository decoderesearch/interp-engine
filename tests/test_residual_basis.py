"""The residual-basis verdict, and the stream coordinate it makes addressable.

The numerical half runs against the shrunk DeepSeek-V4 fixture rather than a checkpoint (V4-Flash is
160 GB). That is not a compromise: the claim under test is *which tensor an address names*, which is
a property of the module tree and the slicing rule, not of the weights. Random weights answer it
exactly, and comparing against raw ``transformers`` on the identical model makes the comparison
tight rather than approximate.
"""

from __future__ import annotations

import pytest
import torch

from interp_engine import (
    Address,
    ResidualBasis,
    ResidualBasisUnsupported,
    eager_residual_basis,
    vllm_residual_basis,
)
from interp_engine.capture import run_with_cache
from interp_engine.lens import capture_residuals
from interp_engine.steer import SteerSpec, steer
from tests.synthetic_families import eager_shrunk_deepseek_v4

DSV4 = "DeepseekV4ForCausalLM"


# --- the verdict itself, with no model attached -------------------------------------------------


def test_a_conventional_trunk_has_every_invariant_and_says_nothing():
    """The overwhelmingly common case must be silent: no blockers, no caveats, nothing to explain."""
    basis = eager_residual_basis(architecture="GPT2LMHeadModel")
    assert (basis.n_streams, basis.additive, basis.sequential, basis.lens_valid) == (1, True, True, True)
    assert basis.blockers == () and basis.caveats == ()
    assert basis.single_additive_stream

    basis.require_single_stream("resid_post")
    basis.require_sequential("resid_mid")
    basis.require_lens()


def test_a_parallel_block_keeps_its_lens_and_loses_only_the_middle():
    """The reason `sequential` is its own field rather than folded into `additive`.

    GPT-J's residual is one stream entered by addition, so the logit lens is exactly as valid there
    as on GPT-2. Only `resid_mid` -- a residual *between* sublayers that are not sequenced -- fails.
    A single flag would have refused the lens on a whole family of models it works on.
    """
    basis = eager_residual_basis(parallel_attn_mlp=True, architecture="GPTJForCausalLM")
    assert basis.lens_valid and basis.additive and basis.single_additive_stream
    assert not basis.sequential

    basis.require_lens()
    basis.require_single_stream("resid_post")
    with pytest.raises(ResidualBasisUnsupported, match="no residual between them"):
        basis.require_sequential("resid_mid")


def test_a_hyper_connection_trunk_blocks_the_lens_and_demands_a_stream():
    basis = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    assert basis.n_streams == 4
    assert not basis.additive and not basis.lens_valid and not basis.single_additive_stream
    assert basis.sequential, "hyper-connections are about how many streams, not about ordering"

    with pytest.raises(ResidualBasisUnsupported, match="4 parallel residual streams"):
        basis.require_single_stream("resid_post")
    basis.require_single_stream("resid_post", stream=2)


def test_the_two_refusals_for_one_point_do_not_read_alike():
    """ "This model has no such tensor" and "say which one" are a dead end and a one-word fix."""
    single = eager_residual_basis(architecture="GPT2LMHeadModel")
    multi = eager_residual_basis(n_residual_streams=4, architecture=DSV4)

    with pytest.raises(ResidualBasisUnsupported, match="has no stream axis") as no_axis:
        single.require_single_stream("resid_post", stream=2)
    with pytest.raises(ResidualBasisUnsupported, match="does not name a single one") as unqualified:
        multi.require_single_stream("resid_post")

    assert "Drop the coordinate" in str(no_axis.value)
    assert "stream=0" in str(unqualified.value), "the refusal must show the fix, not just the problem"


def test_an_out_of_range_stream_names_the_range():
    basis = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    with pytest.raises(ResidualBasisUnsupported, match=r"stream=9 is out of range .*valid: 0\.\.3"):
        basis.require_single_stream("resid_post", stream=9)


# --- the stream reduction: the answer to `require_lens`'s refusal ------------------------------


def test_a_conventional_trunk_wants_no_reduction_and_refuses_one():
    basis = eager_residual_basis(architecture="GPT2LMHeadModel")
    basis.require_stream_reduction("none", point="resid_post")
    with pytest.raises(ResidualBasisUnsupported, match="has no stream axis at 'resid_post'"):
        basis.require_stream_reduction("mean", point="resid_post")


def test_a_hyper_connection_trunk_demands_a_reduction_and_names_the_choices():
    """The gate that lets a fitted lens be served where `require_lens` alone is a dead end."""
    basis = vllm_residual_basis(n_residual_streams=4, architecture=DSV4)
    with pytest.raises(ResidualBasisUnsupported, match="has to say which d_model vector") as refused:
        basis.require_stream_reduction("none", point="resid_streams")
    for choice in ("mean", "sum", "select"):
        assert choice in str(refused.value), "the refusal must show what to declare"

    basis.require_stream_reduction("mean", point="resid_streams")
    basis.require_stream_reduction("select", 3, point="resid_streams")
    with pytest.raises(ResidualBasisUnsupported, match=r"stream index 4 is out of range"):
        basis.require_stream_reduction("select", 4, point="resid_streams")
    with pytest.raises(ValueError, match="needs a stream index"):
        basis.require_stream_reduction("select", point="resid_streams")


def test_a_sublayer_point_on_a_hyper_connection_trunk_takes_no_reduction():
    """The one case the trunk's stream count gets wrong on its own.

    ``attn_out`` is that block's own d_model-wide output, before it is scattered across the streams,
    so a lens fitted there has nothing to reduce even though the model carries four streams. Gating
    on the stream count alone would demand a reduction for an axis the tensor does not have.
    """
    basis = vllm_residual_basis(n_residual_streams=4, architecture=DSV4)
    assert basis.stacked_at("resid_streams") and not basis.stacked_at("attn_out")

    basis.require_stream_reduction("none", point="attn_out")
    with pytest.raises(ResidualBasisUnsupported, match="has no stream axis at 'attn_out'"):
        basis.require_stream_reduction("mean", point="attn_out")


def test_omitting_the_point_asks_about_the_residual_points():
    """The strict reading, for a caller that does not know which point yet."""
    basis = vllm_residual_basis(n_residual_streams=4, architecture=DSV4)
    with pytest.raises(ResidualBasisUnsupported, match="4 parallel residual streams"):
        basis.require_stream_reduction("none")
    basis.require_stream_reduction("mean")


def test_reducing_through_the_basis_checks_before_it_collapses():
    basis = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    stack = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    assert torch.equal(basis.reduce_streams(stack, "mean", point="resid_streams"), stack.mean(dim=-2))
    assert torch.equal(basis.reduce_streams(stack, "select", 1, point="resid_streams"), stack[:, 1, :])
    # The trap `select_stream` documents, in the reducing direction: a stack of the wrong width would
    # average an axis that is not the stream axis and come back a believable shape.
    with pytest.raises(ValueError, match="expects 4 streams"):
        basis.reduce_streams(torch.zeros(2, 5, 3), "mean", point="resid_streams")


def test_vllm_reports_the_same_model_and_a_missing_hook_rather_than_a_missing_wire():
    """The backends must not disagree about the *model*, only about what they can serve.

    The blocker names the hook, not the transport, and that distinction is the whole value of the
    field: the wire grammar carries ``resid_post.7.stream-2`` end to end today, so a reader told the
    coordinate cannot cross would go and re-implement something that already works. What is actually
    missing is a residual reconstruction that means anything on this trunk.
    """
    eager = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    vllm = vllm_residual_basis(n_residual_streams=4, architecture=DSV4)

    assert (vllm.n_streams, vllm.additive, vllm.lens_valid) == (eager.n_streams, eager.additive, eager.lens_valid)
    assert eager.stream_addressable and not vllm.stream_addressable

    with pytest.raises(ResidualBasisUnsupported, match="no residual hook reads this trunk") as refused:
        vllm.require_single_stream("resid_post", stream=2)
    assert "does carry a stream coordinate" in str(refused.value), "the wire is not the blocker"
    assert "eager backend" in vllm.remedy


def test_the_capability_report_is_json_friendly():
    described = eager_residual_basis(n_residual_streams=4, architecture=DSV4).describe()
    import json

    assert json.loads(json.dumps(described))["n_streams"] == 4
    assert set(described) == {
        "backend",
        "architecture",
        "n_streams",
        "additive",
        "sequential",
        "lens_valid",
        "stream_addressable",
        "blockers",
        "caveats",
        "remedy",
    }


# --- the slicing rule ----------------------------------------------------------------------------


def test_selecting_a_stream_refuses_a_tensor_with_no_stream_axis():
    """The failure this rule exists to prevent: indexing the wrong axis also "works"."""
    basis = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    conventional = torch.zeros(1, 8, 128)
    with pytest.raises(ResidualBasisUnsupported, match=r"no axis of 4 second-from-last"):
        basis.select_stream(conventional, 2)


def test_replacing_a_stream_leaves_the_others_and_the_caller_untouched():
    basis = eager_residual_basis(n_residual_streams=4, architecture=DSV4)
    original = torch.randn(1, 8, 4, 16)
    written = basis.replace_stream(original, 2, torch.ones(1, 8, 16))

    assert torch.equal(written[:, :, 2, :], torch.ones(1, 8, 16))
    for other in (0, 1, 3):
        assert torch.equal(written[:, :, other, :], original[:, :, other, :])
    assert not torch.equal(original[:, :, 2, :], torch.ones(1, 8, 16)), "must not write through the model's tensor"


# --- against a real (tiny) DeepSeek-V4 -------------------------------------------------------


@pytest.fixture(scope="module")
def dsv4():
    return eager_shrunk_deepseek_v4()


@pytest.fixture(scope="module")
def dsv4_tokens():
    return torch.randint(0, 512, (1, 8))


def test_the_fixture_really_carries_four_streams(dsv4):
    basis = dsv4.residual_basis
    assert basis.n_streams == 4 and not basis.lens_valid
    assert basis.architecture == DSV4 and basis.backend == "eager"


def test_an_unqualified_residual_point_still_refuses_and_now_names_the_fix(dsv4):
    """The refusal moves rather than disappearing -- and gains an alternative it did not have."""
    with pytest.raises(ResidualBasisUnsupported, match="4 parallel residual streams") as excinfo:
        dsv4.resolve_point("resid_post", 1)
    assert "stream=0" in str(excinfo.value)


def test_a_stream_qualified_residual_point_resolves(dsv4):
    module, side = dsv4.resolve_point("resid_post", 1, stream=2)
    assert side == "output"
    assert module is dsv4.arch.decoder_layers[1]


def test_a_captured_stream_is_exactly_that_slice_of_the_raw_forward(dsv4, dsv4_tokens):
    """The load-bearing numerical claim of this phase, checked against transformers itself.

    `Address("resid_post", 1, stream=k)` must be `hidden_states[:, :, k, :]` and nothing else --
    not a neighbouring stream, not a collapsed combination, not the right numbers off the wrong
    axis. Compared exactly rather than with a tolerance: the engine is not recomputing anything
    here, it is reading the same tensor out of the same forward, so any difference is a bug.
    """
    with torch.no_grad():
        raw = dsv4.hf_model(dsv4_tokens, output_hidden_states=True, use_cache=False)
    # `hidden_states[i]` is the input to block `i`, so block 0's output is `[1]`. Deliberately not
    # the last entry, which `DeepseekV4HyperHead` has already collapsed to a single `d_model` vector
    # -- the streams exist between the blocks, not after them.
    reference = raw.hidden_states[1]
    assert reference.shape[-2] == 4, f"expected a stream axis, got {tuple(reference.shape)}"

    for stream in range(4):
        cache = run_with_cache(dsv4, dsv4_tokens, [Address("resid_post", 0, stream=stream)])
        captured = cache.get("resid_post", 0, stream=stream)
        assert torch.equal(captured, reference[:, :, stream, :])


def test_two_streams_captured_together_do_not_collapse_onto_one_hook(dsv4, dsv4_tokens):
    """Both addresses resolve to the same module and side; only the coordinate separates them.

    The hook-sharing optimization keys on the resolved target, so this is exactly where two streams
    would quietly become one tensor stored twice.
    """
    cache = run_with_cache(dsv4, dsv4_tokens, [Address("resid_post", 1, stream=0), Address("resid_post", 1, stream=3)])
    first = cache.get("resid_post", 1, stream=0)
    last = cache.get("resid_post", 1, stream=3)
    assert first.shape == last.shape
    assert not torch.equal(first, last)


def test_a_captured_stream_has_the_shape_the_capture_contract_promises(dsv4, dsv4_tokens):
    """`[batch, seq, d_model]`, i.e. the stream axis is gone rather than left as a length-1 stub."""
    cache = run_with_cache(dsv4, dsv4_tokens, [Address("resid_post", 1, stream=2)])
    assert cache.get("resid_post", 1, stream=2).shape == (1, 8, dsv4.d_model)


def test_a_stream_coordinate_on_a_point_that_has_no_streams_is_refused(dsv4):
    """`attn_out` is the block's own d_model-wide output, before it is scattered across streams.

    Accepting the coordinate here and ignoring it would be the exact silent-selector failure the
    address grammar rejects unknown coordinates to avoid -- and the tensor returned would look right.
    """
    with pytest.raises(ValueError, match="does not carry residual streams separately"):
        dsv4.resolve_point("attn_out", 1, stream=2)


def test_the_sublayer_points_still_work_unqualified_on_a_multi_stream_trunk(dsv4, dsv4_tokens):
    """The verdict must gate the residual points only. `attn_out` was never in question."""
    cache = run_with_cache(dsv4, dsv4_tokens, [Address("attn_out", 1)])
    assert cache.get("attn_out", 1).shape[-1] == dsv4.d_model


def test_the_lens_refuses_before_running_a_forward(dsv4, dsv4_tokens):
    """`dict[int, Tensor]` cannot say which stream a row is, so the refusal belongs before the cost."""
    with pytest.raises(ResidualBasisUnsupported, match="logit/tuned lens"):
        capture_residuals(dsv4, dsv4_tokens, [0, 1])


def test_steering_one_stream_leaves_the_others_alone(dsv4, dsv4_tokens):
    """A d_model vector added to a (..., streams, d_model) tensor broadcasts across every stream.

    That is a different intervention than the one asked for, and no capture of a single stream would
    show it -- the steered stream looks exactly as expected either way.
    """
    d_model = dsv4.d_model
    spec = SteerSpec(vector=torch.ones(d_model), layer=1, coeff=10.0, point="resid_post", stream=1)

    requests = [Address("resid_post", 1, stream=s) for s in range(4)]
    before = {s: run_with_cache(dsv4, dsv4_tokens, requests).get("resid_post", 1, stream=s) for s in range(4)}
    with steer(dsv4, [spec]):
        after = {s: run_with_cache(dsv4, dsv4_tokens, requests).get("resid_post", 1, stream=s) for s in range(4)}

    assert torch.equal(after[1], before[1] + 10.0)
    for untouched in (0, 2, 3):
        assert torch.equal(after[untouched], before[untouched])


def test_steering_a_multi_stream_trunk_without_a_stream_is_refused(dsv4):
    with (
        pytest.raises(ResidualBasisUnsupported, match="4 parallel residual streams"),
        steer(dsv4, [SteerSpec(vector=torch.ones(dsv4.d_model), layer=1, point="resid_post")]),
    ):
        pass


def test_the_protocol_property_is_satisfied_by_the_real_model(dsv4):
    """`residual_basis` is on `InterpModel`, so a structural check is the point of this one."""
    assert isinstance(dsv4.residual_basis, ResidualBasis)
