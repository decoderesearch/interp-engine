"""The invariant sweep: every family the audit can build, checked against three identities.

See ``tests/invariants.py`` for what the identities are and why random weights are enough. This file is
the sweep plus the cases that random defaults cannot reach -- a residual multiplier, a sublayer that
adds the residual itself, a value the family scales -- each one a shape that was silently wrong before
it was checked here.
"""

from __future__ import annotations

import family_coverage as fc
import pytest
import torch
from interp_engine import EagerModel, per_head_value, run_with_cache
from invariants import (
    EXCEPTIONS,
    INPUT_IDS,
    INVARIANTS,
    NOT_MINIATURIZABLE,
    check,
    config_for,
    probed_architectures,
    tiny_model,
)

PROBED = probed_architectures()
MINIATURIZABLE = [arch for arch in PROBED if arch not in NOT_MINIATURIZABLE]


@pytest.fixture(scope="module")
def _models() -> dict[str, object]:
    """One tiny model per family, built once: 70-odd forwards is the cost of this file."""
    return {}


def _model(cache: dict[str, object], arch: str) -> object:
    if arch not in cache:
        cache[arch] = tiny_model(arch)
    return cache[arch]


# --- the sweep ---------------------------------------------------------------


@pytest.mark.parametrize("arch", MINIATURIZABLE)
def test_each_invariant_holds_or_is_refused_with_a_reason(arch: str, _models):
    """The whole point: a wrong number is a failure, and so is an exception that explains nothing.

    A family with no such quantity (sparse MoE and the neuron basis, MLA and ``value``, a Mamba layer
    and attention) is expected to raise a ``ValueError`` that says which and why. What fails here is a
    mismatch -- the plausible-looking tensor from one sublayer over -- or an ``AttributeError`` from
    inside the capture, which means the engine went looking for a module the family does not have.
    """
    model = _model(_models, arch)
    for invariant in INVARIANTS:
        result = check(model, invariant)
        expected_failure = EXCEPTIONS.get((arch, invariant))
        if expected_failure:
            assert not result.acceptable, (
                f"{arch}'s {invariant} invariant now holds ({result.detail}); delete its EXCEPTIONS entry"
            )
            continue
        assert result.acceptable, f"{arch} {invariant}: {result.outcome} -- {result.detail}"


@pytest.mark.parametrize("arch", MINIATURIZABLE)
def test_every_family_the_audit_can_build_can_also_be_built_tiny(arch: str, _models):
    """Otherwise a family silently leaves the sweep, which is how a gap survives a green suite."""
    assert _model(_models, arch) is not None


def test_the_skip_list_is_only_families_that_really_do_not_build():
    """A stale skip is worse than none: it reads as "checked" while nothing checks it."""
    still_broken = []
    for arch in NOT_MINIATURIZABLE:
        try:
            tiny_model(arch)
        except Exception:  # noqa: BLE001 - any failure means the entry is still earned
            still_broken.append(arch)
    assert sorted(still_broken) == sorted(NOT_MINIATURIZABLE), (
        f"these now build tiny; remove them from NOT_MINIATURIZABLE: "
        f"{sorted(set(NOT_MINIATURIZABLE) - set(still_broken))}"
    )


def test_the_skip_list_and_the_exceptions_name_families_that_exist():
    """Both tables are keyed by architecture, and a renamed family would leave a dead entry behind."""
    known = set(fc.text_generation_archs())
    assert not set(NOT_MINIATURIZABLE) - known
    assert not {arch for arch, _ in EXCEPTIONS} - known


# --- what a random config cannot turn on -------------------------------------


def _built(arch: str, **config_overrides) -> EagerModel:
    _, hf_class = fc.hf_class_for(arch)
    config = config_for(hf_class, arch)
    for field, value in config_overrides.items():
        setattr(config, field, value)
    torch.manual_seed(0)
    return EagerModel(arch, hf_model=hf_class(config).eval(), tokenizer=object(), device="cpu")


def test_a_residual_multiplier_scales_the_contribution():
    """Granite scales each sublayer's output by `residual_multiplier` (0.22 on the 3.x checkpoints).

    So `attn_out_post` is not the attention module's output there, and a decomposition built from the
    raw output is off by a constant factor with nothing in the shape or the magnitude to say so. The
    config default is 1.0, which is why the sweep cannot see this and this test sets it.
    """
    model = _built("GraniteForCausalLM", residual_multiplier=0.22)
    assert model.arch.quirks.residual_multipliers == (0.22, 0.22)

    names = ("resid_pre", "attn_out", "attn_out_post", "mlp_out_post", "resid_post")
    cache = run_with_cache(model, INPUT_IDS, [(name, 0) for name in names])
    got = {name: cache.get(name, 0).float() for name in names}

    torch.testing.assert_close(got["attn_out_post"], got["attn_out"] * 0.22, rtol=1e-5, atol=1e-6)
    rebuilt = got["resid_pre"] + got["attn_out_post"] + got["mlp_out_post"]
    torch.testing.assert_close(rebuilt, got["resid_post"], rtol=1e-5, atol=1e-6)


def test_a_derived_multiplier_is_read_off_the_block():
    """MiniCPM3's is `scale_depth / sqrt(n_layers)`, which no config field states."""
    model = _built("MiniCPM3ForCausalLM", scale_depth=1.4, num_key_value_heads=4)
    attn_side, mlp_side = model.arch.quirks.residual_multipliers
    assert attn_side == mlp_side == pytest.approx(model.hf_model.model.layers[0].residual_scale)
    assert attn_side != 1.0


@pytest.mark.parametrize("arch", ["BloomForCausalLM", "MptForCausalLM"])
def test_a_sublayer_handed_the_residual_is_not_asked_for_its_own_output(arch: str):
    """BLOOM's MLP takes the residual and returns the sum, so its output is `resid_post`.

    Which makes the naive binding wrong by an entire residual stream while staying the right shape:
    `mlp_out` would be the block's output and, on BLOOM, `attn_out` would be `resid_mid`. The
    contribution is the projection's output, one module in.
    """
    model = _built(arch)
    mlp_out_module, side = model.resolve_point("mlp_out", 0)
    assert side == "output"
    assert mlp_out_module is model.arch.mlp_projection(0, "down"), "mlp_out must skip the residual add"

    names = ("resid_pre", "attn_out_post", "mlp_out", "mlp_out_post", "resid_post")
    cache = run_with_cache(model, INPUT_IDS, [(name, 0) for name in names])
    got = {name: cache.get(name, 0).float() for name in names}
    assert (got["mlp_out"] - got["resid_post"]).abs().max() > 1e-3, "mlp_out is still the whole stream"
    rebuilt = got["resid_pre"] + got["attn_out_post"] + got["mlp_out_post"]
    torch.testing.assert_close(rebuilt, got["resid_post"], rtol=1e-4, atol=1e-5)


def test_a_value_the_family_rescales_is_rescaled_before_dfa():
    """MiMo-V2 multiplies `v_proj`'s output by `v_scale` before attention reads it.

    The raw point stays the projection's output, since that is the module's; `per_head_value`, which
    exists to feed DFA, applies what the family applies.
    """
    model = _built("MiMoV2FlashForCausalLM")
    scale = model.arch.value_scale(0)
    assert scale != 1.0, "MiMo-V2's config no longer scales values; this test has nothing to check"

    cache = run_with_cache(model, INPUT_IDS, [("value", 0)])
    raw = cache.get("value", 0).float()
    per_head = per_head_value(model, cache, 0).float()
    torch.testing.assert_close(per_head.reshape(*raw.shape), raw * scale, rtol=1e-5, atol=1e-6)


def test_a_wider_value_head_splits_by_its_own_width():
    """MiMo-V2's value head is 128 wide where its q/k head is 64, so `head_dim` mis-splits `value`."""
    model = _built("MiMoV2FlashForCausalLM")
    assert model.arch.value_head_dim_for_layer(0) != model.arch.head_dim_for_layer(0)
    cache = run_with_cache(model, INPUT_IDS, [("value", 0)])
    assert per_head_value(model, cache, 0).shape[-1] == model.arch.value_head_dim_for_layer(0)


def test_a_head_split_that_does_not_account_for_the_tensor_refuses():
    """Inkling sizes its sliding layers from `swa_*` fields, so the model-level head count is wrong there.

    Refusing beats reshaping: with powers of two everywhere the wrong split usually divides, and then it
    returns a full-looking tensor holding a mixture of heads.
    """
    model = _built("InklingForCausalLM")
    cache = run_with_cache(model, INPUT_IDS, [("value", 0)])
    with pytest.raises(ValueError, match="Cannot split value into heads"):
        per_head_value(model, cache, 0)
