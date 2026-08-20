"""Telling a routing flip apart from kernel noise, which the comparison table cannot do on its own.

The claim the analysis makes is narrow and worth pinning: a top-k is a *selection*, so two engines
that agree about the logits to five decimal places can still disagree about which experts fire, and
when they do, the token's `mlp_out` is a different expert's output rather than a rounding of the
same one. Everything below is about not confusing those two situations -- in either direction.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comparison.routing_flips import compare  # noqa: E402


def _logits(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_a_near_tie_that_lands_the_other_way_is_reported_as_a_flip():
    """The whole point: the logits are within 2e-3 and the selection is different."""
    reference = {"router_logits.3": _logits([[1.0, 0.999, 0.0, 0.0]])}
    engine = {"router_logits.3": _logits([[0.999, 1.0, 0.0, 0.0]])}
    (row,) = compare(reference, engine, top_k=1)
    assert (row["layer"], row["tokens"], row["flipped"]) == ("3", 1, 1)


def test_the_same_experts_in_a_different_order_is_not_a_flip():
    """A set, not a ranking: the same k experts fire and each is weighted by its own logit, so a
    swap between two of them changes nothing about which expert saw the token."""
    reference = {"router_logits.0": _logits([[1.0, 0.999, 0.0, 0.0]])}
    engine = {"router_logits.0": _logits([[0.999, 1.0, 0.0, 0.0]])}
    (row,) = compare(reference, engine, top_k=2)
    assert row["flipped"] == 0


def test_the_flipped_tokens_own_mlp_output_is_reported_beside_it():
    """One flipped token out of fourteen drags a layer's cosine down while the rest are fine, so the
    per-token figures are the evidence -- a layer-level number cannot distinguish the two shapes."""
    reference = {
        "router_logits.5": _logits([[1.0, 0.999], [1.0, 0.0]]),
        "mlp_out.5": _logits([[1.0, 0.0], [1.0, 0.0]]),
    }
    engine = {
        "router_logits.5": _logits([[0.999, 1.0], [1.0, 0.0]]),
        "mlp_out.5": _logits([[0.0, 1.0], [1.0, 0.0]]),
    }
    (row,) = compare(reference, engine, top_k=1)
    assert row["cos_flipped"] == [0.0]
    assert row["cos_agreed_min"] == 1.0


def test_a_layer_the_two_engines_shaped_differently_is_declined_rather_than_compared():
    """Granite's reversed router tuple looked exactly like this, and a top-k over 8 columns of a
    32-expert bank is a number that would have meant nothing."""
    reference = {"router_logits.0": _logits([[1.0, 0.0]])}
    engine = {"router_logits.0": _logits([[1.0, 0.0, 0.0, 0.0]])}
    (row,) = compare(reference, engine, top_k=1)
    assert row["shapes"] == ((1, 2), (1, 4))


def test_layers_are_reported_in_trunk_order_rather_than_alphabetically():
    reference = {f"router_logits.{layer}": _logits([[1.0, 0.0]]) for layer in (2, 12, 23)}
    assert [row["layer"] for row in compare(reference, dict(reference), top_k=1)] == ["2", "12", "23"]
