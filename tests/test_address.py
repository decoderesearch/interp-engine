"""The address grammar, tested as a grammar: round trip, injectivity, and what must be rejected.

A string form is only trustworthy if these three hold together. Round trip alone allows two addresses
to share a spelling; injectivity alone allows a form nothing can read back; and both together still
allow a lenient parser that accepts the legacy ``name:layer`` key and quietly means something else --
which is the specific bug this type replaces.

The generated set is built from ``points.known_names()`` rather than a hand-written list, so a point
added to the table is covered here without anyone remembering to add it.
"""

from __future__ import annotations

import dataclasses

import pytest

from interp_engine import points
from interp_engine.address import (
    MODULE_PATH_PREFIX,
    Address,
    AddressError,
    UnknownCoordinate,
    coordinate_names,
    format_address,
    parse_address,
    to_address,
)

_LAYERS = (None, 0, 1, 7, 128)
_STREAMS = (None, 0, 3)

_MODULE_PATHS = (
    "model.layers.5.mlp.down_proj",
    "transformer.h.0.attn.c_attn",
    "model.layers.11.mixer.conv1d",
    "model.H_module.layers.0",
)


def _generated() -> list[Address]:
    """Every canonical point crossed with every coordinate combination, plus the module paths."""
    out = [
        Address(name, layer, stream)
        for name in sorted(points.known_names())
        for layer in _LAYERS
        for stream in _STREAMS
    ]
    out.extend(Address(path) for path in _MODULE_PATHS)
    return out


# --- the three properties -----------------------------------------------------


@pytest.mark.parametrize("address", _generated(), ids=str)
def test_every_address_round_trips_through_its_string_form(address: Address) -> None:
    assert parse_address(str(address)) == address


def test_the_string_form_is_injective() -> None:
    """No two distinct addresses share a spelling.

    Asserted over the generated set rather than argued from the grammar, because the property is what
    consumers rely on: the vLLM wire key, the dump key and the URL form are all this string, so a
    collision would silently merge two captures.
    """
    seen: dict[str, Address] = {}
    for address in _generated():
        text = str(address)
        assert text not in seen, f"{address!r} and {seen[text]!r} both emit {text!r}"
        seen[text] = address


@pytest.mark.parametrize(
    "text",
    [
        "resid_post:5",  # the legacy vLLM wire key, which must not parse as a point name
        "q:12",
        "model.layers.5.mlp.down_proj",  # a module path missing its `path.` prefix
        "",
        ".",
        "resid_post.",
        ".5",
        "Resid_Post.5",  # uppercase is outside the name grammar
        "resid post.5",
        "resid_post.5.",
        "resid_post.-1",
        "resid_post.5.stream",  # a coordinate with no value
        "resid_post.5.stream-",
        "resid_post.5.stream-x",
        "resid_post.5.stream-2.stream-3",  # a repeated coordinate
        "resid_post.5.2",  # a second bare integer is not a coordinate
        MODULE_PATH_PREFIX,  # the reserved prefix alone names nothing
        f"{MODULE_PATH_PREFIX}.",
    ],
)
def test_malformed_input_is_rejected_with_a_message(text: str) -> None:
    with pytest.raises(AddressError) as excinfo:
        parse_address(text)
    assert str(excinfo.value)


def test_a_non_string_is_rejected_rather_than_coerced() -> None:
    with pytest.raises(AddressError, match="a string"):
        parse_address(5)  # type: ignore[arg-type]


# --- malformed vs. merely newer -----------------------------------------------


def test_an_unknown_coordinate_is_named_and_distinguishable_from_malformed_input() -> None:
    """The one diagnostic that earns its keep across a version boundary.

    A client newer than its worker is the realistic way an unsupported coordinate crosses the wire, so
    the refusal has to say *which* coordinate rather than reporting a malformed key -- and it has to be
    catchable on its own, which is why ``UnknownCoordinate`` is a subclass.
    """
    with pytest.raises(UnknownCoordinate) as excinfo:
        parse_address("attn_out.5.site-1")
    message = str(excinfo.value)
    assert "site" in message
    assert "does not support" in message

    # Still an AddressError, so a caller that does not care keeps one except clause.
    assert isinstance(excinfo.value, AddressError)

    # And a genuinely malformed coordinate is NOT reported as an unknown one.
    with pytest.raises(AddressError) as plain:
        parse_address("attn_out.5.site")
    assert not isinstance(plain.value, UnknownCoordinate)


# --- the extension rules ------------------------------------------------------


def test_field_order_is_append_only() -> None:
    """``Address("resid_post", 5, 2)`` must keep meaning ``stream=2``.

    Inserting a field ahead of ``stream`` would silently change what every positional construction in
    the tree already means, and there is no way to notice that from a diff -- so the order is pinned
    here rather than described in a comment.
    """
    assert [f.name for f in dataclasses.fields(Address)] == ["name", "layer", "stream"]
    assert coordinate_names() == ("stream",)
    assert Address("resid_post", 5, 2).stream == 2


def test_appending_a_coordinate_needs_no_parser_or_emitter_edit() -> None:
    """The claim that makes the closed field set cheap, proven rather than asserted.

    A generated copy of the dataclass with one extra coordinate must round-trip and stay injective
    through the *same* emitter and parser, with no edit to either. If someone hardcodes ``stream``
    anywhere in the grammar, this fails.
    """
    extended = dataclasses.make_dataclass(
        "ExtendedAddress",
        [
            ("name", str),
            ("layer", "int | None", dataclasses.field(default=None)),
            ("stream", "int | None", dataclasses.field(default=None)),
            ("site", "int | None", dataclasses.field(default=None)),
        ],
        frozen=True,
    )
    assert coordinate_names(extended) == ("stream", "site")

    generated = [
        extended("attn_out", layer, stream, site)
        for layer in (None, 0, 5)
        for stream in (None, 2)
        for site in (None, 1)
    ]
    seen: dict[str, object] = {}
    for address in generated:
        text = format_address(address)
        assert parse_address(text, extended) == address
        assert text not in seen, f"{address!r} collides with {seen[text]!r}"
        seen[text] = address

    # The new coordinate is emitted last, after `stream`, because it was appended last.
    assert format_address(extended("attn_out", 5, 2, 1)) == "attn_out.5.stream-2.site-1"

    # And the *old* parser still rejects the new spelling by name, rather than dropping it.
    with pytest.raises(UnknownCoordinate, match="site"):
        parse_address("attn_out.5.site-1")


def test_an_old_string_keeps_parsing_and_emitting_identically() -> None:
    """All "backward compatible" means here: coordinates are optional and emitted only when set."""
    assert str(Address("resid_post", 5)) == "resid_post.5"
    assert str(Address("embeddings")) == "embeddings"
    assert parse_address("resid_post.5") == Address("resid_post", 5)
    assert parse_address("embeddings") == Address("embeddings")


# --- the shapes the grammar has to get right ----------------------------------


def test_the_examples_from_the_grammar_emit_what_the_docs_say() -> None:
    assert str(Address("embeddings")) == "embeddings"
    assert str(Address("resid_post", 5)) == "resid_post.5"
    assert str(Address("resid_post", 5, 2)) == "resid_post.5.stream-2"
    assert str(Address("model.layers.5.mlp.down_proj")) == "path.model.layers.5.mlp.down_proj"


def test_a_stream_without_a_layer_round_trips() -> None:
    """Structurally legal and unambiguous, so the grammar allows it rather than special-casing."""
    assert str(Address("resid_post", None, 2)) == "resid_post.stream-2"
    assert parse_address("resid_post.stream-2") == Address("resid_post", None, 2)


def test_zero_is_a_coordinate_value_and_not_an_absent_one() -> None:
    """``stream=0`` and ``stream=None`` are different addresses, so they must spell differently."""
    assert str(Address("resid_post", 0, 0)) == "resid_post.0.stream-0"
    assert Address("resid_post", 0, 0) != Address("resid_post", 0)
    assert parse_address("resid_post.0.stream-0") == Address("resid_post", 0, 0)


def test_a_module_path_takes_no_coordinates() -> None:
    """A path names one exact module, so a coordinate beside it would have nothing to select.

    It also could not be emitted: ``path.`` takes the rest of the string by design, so a layer after
    it would be indistinguishable from another path segment.
    """
    with pytest.raises(AddressError, match="takes no coordinates"):
        Address("model.layers.5.mlp", 5)
    with pytest.raises(AddressError, match="takes no coordinates"):
        Address("model.layers.5.mlp", None, 2)


def test_the_module_path_prefix_is_reserved_as_a_point_name() -> None:
    """Otherwise ``Address("path")`` would emit ``path``, which reads back as an empty module path."""
    with pytest.raises(AddressError, match="reserved"):
        Address(MODULE_PATH_PREFIX)
    assert MODULE_PATH_PREFIX not in points.known_names()


def test_the_prefix_disambiguates_a_path_that_looks_like_a_point_and_a_layer() -> None:
    """``path.5`` is the module path ``5``, not the point ``path`` at layer 5.

    Worth pinning because it is the one place the two productions of the grammar could have
    overlapped, and the reserved prefix is what stops them: since ``path`` cannot be a point name,
    anything after ``path.`` is unambiguously a raw path -- and ``5`` really is a reachable one, being
    how a ``ModuleList`` entry is spelled.
    """
    assert parse_address("path.5") == Address("5")
    assert parse_address("path.5").is_module_path
    assert str(Address("5")) == "path.5"


def test_passing_an_already_emitted_path_is_caught_rather_than_double_prefixed() -> None:
    """An easy mistake once the emitted form is visible in logs and URLs."""
    with pytest.raises(AddressError, match="pass the module path itself"):
        Address("path.model.layers.0")


def test_a_negative_coordinate_is_refused() -> None:
    """It has no meaning, and the grammar's values are digits only, so it could not be emitted."""
    with pytest.raises(AddressError, match="layer must not be negative"):
        Address("resid_post", -1)
    with pytest.raises(AddressError, match="stream must not be negative"):
        Address("resid_post", 0, -1)


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(AddressError, match="needs a name"):
        Address("")


# --- coercion from what callers pass -----------------------------------------


def test_coercion_accepts_the_three_shapes_callers_actually_pass() -> None:
    assert to_address(Address("resid_post", 5)) == Address("resid_post", 5)
    assert to_address("resid_post.5") == Address("resid_post", 5)
    assert to_address("embeddings") == Address("embeddings")
    assert to_address(("resid_post", 5)) == Address("resid_post", 5)
    assert to_address(("resid_post", 5, 2)) == Address("resid_post", 5, 2)
    assert to_address(("embeddings",)) == Address("embeddings")


def test_coercion_carries_every_coordinate_rather_than_the_first_two() -> None:
    """The truncation that motivated this type: ``out.append((p[0], p[1]))`` dropped the rest."""
    assert to_address(("resid_post", 5, 2)).stream == 2


def test_coercion_refuses_an_over_long_tuple_instead_of_truncating() -> None:
    with pytest.raises(AddressError, match="at most"):
        to_address(("resid_post", 5, 2, 9))


def test_coercion_refuses_a_shape_it_cannot_read() -> None:
    with pytest.raises(AddressError, match="cannot read an address"):
        to_address(5)  # type: ignore[arg-type]
    with pytest.raises(AddressError, match="at least a name"):
        to_address(())
