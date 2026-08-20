"""The canonical capture address, and the one string form it round-trips through.

An address says *which tensor*, in enough coordinates to name it unambiguously. It replaces
``Point = tuple[str, int]``, which conflated the tensor with the invocation and could not be extended
without every consumer silently truncating the extra element -- ``out.append((p[0], p[1]))`` was a
real line in ``capture.py``, and ``key.rsplit(":", 1)`` a real line in ``vllm_capture``.

**Three coordinates, and the field set is closed.**

``name``
    A canonical point name (``resid_post``), or a dotted module path for the open point set's escape
    hatch (``model.layers.5.mlp.down_proj``).

``layer``
    Position in **flattened forward order** -- one total order over sublayer executions in a single
    forward pass. Deliberately not a block index plus a sublayer index: LongcatFlash and HrmText both
    already publish the flattened numbering in their own configs, so a second field would re-derive
    an existing number and give every tensor two spellings (``layer=5, site=1`` and ``layer=11``),
    which is exactly the collision the string form below exists to prevent.

``stream``
    Which parallel residual stream, on a hyper-connection trunk (DeepSeek-V4's mHC). The only
    genuine *tensor axis* among the coordinates: all ``hc_mult`` streams exist simultaneously in a
    ``[B, S, hc_mult, D]`` activation, so nothing can flatten them the way execution order flattens.

Closed rather than an open ``dict[str, int]`` on purpose. An open map lets each family invent its own
axis name, which is the fragmentation :mod:`interp_engine.points` and
``tests/test_vocabulary_boundary.py`` exist to prevent. Extending it stays cheap anyway, because both
the emitter and the parser derive their coordinate list from ``dataclasses.fields``: appending a field
needs no edit here, which is pinned by a test that appends one to a generated copy of the class.

**Why an unknown coordinate raises instead of being ignored.** The usual forward-compatible reflex is
to skip unrecognized fields, and it inverts here, because a coordinate is a *selector* and not an
annotation. A parser that dropped ``site-1`` from ``attn_out.5.site-1`` would return ``attn_out.5``,
which resolves to a real module and yields a real tensor of the right shape -- the caller asked for
one sublayer and silently got another. So :func:`parse_address` raises
:class:`UnknownCoordinate`, which names the coordinate, and callers that want to report version skew
specifically (the vLLM worker) can catch that rather than a generic parse failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any

__all__ = [
    "Address",
    "AddressError",
    "MODULE_PATH_PREFIX",
    "UnknownCoordinate",
    "coordinate_names",
    "format_address",
    "parse_address",
    "to_address",
]

#: Reserved point name introducing a raw dotted module path. Load-bearing rather than decorative:
#: :mod:`interp_engine.points` declares the point set **open**, and an unrecognized name falls
#: through to a module-path lookup, which is currently the only way to tap something the core does
#: not enumerate (a Mamba mixer's state, say). Without a reserved prefix,
#: ``model.layers.5.mlp.down_proj`` has no valid string form at all and that escape hatch quietly
#: stops being addressable.
MODULE_PATH_PREFIX = "path"

# Only RFC 3986 *unreserved* characters appear in an emitted address, so it needs no escaping in a
# URL path or query, no quoting in a shell, and is a legal filename on every OS. `-` separates a
# coordinate from its value rather than `:`, which reads better but is hostile in filenames (the
# the validator writes `f"{point}.{layer}.npy"`) and is the character the legacy `name:layer`
# wire key used -- reusing it would make that key harder to reject cleanly.
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LAYER_RE = re.compile(r"^[0-9]+$")
_COORD_RE = re.compile(r"^([a-z][a-z0-9_]*)-([0-9]+)$")


class AddressError(ValueError):
    """A string that is not a well-formed address."""


class UnknownCoordinate(AddressError):
    """Well formed, but carrying a coordinate this version does not have a field for.

    Separate from :class:`AddressError` because the two call for different responses across a version
    boundary: a malformed key is a bug in the sender, while an unknown coordinate is ordinary skew --
    a client newer than its worker -- and the worker's refusal should say so instead of reporting
    garbage input.
    """


def _is_module_path(name: str) -> bool:
    """Whether ``name`` is a dotted module path rather than a canonical point name."""
    return not _NAME_RE.match(name)


@dataclass(frozen=True, slots=True)
class Address:
    """Which tensor to capture. See the module docstring for what each coordinate means.

    **Field order is append-only, and a test pins it.** ``Address("resid_post", 5, 2)`` means
    ``stream=2``; inserting a field ahead of ``stream`` would silently change what every positional
    construction in the tree already means, and a comment would not survive a refactor.
    """

    name: str
    layer: int | None = None
    stream: int | None = None

    def __post_init__(self) -> None:
        """Reject what could not be emitted injectively, at construction rather than at use.

        Only the four cases where ``str(address)`` would otherwise lose information or produce
        something the parser reads back as a *different* address. Anything else -- an unknown point
        name, an out-of-range layer -- is the resolver's business, not the grammar's.
        """
        if not self.name:
            raise AddressError("an address needs a name")
        if self.name == MODULE_PATH_PREFIX:
            raise AddressError(
                f"{MODULE_PATH_PREFIX!r} is reserved as the dotted-module-path prefix, so it cannot "
                "be a point name; a module path is spelled Address('model.layers.5.mlp.down_proj')"
            )
        if self.name.startswith(f"{MODULE_PATH_PREFIX}."):
            raise AddressError(
                f"pass the module path itself, not its emitted form: Address({self.name[len(MODULE_PATH_PREFIX) + 1 :]!r})"
            )
        for coord in coordinate_names(type(self)):
            value = getattr(self, coord)
            if value is not None and value < 0:
                raise AddressError(f"{coord} must not be negative, got {value}")
        if self.layer is not None and self.layer < 0:
            raise AddressError(f"layer must not be negative, got {self.layer}")
        if _is_module_path(self.name) and (self.layer is not None or self.stream is not None):
            # A module path already names one exact module, so a coordinate beside it has nothing to
            # select -- and `resolve_point`'s fall-through is gated on `layer is None` anyway. Left
            # in the grammar it would be un-emittable, since `path.` takes the rest of the string.
            raise AddressError(
                f"a dotted module path names one module, so it takes no coordinates: {self.name!r} "
                f"with layer={self.layer}, stream={self.stream}"
            )

    def __str__(self) -> str:
        return format_address(self)

    @property
    def is_module_path(self) -> bool:
        """Whether this addresses a raw module path rather than a canonical point."""
        return _is_module_path(self.name)

    def replace(self, **changes: Any) -> Address:
        """A copy with ``changes`` applied. ``dataclasses.replace`` by another name, so callers do
        not have to import it to drop a coordinate."""
        from dataclasses import replace

        return replace(self, **changes)


def coordinate_names(cls: type = Address) -> tuple[str, ...]:
    """The coordinate fields, in order, excluding ``name`` and ``layer``.

    Derived from ``dataclasses.fields`` rather than listed, which is what makes appending a
    coordinate a one-line change: the emitter, the parser and the grammar's vocabulary all read this.
    """
    return tuple(f.name for f in fields(cls)[2:])  # type: ignore[arg-type]


def format_address(address: Any) -> str:
    """The canonical string form. Injective over every address, which a property test pins.

    Unambiguous by construction rather than by care: a name and a coordinate name exclude ``.`` and
    ``-``, while a layer and a coordinate value are digits only, so tokenizing on ``.`` cannot
    mis-split and no two distinct addresses can collide.
    """
    if _is_module_path(address.name):
        return f"{MODULE_PATH_PREFIX}.{address.name}"
    parts = [address.name]
    if address.layer is not None:
        parts.append(str(address.layer))
    parts.extend(
        f"{coord}-{value}"
        for coord in coordinate_names(type(address))
        if (value := getattr(address, coord)) is not None
    )
    return ".".join(parts)


def parse_address(text: str, cls: type = Address) -> Any:
    """Parse the canonical form. Total, and strict: anything else raises.

    Strict on purpose. The hazard this replaces is ``key.rsplit(":", 1)``, which is *lenient* -- it
    turns an address carrying an extra coordinate into a valid-looking wrong key rather than an
    error. A lenient split is not safe as a wire format; a total parser with an explicit rejection
    path is. In particular the legacy ``"resid_post:5"`` key raises, rather than parsing as a point
    named ``resid_post:5``.
    """
    if not isinstance(text, str):
        raise AddressError(f"an address is a string, got {type(text).__name__}")
    if not text:
        raise AddressError("an address cannot be empty")
    if text.startswith(f"{MODULE_PATH_PREFIX}."):
        path = text[len(MODULE_PATH_PREFIX) + 1 :]
        if not path:
            raise AddressError(f"{text!r} has the module-path prefix but no path after it")
        return cls(path)

    head, *rest = text.split(".")
    if not _NAME_RE.match(head):
        hint = (
            " (a dotted module path needs the 'path.' prefix)"
            if "." in text
            else " (the legacy 'name:layer' form is no longer accepted)"
            if ":" in text
            else ""
        )
        raise AddressError(f"{text!r} does not start with a point name matching [a-z][a-z0-9_]*{hint}")
    if head == MODULE_PATH_PREFIX:
        raise AddressError(f"{text!r} uses the reserved name {MODULE_PATH_PREFIX!r} as a point")

    layer: int | None = None
    if rest and _LAYER_RE.match(rest[0]):
        layer = int(rest[0])
        rest = rest[1:]

    known = coordinate_names(cls)
    coords: dict[str, int] = {}
    for token in rest:
        match = _COORD_RE.match(token)
        if match is None:
            raise AddressError(f"{text!r}: {token!r} is not a coordinate of the form 'name-0'")
        coord, value = match.group(1), int(match.group(2))
        if coord not in known:
            raise UnknownCoordinate(
                f"address {text!r} uses coordinate {coord!r}, which this version does not support "
                f"(known: {', '.join(known) or 'none'})"
            )
        if coord in coords:
            raise AddressError(f"{text!r} repeats coordinate {coord!r}")
        coords[coord] = value

    return cls(head, layer, **coords)


def to_address(value: Any) -> Address:
    """Coerce what callers actually pass into an :class:`Address`.

    Three shapes, because the migration off ``tuple[str, int]`` is not worth breaking every call site
    over: an ``Address`` passes through, a string is *parsed* (so ``"resid_post.5"`` works and
    ``"embeddings"`` still means the global point it always did), and a tuple is read positionally in
    field order. The tuple case carries **every** element rather than the first two -- the truncation
    it replaces is the specific bug that motivated this type.
    """
    if isinstance(value, Address):
        return value
    if isinstance(value, str):
        return parse_address(value)
    if isinstance(value, tuple | list):
        if not value:
            raise AddressError("an address tuple needs at least a name")
        width = 1 + len(coordinate_names())
        if len(value) > width + 1:
            raise AddressError(
                f"an address has at most {width + 1} coordinates "
                f"({', '.join(f.name for f in fields(Address))}), got {len(value)}: {value!r}"
            )
        return Address(*value)
    raise AddressError(f"cannot read an address from {type(value).__name__}: {value!r}")
