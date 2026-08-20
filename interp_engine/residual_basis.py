"""What a model's residual stream *is*, so points can refuse on the invariant instead of the family.

The residual points (``resid_pre``, ``resid_mid``, ``resid_post``) do not merely name a tensor. They
carry an implicit claim about its structure -- that there is **one** stream, that each sublayer's
contribution enters it by **addition**, and that the sublayers are **ordered** along it -- and
essentially every downstream use depends on that claim rather than on the tensor. A logit lens
decodes the stream as if it were a partial sum. A steering vector is added on the assumption that
addition is what the model does there. An SAE is trained on one ``d_model`` vector per position.

Two rules this module exists to enforce, mirroring :mod:`interp_engine.autograd_support`.

**The basis never gates loading.** A model whose trunk breaks one of these invariants still loads,
still runs, and still has every non-residual point. Both backends compute the verdict lazily on
first access to ``model.residual_basis``, from config values already read -- no kernel, no forward.

**And no silent degradation.** Where an invariant fails, the affected point raises
:class:`ResidualBasisUnsupported` naming the invariant and the addressable alternative, rather than
returning the nearest plausible tensor. This is the whole point: on a hyper-connection trunk the
block's output *is* hookable and *is* shaped ``d_model`` in its last axis, so a consumer that
broadcasts over the extra one gets a confident answer to a question it never asked.

Three invariants, tracked separately because they fail on different families and cost different
things:

``n_streams``
    How many residual streams the trunk carries at once. DeepSeek-V4's hyper-connections carry
    ``hc_mult = 4`` of them as ``(batch, seq, streams, d_model)`` the whole way down. They are
    simultaneous rather than sequential, so nothing can flatten them: the fix is to address one,
    which is what ``Address(..., stream=k)`` is for.

``additive``
    Whether a sublayer's contribution enters the residual by plain addition, so ``resid_post`` is
    ``resid_pre`` plus the contributions and a lens sees a partial sum. False under
    hyper-connections, which collapse the streams with learned per-token weights before the sublayer
    and scatter the output back across them after -- the residual is a *learned mixture*, and
    "``resid_post`` minus ``resid_pre`` is what this block wrote" stops being true even after a
    stream is selected.

``sequential``
    Whether the sublayers are ordered along the residual, so a residual exists *between* them. False
    on ``parallel_attn_mlp`` (GPT-J, NeoX, Falcon, PaLM-style blocks), where attention and the MLP
    both read the same input and only their sum is added.

Deliberately **not** collapsed into one flag. A parallel block is additive and single-stream, so its
``resid_pre``/``resid_post`` are ordinary residuals a lens reads correctly -- only ``resid_mid`` is
undefined. Folding it into ``additive`` would refuse the logit lens on GPT-J, which works fine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from interp_engine.points import stream_stack_points

# The residual points, whose names encode the single-additive-stream claim. `attn_out`/`mlp_out` are
# deliberately absent: they name a sublayer's own output, which exists and is `d_model`-wide however
# the trunk then carries it, and so stay addressable on every family here.
RESIDUAL_POINTS: frozenset[str] = frozenset({"resid_pre", "resid_mid", "resid_post"})

# How a `(..., streams, d_model)` capture may be collapsed to the one `d_model` vector a read-out
# decodes. `none` is not a degenerate member: it is the only valid value on a conventional trunk,
# where the capture is already that vector, and naming it explicitly is what lets a caller pass the
# reduction through unconditionally instead of branching on the stream count at every call site.
#
# `mean` and `sum` differ by a factor of `n_streams`, which a fitted lens absorbs (scaling source and
# target alike leaves the Jacobian unchanged) and a final norm removes before an unembed. They are
# still separate members, because nothing here knows that the consumer is a fitted lens.
STREAM_REDUCTIONS: frozenset[str] = frozenset({"none", "mean", "sum", "select"})


class ResidualBasisUnsupported(ValueError):
    """A residual point was requested on a trunk whose structure does not support it.

    A ``ValueError`` rather than a ``RuntimeError`` (which is what the gradient gate raises) because
    every refusal it replaces was already a ``ValueError`` on the same code path, and callers -- in
    this repo and in the validator -- catch it as one.
    """


@dataclass(frozen=True)
class ResidualBasis:
    """How this model's residual stream is structured, and what that rules out."""

    n_streams: int = 1
    """Residual streams carried simultaneously. 1 on a conventional trunk."""

    additive: bool = True
    """Whether sublayer contributions enter the residual by addition rather than a learned mixture."""

    sequential: bool = True
    """Whether the sublayers are ordered along the residual, so an intermediate residual exists."""

    lens_valid: bool = True
    """Whether a ``d_model`` read-out (logit lens, SAE, steering vector) applies to a residual point
    as captured. False whenever a single capture is not one stream's partial sum."""

    stream_addressable: bool = False
    """Whether a specific stream can be requested with ``Address(..., stream=k)`` on this backend.
    Multi-stream and *not* addressable is the honest verdict on vLLM, whose wire key has no
    coordinate for it yet -- the model is no different, the transport is."""

    blockers: tuple[str, ...] = ()
    """Human-readable reasons an invariant fails, in the order they were found."""

    caveats: tuple[str, ...] = ()
    """Things that do not block a residual point but change what it means. Separate from
    ``blockers`` because a caveat must never turn into a refusal: the request is honored."""

    architecture: str = ""
    """The architecture this verdict is about, for error messages."""

    backend: str = ""
    """Which backend produced this verdict, for error messages."""

    remedy: str = ""
    """What the caller can do about it, phrased for this backend."""

    @property
    def single_additive_stream(self) -> bool:
        """The claim the residual point *names*: one stream, entered by addition."""
        return self.n_streams == 1 and self.additive

    def _refuse(self, detail: str) -> ResidualBasisUnsupported:
        where = f"{self.architecture} " if self.architecture else ""
        return ResidualBasisUnsupported(f"{where}{detail} {self.remedy}".strip())

    def require_single_stream(self, name: str, *, stream: int | None = None) -> None:
        """Gate an unqualified residual point on there being one stream to mean.

        Passing ``stream`` records that the caller *did* name one, which is the whole difference
        between "this model has no such tensor" and "say which": the first is a dead end and the
        second is a one-word fix, so they must not share an error message.
        """
        if self.n_streams == 1:
            if stream is not None:
                raise self._refuse(
                    f"carries a single residual stream, so {name!r} has no stream axis and "
                    f"stream={stream} selects nothing. Drop the coordinate; the unqualified point "
                    "is the whole tensor."
                )
            return

        if stream is None:
            raise self._refuse(
                f"carries {self.n_streams} parallel residual streams (hyper-connections), so "
                f"{name!r} does not name a single one: the trunk's activations are shaped "
                f"(batch, seq, {self.n_streams}, d_model), each block collapsing the streams with "
                "learned weights before its sublayer and mixing the result back across them after."
            )
        if not 0 <= stream < self.n_streams:
            raise self._refuse(
                f"carries {self.n_streams} residual streams, so stream={stream} is out of range "
                f"for {name!r} (valid: 0..{self.n_streams - 1})."
            )
        if not self.stream_addressable:
            raise self._refuse(
                f"carries {self.n_streams} residual streams, but this backend cannot address one: "
                f"{'; '.join(self.blockers) or 'no reason recorded'}."
            )

    def require_stream_coordinate(self, name: str, stream: int | None = None) -> None:
        """Decide whether ``name`` on this trunk may carry a stream, and refuse with the reason.

        Two questions, in order, because their answers are different kinds of thing. First, does
        this *point* have a stream axis at all -- a vocabulary question, and the answer is no even on
        a hyper-connection trunk for a point like ``attn_out``, which is that block's own
        ``d_model``-wide output before it is scattered. Then, does this *model* carry the streams the
        coordinate selects among, which is :meth:`require_single_stream`.

        The reason this lives on the basis rather than on either backend is that the second question
        must be asked when the caller named **no** stream too. A residual point is a claim about the
        trunk whether or not a coordinate qualifies it, so gating only on ``stream is not None`` lets
        an unqualified ``resid_post`` through on a hyper-connection trunk -- and what comes back is a
        ``[tokens, n_streams, d_model]`` stack whose last axis is ``d_model`` either way, so no width
        check downstream can catch it. Both backends call this so neither can drift into that.
        """
        if stream is not None and name not in RESIDUAL_POINTS:
            raise ValueError(
                f"{name!r} does not carry residual streams separately, so stream={stream} selects "
                f"nothing. Only {sorted(RESIDUAL_POINTS)} take a stream coordinate; a sublayer's "
                "own output is d_model-wide before it is scattered across the streams."
            )
        if stream is not None or name in RESIDUAL_POINTS:
            self.require_single_stream(name, stream=stream)

    def require_hyper_connections(self, name: str) -> None:
        """Gate a point that only exists on a trunk carrying more than one residual stream.

        The mirror of :meth:`require_single_stream`, and needed for the same reason in the opposite
        direction: ``mlp_stream_mix`` is not a point a Llama declines to serve, it is a point a Llama
        does not have -- there is no mixture to name where contributions are added. The eager backend
        gets this from :func:`interp_engine.points.point_spec`, which is passed the stream count and
        returns nothing for a conditional row on a single-stream trunk. vLLM's hook path checks
        against the point *table* instead, which is unconditional, so without this a Llama request
        would reach the worker and be refused there -- correctly, but several frames into an RPC
        rather than in the caller's own stack frame.
        """
        if self.n_streams > 1:
            return
        raise self._refuse(
            f"carries a single residual stream, so {name!r} does not exist on it: there are no "
            "per-stream write weights and no mixing matrix, because a sublayer's contribution enters "
            "one stream by addition. This point is defined only on a hyper-connection trunk."
        )

    def stacked_at(self, point: str) -> bool:
        """Whether a capture at ``point`` on this trunk carries the stream axis.

        Two facts, and neither implies the other. The trunk decides whether the streams exist; the
        POINT decides whether the tensor there is one of them or all of them, and on a
        hyper-connection model both kinds sit side by side -- ``attn_out`` is that sublayer's own
        ``d_model``-wide output, ``mlp_stream_collapse`` is the ``d_model`` vector the MLP read, and
        ``resid_streams`` is the whole stack. All three end in ``d_model``, so no shape check
        distinguishes them; hence a table lookup rather than an inspection.
        """
        return self.n_streams > 1 and point in stream_stack_points()

    def require_stream_reduction(self, reduce: str, index: int | None = None, *, point: str | None = None) -> None:
        """Gate a read-out's declared stream reduction on the tensor it will run against.

        The other half of :meth:`require_lens`, and what makes that refusal answerable rather than
        final. What ``require_lens`` refuses is decoding a *stack* as if it were a stream; a declared
        reduction supplies the missing fact -- which ``d_model`` vector the stack stands for -- so the
        read-out is a read-out again instead of a broadcast over an axis nobody asked about.

        Required exactly where the capture is stacked and forbidden everywhere else, in both
        directions, because a mismatch is not a shape error at the reduction site. ``'none'`` on a
        stack leaves an extra axis while the last one is still ``d_model``, and ``'mean'`` on a
        ``(batch, seq, d_model)`` capture averages the SEQUENCE and returns one vector per prompt --
        both believable, neither what was asked for. :func:`reduce_streams` catches the second by
        shape when ``n_streams`` is passed; nothing downstream catches the first.

        ``point`` names the capture, and omitting it asks about the residual points, which are the
        ones that carry the stack. Pass it when it is known: without it, a lens fitted on ``attn_out``
        of a hyper-connection model is told to declare a reduction for an axis its tensor lacks.
        """
        if reduce not in STREAM_REDUCTIONS:
            raise ValueError(f"unknown stream reduction {reduce!r} (expected one of {sorted(STREAM_REDUCTIONS)})")
        if not (self.n_streams > 1 and (point is None or self.stacked_at(point))):
            where = f"{point!r} " if point is not None else ""
            if reduce != "none":
                raise self._refuse(
                    f"has no stream axis at {where}to reduce with {reduce!r}: the capture there is "
                    "already one d_model vector per position. Declare 'none'."
                )
            if index is not None:
                raise self._refuse(f"has no stream axis at {where}for stream index {index} to select.")
            return
        if reduce == "none":
            raise self._refuse(
                f"carries {self.n_streams} parallel residual streams, so a capture is a stack and a "
                "read-out has to say which d_model vector it decodes. Declare a reduction "
                f"({sorted(STREAM_REDUCTIONS - {'none'})}); a fitted lens carries the one it was "
                "fitted on."
            )
        if reduce == "select":
            if index is None:
                raise ValueError("stream reduction 'select' needs a stream index")
            if not 0 <= index < self.n_streams:
                raise self._refuse(
                    f"carries {self.n_streams} residual streams, so stream index {index} is out of "
                    f"range (valid: 0..{self.n_streams - 1})."
                )
        elif index is not None:
            raise ValueError(f"stream index {index} is meaningless with stream reduction {reduce!r}")

    def reduce_streams(
        self, tensor: torch.Tensor, reduce: str, index: int | None = None, *, point: str | None = None
    ) -> torch.Tensor:
        """Apply ``reduce`` to a capture from this trunk, checked against it first."""
        self.require_stream_reduction(reduce, index, point=point)
        return reduce_streams(tensor, reduce, index=index, n_streams=self.n_streams)

    def require_sequential(self, name: str) -> None:
        """Gate a *between-sublayers* residual on the sublayers being sequenced at all.

        Carries its own remedy rather than the instance's: ``remedy`` answers the multi-stream
        blocker, and pointing a parallel-block caller at a stream coordinate would be nonsense.
        """
        if self.sequential:
            return
        raise self._refuse(
            "runs attention and the MLP in parallel on the same input, so there is no residual "
            f"between them: {name!r} is undefined here. Use 'resid_pre' (what both sublayers read) "
            "or 'resid_post' (after both are added)."
        )

    def require_lens(self, what: str = "a residual read-out") -> None:
        """Gate a ``d_model`` read-out (logit lens, tuned lens) on the capture being one stream.

        The gate the lens path calls, so the text is identical however the request arrived. Named
        for the read-out rather than the point, because what fails is the interpretation.
        """
        if self.lens_valid:
            return
        detail = "; ".join(self.blockers) or "no reason recorded"
        raise self._refuse(f"cannot support {what}: {detail}.")

    def select_stream(self, tensor: torch.Tensor, stream: int) -> torch.Tensor:
        """Take one residual stream out of a ``(..., streams, d_model)`` capture.

        The stream axis is second-from-last, so the slice leaves the shape every consumer already
        expects and ``capture()``'s ``[tokens, d_model]`` contract holds unchanged.

        Checked rather than assumed, and this is the whole reason it is one function instead of an
        inline index: indexing an axis that is *not* the stream axis succeeds just as readily and
        returns a tensor of an entirely believable shape. Read and write must agree on which axis
        that is, or a steering vector lands somewhere a capture would never show it.
        """
        if tensor.ndim < 3 or tensor.shape[-2] != self.n_streams:
            raise self._refuse(
                f"has {self.n_streams} residual streams, but the tensor here is "
                f"{tuple(tensor.shape)} -- no axis of {self.n_streams} second-from-last to take "
                f"stream={stream} from. This point does not carry the streams separately; drop the "
                "coordinate to use it whole."
            )
        return tensor[..., stream, :]

    def replace_stream(self, tensor: torch.Tensor, stream: int, value: torch.Tensor) -> torch.Tensor:
        """``tensor`` with one stream replaced -- the write half of :meth:`select_stream`.

        Out of place, because this runs in a forward hook on a tensor the model still holds: an
        in-place write would also hit whatever else aliases that storage.
        """
        self.select_stream(tensor, stream)
        out = tensor.clone()
        out[..., stream, :] = value
        return out

    def describe(self) -> dict[str, object]:
        """JSON-friendly form, for a ``/capabilities`` response."""
        return {
            "backend": self.backend,
            "architecture": self.architecture,
            "n_streams": self.n_streams,
            "additive": self.additive,
            "sequential": self.sequential,
            "lens_valid": self.lens_valid,
            "stream_addressable": self.stream_addressable,
            "blockers": list(self.blockers),
            "caveats": list(self.caveats),
            "remedy": self.remedy,
        }


def reduce_streams(
    tensor: torch.Tensor,
    reduce: str,
    *,
    index: int | None = None,
    n_streams: int | None = None,
) -> torch.Tensor:
    """Collapse a ``(..., streams, d_model)`` capture to the ``(..., d_model)`` vector a lens decodes.

    :meth:`ResidualBasis.select_stream` for the case where the read is not one stream but a function
    of all of them. Which function is an argument rather than a choice made here, and that is the
    whole point: the mean, the sum and any single stream all come back the same shape with the same
    ``d_model``, so a lens fitted against one of them cannot be told from a lens fitted against
    another by anything downstream. The artifact says which; this applies it.

    A free function as well as a method because the vLLM worker applies it, and the worker has the
    tensor and the spec but no :class:`ResidualBasis` -- that verdict is computed on the client and
    would have to cross ``collective_rpc`` to get here. ``n_streams``, when passed, is checked against
    the axis, guarding what :meth:`select_stream` guards for the same reason: reducing an axis that is
    not the stream axis succeeds and returns a tensor of an entirely believable shape.

    Whether the reduction is coherent with the trunk at all is :meth:`require_stream_reduction`'s
    question, asked once where the request is validated. This function only refuses what it can see.
    """
    if reduce not in STREAM_REDUCTIONS:
        raise ValueError(f"unknown stream reduction {reduce!r} (expected one of {sorted(STREAM_REDUCTIONS)})")
    if reduce == "none":
        return tensor
    if tensor.ndim < 3:
        raise ValueError(
            f"stream reduction {reduce!r} needs a stream axis, but the tensor here is "
            f"{tuple(tensor.shape)} -- there is nothing second-from-last to reduce, and reducing "
            "what is there would collapse positions instead."
        )
    if n_streams is not None and tensor.shape[-2] != n_streams:
        raise ValueError(
            f"stream reduction {reduce!r} expects {n_streams} streams second-from-last, but the "
            f"tensor here is {tuple(tensor.shape)}. This point does not carry the streams separately."
        )
    if reduce == "mean":
        return tensor.mean(dim=-2)
    if reduce == "sum":
        return tensor.sum(dim=-2)
    if index is None:
        raise ValueError("stream reduction 'select' needs a stream index")
    if not 0 <= index < tensor.shape[-2]:
        raise ValueError(f"stream index {index} is out of range for {tensor.shape[-2]} streams")
    return tensor[..., index, :]


def _shared_verdict(
    *,
    n_streams: int,
    parallel_attn_mlp: bool,
    architecture: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Blockers and caveats that depend only on the model, not on how it is being run."""
    blockers: list[str] = []
    caveats: list[str] = []

    if n_streams > 1:
        blockers.append(
            f"{architecture or 'this architecture'} carries {n_streams} residual streams "
            "(hyper-connections), so a capture at a residual point is a stack rather than a stream"
        )
        caveats.append(
            "each block collapses the streams with learned per-token weights before its sublayer "
            "and scatters the output back across them after, so even one selected stream is not a "
            "running sum of sublayer outputs -- 'resid_post - resid_pre' is not what this block wrote"
        )
    if parallel_attn_mlp:
        caveats.append(
            "attention and the MLP run in parallel on the same input, so no residual exists between "
            "them: 'resid_mid' is undefined, while 'resid_pre' and 'resid_post' are ordinary residuals"
        )
    return tuple(blockers), tuple(caveats)


def eager_residual_basis(
    *,
    n_residual_streams: int = 1,
    parallel_attn_mlp: bool = False,
    architecture: str = "",
) -> ResidualBasis:
    """Verdict for :class:`~interp_engine.model.EagerModel`.

    Eager holds the real module tree and slices in-process, so a stream is addressable whenever the
    model has one: the only thing that makes ``lens_valid`` False here is the model itself.
    """
    blockers, caveats = _shared_verdict(
        n_streams=n_residual_streams, parallel_attn_mlp=parallel_attn_mlp, architecture=architecture
    )
    multi = n_residual_streams > 1
    return ResidualBasis(
        n_streams=n_residual_streams,
        additive=not multi,
        sequential=not parallel_attn_mlp,
        lens_valid=not multi,
        stream_addressable=multi,
        blockers=blockers,
        caveats=caveats,
        architecture=architecture,
        backend="eager",
        remedy=(
            "Name a stream -- Address('resid_post', 5, stream=0) -- or use 'attn_out'/'mlp_out', "
            "which are that block's own d_model-wide output before it is scattered."
            if multi
            else ""
        ),
    )


def vllm_residual_basis(
    *,
    n_residual_streams: int = 1,
    parallel_attn_mlp: bool = False,
    architecture: str = "",
) -> ResidualBasis:
    """Verdict for :class:`~interp_engine.vllm_backend.VLLMModel`.

    Identical to the eager verdict except that a stream cannot be *asked for*. The reason is no longer
    the transport: the wire grammar does carry the coordinate (``resid_post.7.stream-2`` round-trips
    through ``parse_address``, ``hook_site`` drops it so two streams of one point share a hook, and
    ``_payload.select_stream`` slices it). What is missing is a residual hook that means anything on
    such a trunk. ``_mk_resid_post_hook`` reconstructs the stream as ``output[0] + output[1]``, which
    is vLLM's usual ``(hidden, residual)`` convention; a DeepSeek-V4 layer returns
    ``(x, residual, post_mix, res_mix)`` where ``output[0]`` is ``d_model``-wide and ``output[1]`` is
    the ``(tokens, hc_mult, d_model)`` stack, so that sum is not the residual at either shape.
    Measured on vLLM 0.26.0: it raises a shape error from inside the worker's forward for any prompt
    whose token count differs from ``hc_mult``, and for a prompt of exactly ``hc_mult`` tokens it
    succeeds and returns ``x`` added to every stream -- a wrong tensor that passes both
    ``_assert_full_width_captured`` (its last axis is ``d_model``) and
    ``_assert_full_prompt_captured`` (its first is the token count).

    So the coordinate stays unaddressable *on the residual points*, and the blocker says which half is
    missing, because the two halves have different fixes: the transport needs nothing, and the hook
    needs a V4-specific residual reconstruction.

    **Not simply "read ``output:1`` instead of summing", which is the fix this docstring used to
    suggest.** That element is a stream stack of the right shape and the wrong sublayer: vLLM defers
    each sublayer's write into the next sublayer's kernel, so it is the stack the MLP read --
    ``resid_mid`` in stream form -- while ``resid_post`` is the block's output, which is formed inside
    the *next* layer and crosses no boundary. Measured on DeepSeek-V4-Flash at layers 0/21/42; see
    ``vllm_capture._tree.LAYER_RETURN_INDEX``.

    What the backend does instead is serve the stack under its own name: ``resid_streams`` is captured
    off the next layer's own kernel, where the block's write actually lands (and off the model's
    closing ``mhc_post`` call for the last layer). So there *is* a route to one stream under vLLM --
    capture ``resid_streams`` and index it, or capture ``attn_stream_collapse``/``mlp_stream_collapse``
    for the ``d_model`` vector the sublayer reads -- and the remedy names it. ``stream_addressable``
    stays False all the same, because it answers a narrower question: whether ``Address('resid_post',
    L, stream=k)`` works, and it does not. Those are different points with different meanings on this
    trunk, and conflating them is what the whole module exists to prevent.
    """
    blockers, caveats = _shared_verdict(
        n_streams=n_residual_streams, parallel_attn_mlp=parallel_attn_mlp, architecture=architecture
    )
    multi = n_residual_streams > 1
    if multi:
        blockers = (
            *blockers,
            "the vllm capture wire key does carry a stream coordinate, but no residual hook reads "
            "this trunk: the resid_post hook sums the layer's first two returns, which on a "
            "hyper-connection block adds a d_model tensor to the whole stream stack -- and the stack "
            "it would add to is the one the mlp read rather than the block's output, since vllm "
            "defers each sublayer's write into the next sublayer's kernel. The stack itself is served "
            "under its own name, resid_streams, which is where the block's output is read from",
        )
    return ResidualBasis(
        n_streams=n_residual_streams,
        additive=not multi,
        sequential=not parallel_attn_mlp,
        lens_valid=not multi,
        stream_addressable=False,
        blockers=blockers,
        caveats=caveats,
        architecture=architecture,
        backend="vllm",
        remedy=(
            "Capture 'resid_streams' and index the stream axis, which this backend does serve; or "
            "'attn_stream_collapse'/'mlp_stream_collapse' for the d_model vector the sublayer reads, "
            "and 'attn_out'/'mlp_out' for its output. Load this model on the eager backend to address "
            "a stream through the residual point itself."
            if multi
            else ""
        ),
    )
