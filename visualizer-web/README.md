# Interp Engine Visualizer

A single page that shows where every `interp-engine` hook point sits in a transformer forward
pass, and what TransformerLens and nnterp call the same tensor.

The diagram runs left to right — embeddings, then each block, then the final norm and the unembed —
with the residual stream as a horizontal spine, attention branching above it and the MLP below.
Every point is a mark on that path, labelled with whichever stack's name the **Names** toggle has
selected. Hovering one explains what the tensor is, how it is derived in terms of the points before
it, and which of the active traits shaped that derivation. Clicking one **pins** that card: it stays
on the point until you click a different point or click away, so the snippet in it can be selected
and copied without the pointer having to stay inside a card that is sometimes a page tall. A pinned
point wears a second, wider ring, because a card that stops following the pointer and looks exactly
like one that does reads as the diagram having frozen. While a pin is held the pointer is inert —
that is what a pin is — and releasing it does not hand the card to whatever the pointer drifted
over in the meantime. A pin does not outlive the drawing it was taken on: change a trait, and the
point it describes may not exist any more. With **interp-engine** selected it ends
with the call that reads the point, against a checkpoint of the architecture the pane is drawing,
over a tab per way of making it: `vllm`, `eager`, `vllm async`. The first two are the _same snippet_
— since engine 1.1 the sync free functions dispatch on the model, so the backend is one argument and
nothing else changes, which is the thing worth showing and which two side-by-side snippets hid. The
third is the async method form a server holding the model in its own event loop writes. Only under
that name: the code is this engine's API, and printing it beside a TransformerLens or nnterp label
would answer a question asked in one vocabulary with a snippet written in another. Where vLLM has no
path to the point, its tabs say so up front and give the engine's own words, which distinguish
"unimplemented" from "unreachable" — the difference between filing a bug and switching backend.
Beside **Copy** under those tabs is **Notebook**, which opens the selected tab's snippet in a Colab
notebook with the engine installed — awaited on the vLLM tabs, since a notebook cell runs inside an
event loop. See [Running a snippet](#running-a-snippet).

The card opens to the right of the point and level with it — the side the diagram already flows
towards, and the same side every time, so it is never above one point and below the next. It flips
left only when there is no room, and it never covers the point it describes or the ones stacked with
it. Below `sm` none of that survives: a card wide enough for a code snippet is most of a phone, so
there it narrows to 320px and opens **under** the point, flipping above when the bottom edge is
closer. Vertical is the one axis a phone has to spare.

It is positioned in the **window**, in a layer over everything, rather than inside the pane whose
point it describes. In compare mode a pane is half the screen tall, and a card drawn inside one was
clipped by that pane's scroll box — so the tallest cards, which are the ones with the most to say,
were cut off at the fold with half the window empty below them. The card is not part of either
drawing; it is about a point in one of them, and it uses the window the way a menu does.

**Architecture**, in a band above the diagram, picks a family and sets its traits. These are
architectures, not checkpoints: two Llama models differ in size, not in where their hook points are.
The list is alphabetical, because it is a name being looked up rather than a taxonomy being read.
The band is the same one in both modes, so the picker is always attached to the drawing it names
rather than sitting in the header until a second diagram appears and then moving.

Editing the toggles by hand until they match a family snaps the name back to that family, which
leaves the case where two families are wired identically in this vocabulary — Llama and Qwen2.5,
Gemma 3 and Gemma 4. There the last explicit pick wins, for as long as its traits still hold, so
choosing the second of a pair does not silently rename itself to the first. Two presets sharing a
trait set is therefore allowed, and it is worth reading as a to-do: it means the thing that
actually separates them is something the diagram cannot yet draw.

The **timeline** beside it is the same choice in release order, and the two move together. The list
answers "which family"; the timeline answers "what came after what", which is most of why these
architectures differ at all — QK-norm and MoE and sandwich norms each swept the field in a
particular year.

Its stops are evenly spaced rather than positioned by date. A true time axis is the more honest
picture and the worse control: three quarters of these families shipped inside an eighteen-month
window, so their stops landed a few pixels apart — two of them exactly on top of each other — while
GPT-2 sat alone at the far left with four empty years beside it. Even spacing makes every family the
same size target, and the ordering, which is the part worth seeing, still reads left to right. The
dates are on the thumb's label and in the picker's caption, where they can be read exactly instead
of estimated from a position.

The stops are not drawn. Twenty-five of them is a tick every 12px at full width and every 7px on a
phone, which at that pitch reads as a hatched track rather than as countable stops, and what they were
there to say — that the control is discrete — the thumb says by snapping. The thumb itself is upright
and taller than the track, like a scrubber's handle: as a dot the height of the track it read as one
more stop on it.

Dates are `released` in `data/architectures.ts`, to the month, and record when the architecture
class first shipped rather than when the newest checkpoint using it did — the picker's caption reads
them back as `RELEASED APR 2025`. They order a timeline, they do not date a release. The caption is
dropped below `sm`, where a second line of text on the narrowest control costs more than it says: the
date is on the timeline's own thumb, beside it.

Where the timeline does not fit beside the picker it goes **under** it rather than disappearing. It
used to hide at those sizes, which meant a phone got the list and no sense of order at all, and order
is most of the point. In single mode it is beside the picker at every width, taking whatever the row
has left on a phone — about 150px, which is enough to drag. Comparing, where it lands is not monotonic
in width, because what it competes with is not either: beside the picker on a phone, where the pills
have moved out from under it and the picker has given back its caption width, then under it from `sm`,
where the pills are alongside again and 228px will not fit next to them, then beside it once more at
`xl`.

The years at the two ends of the track are also dropped below `sm`. They are the least of what the
control says — the thumb carries the family and its exact date, and the stops are evenly spaced, so
the ends are not even reading a scale — and on a 150px track two more numbers under it read as
clutter. Without them the control is two rows instead of three, which is what lets it match the height
of the picker beside it.

The **search** button to the left of the picker replaces the list with a field, for when you already
know the name. It matches on the family label, the example checkpoints and the architecture class, so
`gemma-3`, `Qwen3-Next` and `GptOssForCausalLM` all find their family — the first two because that is
how these are known outside this app, the third because it is what someone arriving from a
`config.json` has in hand. Typing selects the first match as it goes, so the diagram follows the
field and three keystrokes usually land it, and the list stays open below with the rest of the
matches: a field that only took an exact name would be worse than the dropdown it replaced. That
running selection is provisional and search stays open. Committing one — Enter, or a click on a row —
hands the band back to the dropdown, since at that point you have what you came for and a text field
is the wrong thing to leave in front of you.

**Why the family matters** is two sentences in `significance` — what it changed, and what that
bought — printed beside the timeline when there is one diagram, and always available from the
circled i beside the picker, and from every row of the open list. Prose about the model, where `note`
is prose about the drawing.

It answers the question the diagram cannot. A hook point map shows that DeepSeek V3 has no `v_proj`
and Gemma 2 wraps its sublayers in norms; it cannot say that the first cut the KV cache tenfold or
that the second is why small-model distillation started working. Without that, the timeline is a
sequence of arbitrary rewirings rather than a record of people solving problems, and the ordering
stops being interesting.

Each opens with the family's own name, so the paragraph reads as a statement about Gemma 3 rather
than as an unlabelled caption whose subject you have to infer from whatever the picker last said.

Two placements because the space is not always there. Beside the timeline it is hidden below `lg`, and
in compare mode the diff pills own that space and it does not appear at all — but where it does
appear it is printed in full. It was clamped to three lines, and the ellipsis fell in the middle of
the second sentence every time: the half that says what the change bought, which is the half the
prose is there for. The width cap is what keeps the band from eating the canvas instead.

The two never appear together. At `lg` in single mode the paragraph is on screen, so the circled i
beside the picker goes away rather than offering a card that repeats the prose beside it; it is back
below `lg`, and at every width in compare mode, where that space is the diff's. The icon is the copy
that carries this wherever the printed one does not fit.

Putting one on every row of the open list is separate, and unconditional — the question "which of
these do I want" is exactly the one the labels cannot answer.

It sits **outside** the picker, to its right. Inside the trigger it was unreachable by touch: the tap
landed on the trigger and opened the list, so on a phone the only always-present copy of this prose
was the one you could not open. Outside it is its own target, and it is a popover driven by hand
rather than a hover card — a hover card never opens on touch. Mouse keeps hover, every other pointer
gets a tap, and the two are told apart by `pointerType` rather than by a viewport width, since a
hybrid laptop is both and a width does not say which one is in your hand.

Its card opens **below** the icon beside the picker and to the **right** of a list row, because the
card anchors on the icon rather than on the control. Below is the only side that is neither
off-screen nor on top of the picker, since the band is at the top of the window with the diagram
underneath. Beside a row, an icon on the left would open the card across the list it is helping you
read, so the icon sits at the edge the card comes out of, and the card clears the list entirely.

The header keeps only two things: the wordmark, and **Single Mode | Compare Mode** on the left. A
toggle rather than a button that swaps for a different button — the mode is legible from either
state, entering and leaving are the same control, and the header's left edge keeps its width, so
nothing shifts under the pointer when the mode changes.

Two words of the caption under the wordmark are hoverable, and each opens the evidence for its own
claim. **Fast** gives the throughput table — one machine's tok/s on four checkpoints across three
backends, eager against hooked vLLM against vLLM with graph static, each cell one stream over eight
concurrent — with the GPU, the dtype and the two lengths printed above it, because a rate without
them is not a number anyone can hold their own run against. A column is a backend and a cell carries
both concurrency regimes, rather than the reverse, so that every multiplier can be divided out
against an eager figure that is actually on screen. **Standardized**
gives every point against both backends, scrolling, from the same `data/points.ts` the diagram is
drawn from, so a check here and a refusal on a card cannot disagree. Both are claims a reader is
entitled to be sceptical about, and until now the numbers behind them were only in the repo.

**Served or not served, and nothing in between**, though `data/points.ts` has three states. It
separates a point vLLM serves from its worker hooks from one it rebuilds off-kernel from captured
q/k, and the point's own card on the diagram says which and quotes the engine's own reason — but that
is a fact about how the engine gets there, not about whether you can have the tensor. What this table
is asked is what the backend serves, and a recompute is served. The third mark it used to carry spent
a legend entry and an amber glyph putting an asterisk on a claim that does not have one.

They are hover cards rather than the hand-driven popover the circled i beside the picker is, for the
reason the trait pills are too: the caption is hidden below `lg`, so no pointer that cannot hover
ever reaches the trigger — and the support table has to be scrolled, which needs the card to stay
open while the pointer crosses into it. Neither claim is a link. Each opens on focus as well as on
hover, and the card is the whole of what it has to say; a claim that also navigated would take the
reader off the diagram to a document that says the same thing at length.

Neither card is written in the caption's own component. Both live in `components/evidence/`, because
the welcome dialog makes the same two claims at a width the caption does not have, and a second copy
of a chart is the copy that keeps the old figures after a sweep republishes.

Everything else lives in the right sidebar, which is the whole of the UI that is not the diagram.
The controls used to ring the canvas, which cost the drawing its four corners and left each of them
too little room; one column keeps the canvas rectangular and gives every control a full width.
Below `lg` the same component fills a sheet, so there is one list of controls rather than a desktop
set and a mobile set that drift apart.

- **Legend** names the five point roles by colour, plus the red ring while comparing.
- **Dimensions** sets layer, head, neuron and expert counts. Only the sliders that mean something
  under the current traits are shown.
- **Traits** toggles the architectural variations individually, grouped by section — attention,
  norms, MLP, residual. Each trait's hover card explains it, lists real models that have it, and
  names the points it adds, removes, splits or rewrites.
- **Models matching these traits** answers the section above it, and says so when no named family
  has exactly the combination now toggled.
- **Search** filters by name. It searches whatever **Names** is currently showing, so `hook_z` is a
  TransformerLens question and `attentions_output` an nnterp one, and it matches the full name
  rather than the shortened label, so `blocks.1.attn` picks out one layer's attention. Hits are
  ringed and the rest of the diagram fades back. A point the current stack has no name for never
  matches, which is the honest answer to a question asked in that stack's vocabulary.
- **Names** chooses which stack's names to print on the diagram, and with it whether each point's
  card carries the interp-engine calls that read it.

The counts are schematic and capped low enough that each head, neuron and expert can be drawn as
its own mark — that is the point of showing them.

**Every layer is drawn in full, at every depth the slider reaches**, and the layer bands are inert:
there is nothing left for a click to select. There used to be a visibility pass that expanded one
focused layer and elided runs of the rest into labelled gaps, sized against a 128-layer stack.
Against a slider that stops at sixteen it was doing the reverse of its job. At the default depth of
four it never triggered, so the band click that chose the expanded layer looked like it did nothing
— and past four it did something worse, turning every layer but one into two featureless boxes and
hiding the per-point detail that is the entire subject of the diagram. Sixteen full layers measures
~13,300px and re-renders in ~175ms on a slider nudge: wide, but the scrubber under the canvas exists
to move around exactly that. A much deeper stack wants virtualisation over the horizontal scroll, not
a mode where most of the drawing is missing.

**Each name is printed on a pill**, in slate, turning sky with darker type and a `sky-600` border on
the point under the pointer or holding the pin. The canvas under a label is not blank — it is edges,
layer bands and sometimes a neighbouring glyph — and small grey type crossing a curve was the least
legible thing in the drawing. The pill also makes the highlight legible from across the diagram,
where a ring around one dot among a hundred is not — and the border is what does most of that, since
`sky-100` against `slate-200` is a change of hue at nearly the same lightness.

**The name answers to the pointer too**, hovering and pinning the point it names. It is the larger of
the two targets by some margin: a `resid_mid` is a single 11px dot, and its name is 70px of pill right
beside it.

Its width comes from a character count, in `labelBox`. Nothing measures text here: the labels are
monospace, so a count _is_ a width, which is what lets a server-rendered SVG place a box around a
label it never laid out.

Its height comes from `spacing()`, with the rest of the vertical geometry, because vertical padding
in a pill is paid for twice: once inside the pill and again in `stagger`, which has to clear a whole
pill or two neighbouring labels overlap. Two pixels above and below the type therefore cost four more
of row clearance, which is a bargain on a laptop and not on a phone in compare mode — the one place
the pill is drawn a pixel shallower. Horizontal padding is 6px at both sizes: width is scarce here
too, but not in a way that spending rows would fix.

Names still outrun their column: `expert_weights.1` is 16 characters in a 58px column, and every
TransformerLens name is longer than the interp-engine one it translates. Those labels overlapped as
bare text too, so what the pill changes is that the overlap is now visible — and, once a name is a
target, that the overlap has to be arbitrated rather than merely looked at.

Three things do that, all of them paint order, which is the only depth SVG has. The glyphs are drawn
in one pass and the names in another, so a name is never underneath a neighbouring point's target and
the tip of a long name reaches the point it names rather than the one it sits over. Within each pass
the point already being pointed at is drawn **last**, so moving along a label and into the part a
neighbour covers stays on the label you started from instead of flickering between the two. And the
resting pill's outline is the canvas' own white, showing only where two pills touch, which keeps them
from merging into one shape with a word buried in it.

The canvas is always wider than the window, so it can be dragged to pan, and the strip along the
bottom is a minimap of the whole stack: click it to jump, or drag the window to scrub.

**Nothing on the diagram moves on its own.** It used to carry a light travelling left to right along
the edges, under the idea that a forward pass is a thing that flows. It did not read as one: the
light was a moving band of gradient, a smear on the long straight spine and a rendering artefact on
the branches, and it left the static edges drawn faintly underneath it so the drawing was at its
least legible whenever the light was elsewhere — which, at six and a half seconds a crossing, was
nearly always. The edge weights here are set to be read on their own.

What remains is animation the reader causes: the ripple on a reworked point, the card fading in
under the pointer, a dimension slider opening when a trait makes it meaningful, and the scroll that
centres an off-screen difference. All are short, one-shot, and tied to something that just happened.
That is also why there is no `prefers-reduced-motion` branch anywhere — the one animation that
warranted one ran forever whether or not anybody had asked for it.

## Comparing two architectures

**Compare Mode** splits the screen: a second diagram appears below the first, separated by a
hairline, and each carries its own picker and timeline in the band above it. Neither picker offers
what the other is already showing, though both keep the full axis, since dropping a year off one end
would read as the range having changed rather than one release being unavailable. The pair scrolls
as one, and points the two disagree about are ringed in red on both — an **engine difference**,
which the legend spells out as missing on one side or derived differently.

Beside each picker, **Architecture diff vs …** says what that architecture is in terms of the other:
`has QK-norm`, `no MoE`, one pill per trait the two disagree about, green for present and grey for
absent. It is the same comparison as the red rings, one level up — traits rather than points — and
reads as a description of the pane it sits on rather than half of a difference. Below `sm` it moves
**under** the picker rather than taking a third of the row beside it: six pills in 120px is a
horizontal scrollbar, and what a pill says is most of its value.

There it scrolls sideways on one line, which took saying twice. A box that scrolls on one axis is not
`visible` on the other however you write it — CSS promotes the other axis to `auto` — so the row also
scrolled a pill's own height vertically, in a box only one pill tall. And it is drawn with no scrollbar
rather than a thin one, because a thin one is laid out _inside_ that box and there was nothing to give
it. The pill cut off at the edge is the affordance.

Both diagrams are also drawn with **tighter rows** on a phone — about 22% shorter, from `spacing()` in
`lib/layout.ts`. Every number cut there is label clearance, so the label offsets in `HookPoint` come
out of the same table: a row clearance that no longer covers `labelAbove + stagger + labelBox().top`
puts one column's labels through the row above. Both panes must be built with the same value or the
pair stops lining up, which is why it is an argument to `buildGraph` rather than a class. It is only
on where the height is genuinely scarce — two diagrams on a phone. One diagram there has the whole
screen and reads better with the room.

Hovering a pill opens the trait's usual card, led by which side has it, and narrows the red rings to
the points that one difference is responsible for: the other hundred go out, and everything left
unmarked fades back the way it does under a search. That is the only way to read a diff this size —
not "where do these two disagree" but "where does _this_ disagreement land". Attribution comes from
the same `pointEffects` that writes the card, so a card that says it rewrites `attn_scores` lights up
`attn_scores` and nothing else.

If none of those points are on screen — MoE's are all in the MLP rows of the sparse layers, a long
way right of where the diagram opens — the pair scrolls until the first of them is centred, because
a highlight nobody can see looks exactly like no highlight. Only the top pane runs the animation;
the other is carried by the existing scroll sync, and ignores it for as long as it lasts, or the
echo would cancel the scroll a frame after it started.

Opening compare mode without a partner already chosen picks the architecture that differs in the
most traits, since a comparison against a near-identical family looks broken.

A point counts as different when one side has it and the other does not, or when both have it but
arrive at it differently — see `lib/diff.ts` for exactly what is compared. Same-looking is not the
same as same: `resid_post` is `resid_mid + mlp_out_post` on both a dense and a sparse block, so it
is not marked, even though the number it holds is nothing alike.

Both diagrams are built against one **alignment** (`alignmentFor` in `lib/buildGraph.ts`): the
columns and rows are planned for the union of the two architectures, and each side then fills in
the subset it has. That is what puts `attn_scores` at the same x in both panes and leaves a visible
hole where a side is missing something, instead of sliding everything after it to the left. The
union is only used to _reserve_ space — whether a point exists is still asked of each side
separately, or one architecture's refusal would delete the other's column.

The trait pills are hidden while comparing — the sidebar says so where they were, with a second way
out. Both diagrams show a named architecture, and hand-editing the traits of one of them would leave
the comparison describing something with no name. Dimensions still apply to both, and a slider that
either architecture needs is shown, so the only thing that varies is the architecture.

## Linking to a point

The address bar names what is on screen, so the link to a diagram is the one already in it and there
is nothing to press. Four parameters, all optional, and a default is left out rather than spelled
out — a plain visit stays at `/`, and a link carries only what it changed:

| Parameter | What it names                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------ |
| `arch`    | The architecture, by HF class: `?arch=Gemma3ForCausalLM`. Absent means whichever one the app opens on.  |
| `vs`      | A second architecture, which opens compare mode with it in the lower pane.                             |
| `point`   | A point's card, open, by address: `?point=resid_post.9`, `?point=resid_pre.2.stream-1`, `?point=embeddings`. |
| `pane`    | `1` when the linked point is in the lower pane of a comparison. Meaningless, and dropped, without `vs`. |

The address is the engine's own — `format_address` from `interp_engine/address.py`, the same string a
`Cache` takes and the same string the card prints — so a point you can read about is a point you can
link to without translating anything. **A linked point brings its depth with it**: `resid_post.9` is a
point on a ten-layer stack and nothing whatever on the four-layer default, so the layer slider opens
where the address needs it, and the pane opens scrolled to the point rather than to layer zero, since
a card open on something off the side of the canvas reads as a card about nothing.

Everything else is dropped on its own terms. A point the architecture does not have — `resid_mid` on
a parallel block, a router on a dense layer — still opens the diagram, just without a card; a family
that does not exist, a point past the sliders' reach, or a comparison of an architecture against
itself is treated as though it had not been written. A URL retyped out of a screenshot opens the
default diagram rather than an error, and gets tidied to what it actually meant.

That tolerance is why the engine's `docs/SUPPORTED_POINTS.md` links all 31 of its drawn points here —
each on an architecture that has it — and why `npm run links:check` exists: a dropped parameter opens
a plausible diagram rather than an error, so nothing about a link that stopped meaning what it said is
visible to a reader. The check builds each linked graph and fails if the card would not open.

The one thing a link cannot carry is a **hand-edited trait set**, which has no name to write down —
the toggles are not in the query. Editing them takes `arch` out of the URL rather than leaving it
naming the family the toggles started from, which would be a link to a diagram the sender is not
looking at.

Written with `replaceState`, not a navigation: this is the same page renamed, and a history entry per
point clicked would make Back an undo button for a card the next click closes anyway. Reading it back
is the part with a trap in it, and `lib/link.ts` carries the argument — the query is not readable
during the prerender, nor during the hydration render that has to match the prerendered HTML, so it
arrives one render late, and `Page` keys the app on it rather than applying it from an effect. That
keeps `/` prerendered as a whole diagram instead of a `Suspense` fallback, which is what
`useSearchParams` at the top of a tree this size would cost.

## Running a snippet

**Notebook**, beside Copy under the tabs, opens the selected snippet in a fresh Google Colab notebook
with `interp-engine` installed from PyPI. One press does two things and it is worth knowing which: the
snippet goes on the clipboard, and the notebook that opens is a template whose last cell is where it
gets pasted.

That split is Colab's, not a shortcut. Colab opens a notebook that is already on GitHub — or in a Gist,
or in Drive — and there is no parameter that carries cell contents, no import-from-any-URL, and no way
to hand it a file; the request for one is [colabtools#1305], still open. The alternative that would
avoid the paste is creating a Gist per press, which means a credential in this deployment, an
unauthenticated route that writes to a GitHub account, and a public artefact left behind for every
reader who was merely curious. So the notebook is committed and the snippet is not.

**On the two vLLM tabs the clipboard does not hold what the card shows.** A notebook kernel runs its
cells inside an event loop, and there the sync free functions raise `NestedEventLoop` rather than nest
a second one — so `run_with_cache` on a vLLM model, which reaches the engine through that bridge, is
handed over as `await model.capture(...)` instead, with the substitution written into the snippet as a
comment. `eager` is copied as it stands, because its free functions keep in-process bodies for an
`EagerModel` and never reach the bridge. `data/snippets.ts` builds both forms; the card shows `code`
and the button carries `notebook`.

Three templates, in `notebooks/` at the repository root:

| Template                          | Tabs it serves       | Installs              |
| --------------------------------- | -------------------- | --------------------- |
| `interp_engine_vllm.ipynb`        | `vllm`, `vllm async` | `interp-engine[vllm]` |
| `interp_engine_vllm_static.ipynb` | `vllm static`        | `interp-engine[vllm]` |
| `interp_engine_eager.ipynb`       | `eager`              | `interp-engine`       |

Between the eager one and the other two the extra is the whole difference, and it is not a small one:
gigabytes of CUDA wheel, a torch Colab has to be restarted to swap, and a GPU runtime the backend
cannot start without, against transformers and the engine, which run on Colab's CPU runtime.

`vllm static` installs the same extra as `vllm` and is still a separate template, because the install
is not what a reader of that tab needs explaining. Its taps are recorded into CUDA graphs at load, so
the snippet names its point twice — once in `static_points=`, once in the capture — reading a second
point means editing the load and re-running, and the static buffers want more of the card than the
hooked backend does. None of that applies to the hooked template, and all of it belongs before the
paste rather than in a footnote after it.

`vllm async` maps to the vLLM template rather than to a fourth one: it is a method on a vLLM model, so
it needs the same extra, and its top-level `await` is something Colab's kernel runs as it stands. Each
template puts what the reader needs *before* the paste rather than after it — which runtime to pick and
what a free T4 does and does not fit, a Hugging Face login for the gated families commented out with
its token in Colab's Secrets rather than in a cell, and the diagram's own URL scheme, so a notebook
that outlives the tab it was opened from still says where the snippet is documented.

**The repository has to be public for the button to work at all.** Colab fetches a `/github/` path
anonymously with no authorization step — that is the whole reason the form exists — so against a private
`decoderesearch/interp-engine` a reader gets a GitHub sign-in wall in a new tab rather than a notebook.
It was still private when this was written, alongside `interp-engine` not yet being on PyPI; the button
starts working when both of those land, and needs no change here when they do.

They sit at the repository root, outside this app, for the same reason the query string above is stable:
Colab reads that path from GitHub's `main`, so it lands in browser history and in links people send each
other. Nothing here is built from those files and nothing checks them, which is the hazard — a template
renamed without `lib/colab.ts`, or edited in a commit that has not been pushed, is a 404 in a new tab
that this deployment cannot see. `npm run build` will not catch it; pushing the pair together is what
does.

[colabtools#1305]: https://github.com/googlecolab/colabtools/issues/1305

## Ask Riz

The frog in the header opens a chat panel that answers questions about interp-engine: which point
holds the tensor you want, what another stack calls it, whether a backend can serve it, and the code
that reads it. It is the one part of this app that is not a pure function of `data/*.ts`.

**It knows this repository and nothing else.** `knowledge/bundle.generated.ts` is the engine's
README, every file in `docs/`, four modules of `interp_engine/` and this directory's README and
data tables, compiled verbatim into one string — 21 documents, about 398 KB. That whole thing is the
head of every request, marked `cache_control: ephemeral`.

Size it by measuring, per model, and not by dividing bytes by four. At 367 KB this bundle measured
**103k tokens** on the Haiku 4.5 default and 132k on Sonnet 5, against the 93k that bytes/4 predicts
— 10% under on Haiku and 40% under on Sonnet, because the tokenizer changed at Claude 4.7 and
produces roughly 30% more tokens for the same text. A byte estimate is not merely imprecise, it is
wrong by a different amount per model. Anything sizing this prefix should call `count_tokens` for the
model it is actually going to use.

There is no vector database, no embedding pipeline and no sync job, because at this size retrieval
would be infrastructure bought with nothing. A corpus that fits in a context window is cheaper to
send whole than to index: Anthropic bills a cached prefix at a tenth of the input rate on a hit, so a
question that hits the cache costs about a cent against twenty-one for the one that wrote it.
Retrieval would save a fraction of that and lose the thing that makes the answers good, which is that
the model can see the point table and the refusal table at the same time as the question.

A consequence worth knowing before reading a usage dashboard: **a cache read is still counted as
input tokens, and billed at a tenth.** Three questions in one sitting report around 310k tokens and
cost about 23 cents, nearly all of it the first one. The dashboard's token count will not show you
that split; the route's log line is the honest view, where `cache hit` on everything after the first
is the shape to expect and `UNCACHED` is the one to chase.

**The prefix is cached for an hour, not the default five minutes**, and that is a decision about
traffic rather than about correctness. Anthropic keys the cache on the content of the prefix, so
every reader of this site shares one entry — the question is only whether a visitor finds it warm.
Five minutes rarely spans two visitors to a docs page, so nearly every conversation was paying to
write the whole prefix again. An hour usually does span them, and a hit refreshes the hour for free,
so a steady trickle of readers can run all day on one write. It costs 2x to write instead of 1.25x,
which pays for itself at about two conversations an hour and is a slight loss below one; if this page
ever goes quiet enough that most visitors arrive to a cold cache anyway, `ttl` in `app/api/ask/route.ts`
is the thing to put back.

Feeding it source rather than only prose is deliberate. Whether something is supported is the
question it gets most and the one where a plausible guess costs the most, so `points.py` and
`dispatch.py` go in whole: the bot is instructed to quote the engine's own refusal wording rather
than paraphrase it, because "unimplemented" and "unreachable" are the difference between filing a bug
and switching backend, and a paraphrase loses exactly that. `__init__.py` is there so that a name
absent from `__all__` is checkably absent rather than plausibly present.

**The bundle is committed, and that is load-bearing twice over.** Vercel builds this directory with
its Root Directory set to `visualizer-web`, so reading `../docs` at build time would make the
deployment depend on a dashboard toggle that no file in the repository can declare. And
`vercel.json`'s `ignoreCommand` decides whether a commit rebuilds at all. Building without `../docs`
present is tolerated and keeps the committed bundle; building with `docs/` holding a file the
manifest does not name is an error, because that gap is precisely how a bot goes quietly out of date.

**Editing `docs/` refreshes Riz without anyone remembering to do anything**, which took three
separate pieces and is worth spelling out, because each covers a different way the previous
arrangement went stale silently:

| Piece | What it stops |
| ----- | ------------- |
| `ignoreCommand` names `../docs`, `../README.md` and `../interp_engine` alongside `./` | A docs-only commit cancelling its own deployment. It changed nothing in this directory, so the old pathspec skipped the build entirely and the reader kept the previous release's answers. |
| `prebuild` runs the generator before `next build` | A deployment shipping a bundle nobody regenerated. If the sources are reachable it rebuilds from them; if they are not, it keeps the committed file. |
| `.github/workflows/visualizer-tests.yml` runs `knowledge:check` on any PR touching those same paths | Drift reaching main. The check runs **before** the build in that workflow, since `prebuild` would otherwise repair the drift the step exists to report. |

An absent path makes `git diff` exit non-zero, which is the fail-safe direction: a checkout that
somehow lacks `../docs` builds rather than skipping.

Locally, `npm run knowledge` rebuilds the bundle, `npm run knowledge:check` fails when it has
drifted, and `make viz-check` runs the second beside eslint and tsc. Note that `npm run build` now
regenerates as a side effect, so a local build can leave a modified file in the tree — that is drift
being surfaced, not the build being dirty.

**What the reader is looking at goes in a second system block, deliberately outside the cache.** The
architecture, the trait set, the dimensions, the naming stack, and the point whose card is open, sent
with each message. That is what lets "why is `resid_mid` not on my diagram" be answered from the
parallel-block trait that removed it rather than in general. It is a separate block because the
cached one has to be byte-identical between messages: folding a slider position into it would rewrite
the prefix every time, pay the 1.25x cache-write premium on every message and never once read. That
failure is invisible — the answer is still correct and the bill is ten times larger — so the route
logs `cache hit` / `cache write` / `UNCACHED` per answer, and `UNCACHED` on a second message is the
symptom to look for.

**Cost is bounded in four places**, because an unauthenticated route in front of a billed API key is
bounded by nothing else. Two sliding windows per IP in Upstash Redis (twenty questions per five
minutes, two hundred per day) catch a held-down enter key and a paced script respectively. Below them sit
caps a rate limit cannot express: 4,000 characters per question, twelve messages of history, 1,200
output tokens. Missing Redis credentials make the route refuse rather than serve unlimited — a
limiter that fails open is a limiter that is off, and the first you would hear of it is the invoice.
Development is exempt, on `NODE_ENV`, so the panel runs locally without provisioning a database.

**The fourth is the only one that actually caps the bill**, and it is worth being clear about why the
first three do not. A per-IP limit prices abuse; it does not bound it. Two hundred questions a day is
about $5.80 at the worst a single question can cost, and a rotating proxy pool rents a thousand
addresses for less than that — so per-IP windows alone multiply rather than cap. `GLOBAL` is a third
window keyed on a constant, 2,000 questions a day across every reader, overridable with
`RIZ_GLOBAL_DAILY` because the right number is a budget decision. With the per-IP allowance set where
it is, ten addresses reach that ceiling, which is the intended split: the windows above are sized for
someone genuinely reading, and this is the one a proxy pool hits. It is consumed **only after** the
per-IP checks pass:
charging the shared budget for traffic that was already refused would let one blocked address spend
everyone else's allowance, which converts a rate limit into a free denial of service.

IPv6 is bucketed to the /64 rather than the address, for the same reason. The smallest block anyone
is assigned is a /64 and most ISPs hand out more, so a limiter keyed on all 128 bits hands such a
client 18 quintillion fresh allowances — no limiter at all, for exactly the population best equipped
to abuse it, and invisible while it happens because every request looks like a new reader. Note that
`::` has to be expanded before the prefix is taken: `2001:db8::1` truncated as text is a different
/64 from the one it means.

The panel is portalled to `document.body` and positioned in the window at `z-40`, the same treatment
and the same reason as the hook card: the header it hangs from is a flex row in an `overflow-hidden`
column, and a panel rendered in place would be clipped to the height of the row that opened it. It is
not a Radix popover, though one would hand over focus management for free — a popover dismisses on
outside click, and the whole point of this panel is to be read while pointing at the diagram behind
it. Escape and the button do the dismissing. Its code blocks are the hook card's, down to the plate
and the copy button, because a snippet from Riz and a snippet from a point's card are the same kind
of thing.

**It starts open**, and the launcher is emerald against a header of slate and sky so that it reads as
the one thing on the page that is not part of the diagram. A chat box does not advertise that it has
read the docs, and a reader who came for a hook point map has no reason to go looking — so it
introduces itself once, and Escape or either button closes it. Nothing is sent until a question is
asked, so an ignored panel costs nothing but the space.

Not on a first visit, though. There the welcome dialog is the introduction, and two of them at once is
neither — so the panel waits behind it and opens as the dialog closes, because closing the dialog is
what marks the visit. It still introduces itself exactly once, one beat later.

Escape is the exception: it clears both, and one press does it. React flushes a discrete event's
updates before the keydown has finished propagating, so the press that dismissed the dialog reaches
the panel's own Escape listener as that listener is being attached. The outcome is the one you would
have chosen, but event ordering is what produces it and nothing declares it — worth knowing before
either half of that is rearranged.

**The launcher has two placements and one state.** From `sm` up it is a pill in the header row. On a
phone the header cannot afford it — that row is already carrying the wordmark, the mode toggle, the
Controls trigger and the repo link — so it becomes the intercom convention instead, a 48px circle
pinned over the bottom right corner, and the panel anchors to that corner too and grows upward out of
it. The floating one is portaled to `document.body` rather than left where it sits in the tree: the
header sets `backdrop-blur`, and a `backdrop-filter` makes an element the containing block for its
`position: fixed` descendants, so a fixed button inside the header pins itself to the header's corner
instead of the viewport's. It sits over the scrubber strip, which is the price of that corner and the
reason it is only there.

The input is **not** focused in that state, only when the panel is opened by a click. Focusing a field
nobody asked for raises the keyboard over the diagram on a phone, and takes the caret off the page
everywhere else, so the effect watches the false-to-true transition rather than the value.

### Which model

`claude-haiku-4-5` by default, overridable with `ANTHROPIC_MODEL`, and that default is a result rather
than a preference. `npm run eval` asks two models the same eight questions through the real
`SYSTEM_PROMPT` and `KNOWLEDGE` — imported, not paraphrased, so a result is evidence about the
deployed configuration — and scores them on an identifier that has to appear in the answer. The cases
are drawn from where the docs say a wrong answer is expensive: the `hook_mlp_out` collision that
silently mis-trains an SAE, the unimplemented/unreachable split that decides whether you file a bug or
switch backend, and two facts that changed after both models' cutoffs.

**Both score 8/8, and Haiku 4.5 is 2.6x cheaper**, which is a wider gap than the rate card shows
because it compounds two things: half the price per token, and 22% fewer tokens for the identical
prefix, since Sonnet 5 uses the newer, hungrier tokenizer and Haiku 4.5 does not.

| | Prefix tokens | Warm question | 1h cache write |
| --- | --- | --- | --- |
| Haiku 4.5 (default) | 102,691 | $0.010 | $0.21 |
| Sonnet 5 | 132,214 | $0.026 | $0.53 |

Take 8/8 for what it is: eight questions is enough to catch a model that cannot do this job and
nowhere near enough to rank two that can. The honest reading is that Haiku clears the bar these cases
set, not that it equals Sonnet — so extend `CASES` before leaning harder on the result, read the
transcripts the script prints for that purpose, and re-run before any swap.

The one thing to know before writing a case is that `forbid` is where this goes wrong. The first three
runs of that file were all corrections to a `forbid`, never findings about a model: a good answer
names the confusable tensor precisely in order to warn you off it, so forbidding the identifier scores
the warning as the error and rewards whichever model explained least. Forbid a claim, never a name.

Haiku honors `temperature`, which the route sets to 0.2. Sonnet 5 does not, and the SDK logs that it
is dropping it once per request — worth knowing if `ANTHROPIC_MODEL` ever points back at it, since
the answers get slightly more variable and the logs get noisier at the same time.

**Start over** beside the close button empties the thread, and it saves money as well as screen. The
prefix is cached and identical for every thread, so the history is simultaneously the only part of
the prompt billed at full rate and the only part that grows — twelve messages in, a question about
something unrelated still carries every earlier answer. It is disabled rather than hidden on an empty
thread, so the close button does not move sideways the moment a first answer lands.

With none of the environment variables set, the app builds, deploys and renders exactly as it did
before the panel existed, and the panel says it is not configured. Nothing else degrades.

### The frog

`public/riz.png` is the source, 818px square, and it is used at two different crops. The launcher
circle crops it in CSS — `scale-[1.35]` at `object-position: 50% 38%` — which keeps a little lily pad
in frame, because at 24px beside a text label the silhouette is doing the work. The same crop scales
up for the floating button on a phone, where the circle is the whole control and there is no label
beside it. It loads with `priority`, since the panel is open from the first paint and the same frog
is in its header.

The favicons crop tighter and are baked, since a tab favicon is 16px and the only things that survive
at that size are the eyes. `app/icon.png` (512), `app/apple-icon.png` (180, flattened onto
`lime-50` because a transparent Apple touch icon gets composited on black) and `app/favicon.ico`
(48/32/16 in one file, for the browsers and crawlers that ask for `/favicon.ico` regardless of what
the markup says) are all the same region:

```bash
magick public/riz.png -crop 370x370+197+128 +repage -resize 512x512 app/icon.png
```

That rectangle was picked by rendering the candidates at true 16px rather than by eye at full size,
which is the step worth repeating if the artwork ever changes: the framing that looks best large
loses the eyes first, and the one with comfortable headroom loses them fastest. Next.js picks all
three up by filename — there is no `<link>` tag anywhere in `app/layout.tsx`.

## Running it

```bash
npm install
npm run dev
```

The chat panel additionally needs `cp .env.example .env.local` and an `ANTHROPIC_API_KEY` in it. The
diagram does not.

## The samples site at `/docs`

`docs-site/` is a separate Docusaurus site — short, copyable interp-engine examples, one page per
job, reached from the **Docs** button in the header. Editing a page:

```bash
npm --prefix docs-site start   # its own dev server, at /docs
npm run docs                   # build it into public/docs, which is what ships
```

`npm run docs` is part of `prebuild`, so `npm run build` produces both. Two things are load-bearing
in how it is mounted, and both are small:

- **`baseUrl: "/docs/"` with `trailingSlash: false`**, which emits `steering.html` rather than
  `steering/index.html`. The build writes into `public/docs`, so Next serves every page and asset as
  a static file and the deployment stays a CDN.
- **Two rewrites in `next.config.ts`**, in `afterFiles`, serving the extensionless URL the site's own
  navbar links to. `afterFiles` runs after the public directory is checked, so a real file is never
  rewritten and only a path with nothing behind it reaches the rules. They match one segment, which
  is why every doc id in `sidebars.ts` is flat.

A link from the samples back to the diagram cannot be written as `/`: Docusaurus resolves an
internal href against `baseUrl` and hands back `/docs/`, including through the `pathname://` escape
hatch. There are two such links, and neither goes through a Docusaurus href. The navbar entry is a
`type: "html"` item, which is emitted verbatim. The brand in the top-left corner is
`src/theme/Navbar/Logo`, swizzled for the same reason — `navbar.logo.href` is resolved against
`baseUrl` too — and it renders a plain `<a>` rather than `@docusaurus/Link`, because `/` is not a
route in this SPA and a client-side navigation to it lands on this site's own 404 instead of leaving
it.

The snippets are covered by the engine's own `tests/test_doc_code_fences.py`, which asserts every
page parses as Python and imports only names in `interp_engine.__all__` — the samples site would
build happily around a rename, since it never imports the package.

## Layout

Data and UI are kept apart so the model of what a transformer _is_ can be edited without touching
any rendering code.

| Path                    | What lives there                                                      |
| ----------------------- | --------------------------------------------------------------------- |
| `data/points.ts`        | The canonical point table, transcribed from `interp_engine/points.py` |
| `data/benchmarks.ts`    | The throughput card's types and display rules; the figures beside it in `benchmarks.generated.ts` are written by `python -m benchmarks.publish` |
| `data/engines.ts`       | Each stack's naming rules, from `interp_engine/mappers.py`            |
| `data/traits.ts`        | Architectural traits, mirroring `interp_engine/arch.py`'s `Quirks`    |
| `data/architectures.ts` | Family presets, keyed by HF architecture class                        |
| `data/dimensions.ts`    | The sliders and the traits that make each one meaningful              |
| `data/formulas.ts`      | How each point is derived, trait by trait                             |
| `data/snippets.ts`      | The interp-engine call that reads a point, one per tab                |
| `lib/buildGraph.ts`     | Pure `(dimensions, traits) -> nodes, edges, layer bands`              |
| `lib/diff.ts`           | What counts as a difference between two architectures                 |
| `lib/link.ts`           | What the query string can say, and when it is safe to read            |
| `lib/colab.ts`          | Which committed Colab template a snippet's backend needs              |
| `lib/layout.ts`         | Glyph sizes, row placement, path shapes                               |
| `lib/firstVisit.ts`     | Whether the reader has been here before, and who is told when that changes |
| `lib/assets.ts`         | The one asset this app does not serve itself, and the CSP host it needs |
| `components/`           | Rendering only                                                        |
| `components/evidence/`  | The charts and tables that back a claim, shared by the caption's hover cards and the introduction |
| `knowledge/manifest.mjs` | Which files Riz is allowed to know, in prefix order                  |
| `knowledge/prompt.ts`   | Riz's instructions, and how the reader's diagram is described to it   |
| `knowledge/index.ts`    | How the generated documents are framed as one cacheable string        |
| `app/api/ask/route.ts`  | The only route that is not prerendered                                |
| `scripts/model-eval.ts` | Scores two models on the real prompt — `npm run eval`                 |
| `scripts/check-doc-links.ts` | Builds every graph `docs/SUPPORTED_POINTS.md` links to — `npm run links:check` |
| `docs-site/docs/`       | The samples, one page per job — served at `/docs`, see above           |
| `docs-site/sidebars.ts` | Their reading order, and why every doc id is flat                     |

Adding a fourth naming stack means adding one entry to `ENGINES`; nothing else knows how many
there are.

### Two rules the graph builder keeps

They are the difference between a diagram that teaches and one that misleads:

- A point the architecture does not have is **not drawn** — no `resid_mid` on a parallel block, no
  routing points on a dense model.
- A point the architecture has but the engine refuses on _this_ layer is **drawn dimmed**, carrying
  the reason — attention points on the linear-attention layers of a hybrid stack.

Both rules read the trait set and nothing else — `exists` is not told which architecture it is
drawing, and the hyper-connection points are the case that tempts you to change that. Only DeepSeek-V4
has them here, but the thing they follow from is the trunk having more than one stream, not the name of
the family that ships it: upstream, `points_for` takes a stream count. A second family with the same
shape gets those rows here for free, even where it puts the tensors behind other module names and in
another order — that is `facts.HYPER_CONNECTION_LAYOUTS`' problem, an address, which the diagram never
draws.

All seven of those points are drawn, and three of them are not tensors on the spine the way the rest of
the diagram's marks are. `resid_streams` is the stack the per-stream trunk nodes are slices of, so it sits
on the centre line one column along from `resid_pre`, with a gather edge from each stream — the same
tensor, said the other way round, rather than a later moment in the block. The two mixing matrices are
weights, not activations, and they are drawn one row out from the collapse they feed, on the row `value`
uses above the spine and `router_logits` uses below it. That row is claimed in `rowFanouts` rather than
assumed: every hyper-connection family so far also runs latent attention, which leaves the row above the
spine otherwise empty, and a glyph on an unclaimed row would land on the spine itself.

### Traits that move nothing

Fused QKV, attention sinks, logit softcapping, residual multipliers and shared experts rewrite the
arithmetic behind points that stay exactly where they were, so toggling one would otherwise look like
a no-op. Three things surface that:

- The affected points **ripple once** on the change.
- The trait's hover card names them under _Rewrites_.
- Each point's popover ends with **Why it reads this way** — every active trait that shaped this
  particular point, and what it did to it.

The first two are found by diffing `data/formulas.ts` across the two trait sets, so they cannot
drift from what the popover then goes on to say. The third reads a table in the same file whose
keys are exactly the pairs that diff picks out.

## Deploying

Vercel, with the project's **Root Directory** set to `visualizer-web`. That one setting lives in the
dashboard and nowhere else — a repository cannot declare it, because Vercel has to know which
directory to read a `vercel.json` out of before it can read one. Left at the repository root, Vercel
finds no `package.json`, falls back to the framework preset "Other", and then fails looking for the
static directory that preset expects:

```
No Output Directory named "public" found after the Build completed.
```

That is the whole of the manual setup. Everything downstream of it is in `vercel.json` here, which
overrides the dashboard rather than repeating it, so a project pointed at this directory deploys
correctly whatever its settings page happens to say:

| Key               | Why                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `framework`       | Pins the Next.js preset, which is what supplies the real output directory.                                                                                                                             |
| `outputDirectory` | `null` clears a stale dashboard override. `framework` alone does not — the two are separate settings, and a project that once said `public` goes on saying it.                                         |
| `ignoreCommand`   | Cancels the build when the commit touched nothing this app is built from. The repository is mostly Python, so most pushes have nothing to rebuild. `git diff` runs from the Root Directory, hence the `./` — and the three `../` paths beside it, which are Riz's knowledge sources. Without them a docs edit would deploy nothing, since it changes no file in this directory. |

Every page is prerendered at build time, so the deployment is a CDN and exactly one function:
`/api/ask`, which Ask Riz streams from. The whole model of a transformer is still the `data/*.ts`
files compiled in, and no page fetches anything at runtime — the diagram works with the function cold
and with it absent. `/docs` does not change that: it is static HTML out of `public/`, built by
`prebuild` and rewritten to by `next.config.ts`, so it needs no Vercel setting of its own.

That function is the only reason this project has environment variables. `ANTHROPIC_API_KEY`,
optionally `ANTHROPIC_MODEL`, and the two Upstash Redis credentials, all set in the project's
settings and listed in `.env.example`. Unset, the route answers that Riz is not configured on this
deployment and nothing else changes.

Response headers are set in `next.config.ts` rather than in `vercel.json`, so `next start` serves
what Vercel serves and the host stays replaceable. The Content-Security-Policy is `'self'` for every
directive except `script-src` and `style-src`, which also allow `'unsafe-inline'`: a nonce has to be
minted per request by middleware, and that would make the prerendered pages dynamic — a server
rendering a diagram whose every input is client state. Read `'unsafe-inline'` as the price of the
static build, not as a default that was never examined.

Ask Riz needed no directive widened, which is worth stating because it is the kind of thing that gets
widened pre-emptively. The panel calls `/api/ask` on its own origin, so `connect-src 'self'` already
covers it; the frog is `public/riz.png`, so `img-src 'self'` already covers that. The Anthropic call
is made by the function, server side, where a browser policy does not apply.

`img-src` is the one directive that has since been widened, and it is one host for one file: the
introduction's screen recording, on the bucket the rest of Neuronpedia's site assets live on. The
origin is imported from `lib/assets.ts` rather than typed into the policy, because this is a failure
that hides — the slide renders, the recording does not, and the only sign of it is a console line
nobody is reading.

HSTS is `max-age` alone. `includeSubDomains` and `preload` are the right header for whoever owns a
domain's apex and is prepared to serve all of it over HTTPS for the next two years; they are the
wrong one to ship from a single page that happens to be mounted on a subdomain of it, where they
would take out siblings this project has never heard of.
