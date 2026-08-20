/**
 * Everything Riz is allowed to know, named one file at a time.
 *
 * This is a list rather than a glob because the order of these entries is the
 * byte order of a prompt prefix that Anthropic caches: a glob whose directory
 * listing came back in a different order would rewrite the prefix, miss the
 * cache and pay the write premium again, with nothing in the output saying so.
 * Explicit order is the cheapest way to make that impossible.
 *
 * `docs/` is nevertheless cross-checked against this list by the generator, so
 * a doc added upstream is a build failure here rather than a gap the bot
 * discovers in front of a reader. The other directories are not swept: they
 * hold far more than belongs in a prompt, and which of their files is worth the
 * tokens is a judgement, not a pattern.
 *
 * `.mjs` and not `.ts` so that `scripts/build-knowledge.mjs` can import it
 * without a compile step. Nothing in the app reads this file.
 */

/**
 * @typedef {object} Source
 * @property {string} path     Repo-root-relative, and what the model cites.
 * @property {string} title    Shown in the prefix's table of contents.
 * @property {string} why      One line telling the model when to reach for it.
 * @property {"markdown" | "python" | "typescript"} kind
 */

/**
 * The engine's own documentation, in the order docs/README.md's routing table
 * introduces it: what the thing is, then how to use it, then the reference
 * material you only open with a specific question.
 *
 * @type {readonly Source[]}
 */
export const ENGINE_DOCS = [
  {
    path: "README.md",
    title: "interp-engine README",
    why: "The pitch, install, and the one-screen usage example. Start here for orientation; it delegates the detail to docs/.",
    kind: "markdown",
  },
  {
    path: "docs/README.md",
    title: "Documentation index",
    why: "Which doc answers which question. Use it to pick the right source before quoting one.",
    kind: "markdown",
  },
  {
    path: "docs/USAGE.md",
    title: "Usage walkthrough",
    why: "Install to first capture. The canonical snippets: load, capture, generate, steer, lens. Prefer copying from here.",
    kind: "markdown",
  },
  {
    path: "docs/SUPPORTED_POINTS.md",
    title: "Per-point backend support",
    why: "Every point with its width and whether eager and vLLM serve it, plus what a refusal means and how tensor parallelism narrows the vLLM column. The authority on 'can I capture X on vLLM'.",
    kind: "markdown",
  },
  {
    path: "docs/AGENT_INTEGRATION.md",
    title: "Integration and migration recipes",
    why: "Replacing an existing hooking pattern with the engine. Keyed on what is being migrated from.",
    kind: "markdown",
  },
  {
    path: "docs/PORTING.md",
    title: "Porting off HookedTransformer / StandardizedTransformer",
    why: "The short version of the same migration, for TransformerLens and nnterp callers.",
    kind: "markdown",
  },
  {
    path: "docs/ENGINE_HOOK_MAPPINGS.md",
    title: "Cross-stack hook name dictionary",
    why: "Canonical point names against TransformerLens hooks and nnsight/nnterp accessors. The authority on what another stack calls a point.",
    kind: "markdown",
  },
  {
    path: "docs/ARCHITECTURE_QUIRKS.md",
    title: "Architecture quirks",
    why: "Every structural model fact the engine cannot get by asking the model, and the per-family traps where inspection returns a shape-correct wrong answer.",
    kind: "markdown",
  },
  {
    path: "docs/GRADIENTS.md",
    title: "Gradients",
    why: "What is differentiable on which backend, and why gradient support never gates loading.",
    kind: "markdown",
  },
  {
    path: "docs/PERFORMANCE.md",
    title: "Performance",
    why: "vLLM-side speed and feature tradeoffs, and which quantized checkpoints load.",
    kind: "markdown",
  },
  {
    path: "docs/COMPATIBILITY.md",
    title: "Compatibility",
    why: "Which transformers versions are exercised, and which are known to compute a model wrongly.",
    kind: "markdown",
  },
  {
    path: "docs/INTERNALS.md",
    title: "Modules and correctness",
    why: "What each file in interp_engine/ owns, and what the test suite checks. Reach for it when the question is about how the engine is built or how it is verified.",
    kind: "markdown",
  },
];

/**
 * The four modules that decide what an answer about support is allowed to say.
 * Source rather than prose because a refusal should be quoted, not paraphrased:
 * `points.py` and `dispatch.py` carry the engine's own wording for why a thing
 * cannot be done, and that wording is the difference between "file a bug" and
 * "switch backend".
 *
 * @type {readonly Source[]}
 */
export const ENGINE_API = [
  {
    path: "interp_engine/__init__.py",
    title: "Public API surface",
    why: "`__all__`, grouped by task. A name absent from here is not public API and must not appear in a snippet.",
    kind: "python",
  },
  {
    path: "interp_engine/points.py",
    title: "Canonical point registry",
    why: "The point table with each point's scope, width and vLLM support, plus the engine's own refusal reasons.",
    kind: "python",
  },
  {
    path: "interp_engine/protocol.py",
    title: "The InterpModel contract",
    why: "What both backends implement, and what is deliberately excluded from the protocol.",
    kind: "python",
  },
  {
    path: "interp_engine/dispatch.py",
    title: "Capability refusals",
    why: "The CAPABILITIES table: everything exactly one backend can do, why the other cannot, and what to call instead.",
    kind: "python",
  },
];

/**
 * The visualizer itself. The README explains what the diagram is claiming; the
 * data files are what it draws from, and are the same tables the reader is
 * looking at when they ask why a point is missing or dimmed.
 *
 * `formulas.ts`, `snippets.ts` and `dimensions.ts` are deliberately absent:
 * the first two restate what `points.py` and `USAGE.md` already say, at a cost
 * of another 25 KB of prefix, and the third is slider bounds.
 *
 * @type {readonly Source[]}
 */
export const VISUALIZER = [
  {
    path: "visualizer-web/README.md",
    title: "Visualizer README",
    why: "What the diagram shows and the rules it keeps -- when a point is not drawn versus drawn dimmed.",
    kind: "markdown",
  },
  {
    path: "visualizer-web/data/points.ts",
    title: "Visualizer point table",
    why: "The points as the diagram has them, transcribed from points.py. Includes merge behaviour the engine table does not.",
    kind: "typescript",
  },
  {
    path: "visualizer-web/data/traits.ts",
    title: "Architectural traits",
    why: "The toggles down the side of the diagram, and which models exhibit each.",
    kind: "typescript",
  },
  {
    path: "visualizer-web/data/architectures.ts",
    title: "Architecture presets",
    why: "Each family preset as a trait set, keyed by HF architecture class.",
    kind: "typescript",
  },
  {
    path: "visualizer-web/data/engines.ts",
    title: "Naming stacks",
    why: "How each stack spells a point, which is what the naming toggle switches between.",
    kind: "typescript",
  },
];

/** Every source, in prefix order. */
export const MANIFEST = [...ENGINE_DOCS, ...ENGINE_API, ...VISUALIZER];

/**
 * `docs/` entries above, so the generator can assert the directory holds
 * nothing this list has not named.
 */
export const DOCS_LISTED = ENGINE_DOCS.filter((s) =>
  s.path.startsWith("docs/"),
).map((s) => s.path.slice("docs/".length));
