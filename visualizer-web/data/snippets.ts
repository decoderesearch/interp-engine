/**
 * How to read one point with interp-engine.
 *
 * There is one call, not two. Since 1.1 the sync free functions dispatch on the
 * model they are handed, so the eager and vLLM readings of a point differ by the
 * `backend=` argument and nothing else — which is the thing worth showing, and
 * which two side-by-side snippets actively hid. The card offers them as tabs
 * over one snippet so that switching tabs moves exactly the line that differs.
 *
 * Under the tabs, a second switch picks the form of the call: the sync free
 * functions, or the awaited methods on the model. The awaited form is a
 * genuinely different call rather than the same one relabelled: it is what a
 * server holding the model inside its own event loop writes, where the sync
 * functions refuse rather than nest a loop. It returns a plain dict keyed by
 * `Address` with no batch axis, so both differences show up in the last two
 * lines. Every backend has both forms, since every method on the model has a
 * sync twin.
 *
 * Both forms name the tensor with an `Address` bound to a variable, then use it
 * for the request and the read. Every address form is accepted on the way in —
 * the canonical string the diagram prints included — but a `Cache` is the only
 * thing that coerces on the way *out*; `capture` returns a dict where a string
 * is a `KeyError` on a dict that visibly holds the tensor.
 *
 * Which of the three vLLM cases a point falls into is not decided here. It is
 * `spec.vllm`, mirrored from `points.py`: served by the worker's forward hooks,
 * rebuilt off-kernel from captured q/k, or not served at all. The refusal quotes
 * the engine's own reason rather than paraphrasing it, because "unimplemented"
 * and "unreachable" are the difference between filing a bug and switching
 * backend.
 *
 * The shape comments say only what holds for every point. The last axis is the
 * card's width label, and is not restated here where it would have to be guessed
 * per point.
 *
 * Every snippet is built twice, because a notebook cannot run all of them: see
 * `Snippet.notebook`, which is what the Notebook button puts on the clipboard.
 */

import { pointSpec, vllmReason } from "@/data/points";
import type { GraphNode } from "@/lib/types";

/** The repo id to write into a snippet when no named family is selected. */
export const PLACEHOLDER_HF_ID = "org/model";

/** Short enough that the address, not the prompt, is the widest thing here. */
const PROMPT = "Hello, world";

/**
 * The three ways to make the call, grouped by the machine a reader is on.
 *
 * `eager` stands alone under CPU: it is the one backend that runs without a
 * GPU, and the reference the others are scored against. The CUDA section is
 * vLLM's two engines, and `vllm-static` is the one that is *not* the same
 * snippet as the rest: it names the point a second time, at load, which is the
 * whole content of the tab.
 */
export type Variant = "eager" | "vllm" | "vllm-static";

/** One section of the tab row: its eyebrow, and the tabs under it. */
export interface Section {
  eyebrow: string;
  variants: readonly Variant[];
}

export const SECTIONS: readonly Section[] = [
  { eyebrow: "CPU", variants: ["eager"] },
  { eyebrow: "CUDA", variants: ["vllm", "vllm-static"] },
];

export const VARIANT_ORDER: readonly Variant[] = SECTIONS.flatMap(
  (s) => s.variants,
);

/**
 * The tab a card opens on when the point has a path there: what a served
 * deployment runs. A point vLLM cannot serve opens on the first tab that can.
 */
export const DEFAULT_VARIANT: Variant = "vllm";

/**
 * The tab labels. Short, because the two sections share one row inside a card
 * narrower than 500px, and because the surrounding section already says this is
 * interp-engine — a bare `vllm` would otherwise read as vLLM's own hooks, which
 * is a different thing and one this repo also scores.
 */
export const VARIANT_LABEL: Record<Variant, string> = {
  eager: "eager",
  vllm: "vllm",
  "vllm-static": "vllm static",
};

export interface Snippet {
  variant: Variant;
  form: Form;
  /** Python, or null where this variant has no path to the point. */
  code: string | null;
  /**
   * The same reading, as a notebook kernel can run it. Null exactly when `code`
   * is, since the form of the call is not what decides whether there is one.
   */
  notebook: string | null;
  /** Why there is no code, or what the code does not say for itself. */
  note?: string;
}

/** What one builder produces: the code, or why there is none. */
type Reading = Pick<Snippet, "code" | "note">;

/**
 * How the call is written: through the sync free functions, or awaiting the
 * methods on the model. The card's second switch, under the backend tabs.
 */
export type Form = "sync" | "await";

export const FORM_LABEL: Record<Form, string> = {
  sync: "sync",
  await: "async",
};

/** The forms every tab offers: each method on `InterpModel` has a sync twin. */
export const FORMS: readonly Form[] = ["sync", "await"];

/**
 * The form each tab is handed over in when it is going to a notebook.
 *
 * A notebook kernel runs its cells inside an event loop — Colab's does, and so
 * does anything on ipykernel 6 — and there the sync free functions raise
 * `NestedEventLoop` rather than nest a second one. That is every call which
 * reaches the engine through the sync bridge, which is both vLLM tabs:
 * `capture` on a vLLM model dispatches through `sync_model`.
 *
 * `eager` is left alone rather than awaited for symmetry. Its free functions
 * keep in-process bodies for an `EagerModel` and never reach that bridge, so the
 * tab's own snippet is already the one to paste.
 */
const NOTEBOOK_FORM: Record<Variant, Form> = {
  eager: "sync",
  vllm: "await",
  "vllm-static": "await",
};

/**
 * All a snippet reads of a point: which tensor, and where. A `GraphNode`
 * satisfies it, and so does an address written by hand — which is what lets the
 * welcome tour print one reading without building a graph to get a node out of.
 */
export type Located = Pick<GraphNode, "point" | "layer" | "stream">;

/** The attention pattern is a matrix per head, so it has no `pos` axis to name. */
const PATTERN_POINTS = new Set(["attn_probs", "attn_scores"]);

/**
 * Every reading of `node`, one per variant in `VARIANT_ORDER`, in the sync form.
 * Whether a tab has a path is read off the point and not the form, so this is
 * what the tab row needs; the card asks `readingSnippet` for the form it shows.
 */
export function readingSnippets(node: Located, hfId: string): Snippet[] {
  return VARIANT_ORDER.map((variant) => readingSnippet(variant, node, hfId));
}

/** One tab's reading in one form, for a caller that wants a particular pair. */
export function readingSnippet(
  variant: Variant,
  node: Located,
  hfId: string,
  form: Form = "sync",
): Snippet {
  const reading = snippet(variant, node, hfId, form);
  return {
    variant,
    form,
    ...reading,
    notebook: notebookForm(variant, node, hfId, reading, form),
  };
}

/**
 * `reading` again in the form `NOTEBOOK_FORM` asks for, and said so in the code.
 *
 * The substitution is written into the snippet because it is the one thing about
 * it a reader did not choose: they pressed the button on the `vllm` tab's sync
 * form and are about to paste something that does not match it line for line.
 * The alternative — copying the tab verbatim — is a `NestedEventLoop` traceback
 * on the first run, which says the same thing much later and after an install.
 */
function notebookForm(
  variant: Variant,
  node: Located,
  hfId: string,
  reading: Reading,
  shown: Form,
): string | null {
  const form = NOTEBOOK_FORM[variant];
  if (reading.code === null || form === shown) return reading.code;
  // A refusal is read off the point, not off the form, so this is the same
  // non-null code as above and the fallback is unreachable.
  const awaited = snippet(variant, node, hfId, form).code ?? reading.code;
  return [
    `# The ${VARIANT_LABEL[variant]} tab's sync snippet, awaited: a notebook runs its cells inside an`,
    "# event loop, where interp-engine's sync functions refuse rather than nest a second one.",
    awaited,
  ].join("\n");
}

/**
 * `Address(...)` as Python, carrying whatever coordinates the node has —
 * positional, in the field order `address.py` declares append-only, which is
 * what makes the third argument mean `stream`.
 *
 * Exported because the card's heading prints it beside the canonical string,
 * and the two forms of one address should not be built by two functions that
 * can drift.
 */
export function addressCall(node: Located): string {
  const coordinates = [node.layer, node.stream].filter((c) => c !== null);
  return `Address(${['"' + node.point + '"', ...coordinates].join(", ")})`;
}

function snippet(
  variant: Variant,
  node: Located,
  hfId: string,
  form: Form,
): Reading {
  const refusal = variant === "eager" ? null : vllmRefusal(node);
  if (refusal) return refusal;
  if (variant === "vllm-static") return staticSnippet(node, hfId, form);

  const backend = variant === "eager" ? "eager" : "vllm";
  const load = `model = load_model("${hfId}", backend="${backend}")`;

  if (PATTERN_POINTS.has(node.point)) return pattern(variant, node, load, form);
  return activation(node, load, form);
}

/**
 * The static backend, whose snippet is not the same one with a different `backend=`.
 *
 * Its taps are recorded into CUDA graphs at load, so the address has to exist
 * *before* the model does — which inverts the first two lines and is the only
 * thing this tab is here to show. Every other backend can be handed a point it
 * has never seen; this one refuses it, and the refusal is a reload.
 */
function staticSnippet(node: Located, hfId: string, form: Form): Reading {
  const blocked = staticRefusal(node);
  if (blocked) return { code: null, note: blocked };

  const awaited = form === "await";

  // The attention trio is declared as the `attn` tap it is rebuilt from, not under
  // its own name — the same substitution `static_unsupported_reason` tells a caller
  // to make, and the reason this is not simply `static_points=[point]`.
  if (PATTERN_POINTS.has(node.point)) {
    const key = node.point === "attn_scores" ? "scores" : "probs";
    return {
      code: [
        awaited
          ? "from interp_engine import Address, load_model"
          : "from interp_engine import Address, capture_attention, load_model",
        "",
        `tap = Address("attn", ${node.layer})  # q/k, which the matrix is rebuilt from`,
        `model = load_model("${hfId}", backend="vllm-static", static_points=[tap])`,
        ...(awaited
          ? [
              "await model.warmup()",
              `ids = model.to_tokens("${PROMPT}")[0].tolist()`,
              `out = await model.capture_attention(ids, [${node.layer}])`,
            ]
          : [
              `ids = model.to_tokens("${PROMPT}")`,
              `out = capture_attention(model, ids, [${node.layer}])`,
            ]),
        `out[${node.layer}]["${key}"]  # [n_heads, dest, src]`,
      ].join("\n"),
      note: `${node.point} is not a tap of its own on this backend: declare the attn tap and capture_attention recomputes the matrix from the captured q/k.`,
    };
  }

  return {
    code: [
      awaited
        ? "from interp_engine import Address, load_model"
        : "from interp_engine import Address, capture, load_model",
      "",
      `point = ${addressCall(node)}`,
      `model = load_model("${hfId}", backend="vllm-static", static_points=[point])`,
      ...(awaited
        ? [
            "await model.warmup()",
            `ids = model.to_tokens("${PROMPT}")[0].tolist()`,
            "acts = await model.capture(ids, [point])",
            "acts[point]  # [pos, ...]",
          ]
        : [
            `ids = model.to_tokens("${PROMPT}")`,
            "cache = capture(model, ids, [point])",
            "cache[point]  # [batch, pos, ...]",
          ]),
    ].join("\n"),
    note: 'The tap set is baked into the CUDA graphs at load, so this engine serves this point and refuses any other. Pass static_points="auto" for resid_post at every layer instead.',
  };
}

/**
 * Why no tap set can serve this point, or null when one can. Mirrors
 * `static_unsupported_reason` in `vllm_capture/static.py`, which is where the
 * engine refuses the same two points — a wrap goes on a decoder layer, and these
 * are not on one.
 */
function staticRefusal(node: Located): string | null {
  if (node.point === "embeddings" || node.point === "final_norm") {
    return `${node.point} hangs off the trunk rather than a decoder layer, and a static tap wraps a layer module. Use the vllm tab, whose hooks reach it.`;
  }
  return null;
}

/**
 * Why vLLM cannot read this point, or null if it can. Read off the point's spec
 * rather than tried and caught, so the card can say so without a model.
 */
function vllmRefusal(node: Located): Reading | null {
  const spec = pointSpec(node.point);
  const served = spec?.vllm === "hooks" || spec?.vllm === "recompute";
  if (!served) return { code: null, note: vllmReason(node.point) };

  // The worker hangs its hooks off a decoder layer, so a layerless request is
  // refused — except for the trunk-level points, which it reaches by walking the
  // model instead and which therefore take no layer at all. That exception is
  // the engine's, keyed on the point rather than on the layer being absent, so
  // it is read off `scope` here rather than inferred from `layer === null`.
  if (node.layer === null && spec?.scope !== "global") {
    return {
      code: null,
      note: "vLLM worker-hook capture installs hooks on a decoder layer, so a point with no layer has no module to hang off.",
    };
  }
  return null;
}

/**
 * The attention trio, which is `capture_attention` on both backends: no module
 * boundary holds a score matrix on either, so both reconstruct it, and one call
 * serves scores, probs and value off the same pass. Only the key differs between
 * the two pattern points.
 */
function pattern(
  variant: Variant,
  node: Located,
  load: string,
  form: Form,
): Reading {
  const key = node.point === "attn_scores" ? "scores" : "probs";
  const note =
    variant === "eager"
      ? 'Rebuilt from the real softmax, so the model has to be loaded with attn_implementation="eager" — which the eager backend does.'
      : vllmReason(node.point);

  if (form === "await") {
    return {
      code: [
        "from interp_engine import load_model",
        "",
        load,
        "await model.warmup()",
        `ids = model.to_tokens("${PROMPT}")[0].tolist()`,
        `out = await model.capture_attention(ids, [${node.layer}])`,
        `out[${node.layer}]["${key}"]  # [n_heads, dest, src]`,
      ].join("\n"),
      note,
    };
  }

  return {
    code: [
      "from interp_engine import capture_attention, load_model",
      "",
      load,
      `ids = model.to_tokens("${PROMPT}")`,
      `out = capture_attention(model, ids, [${node.layer}])`,
      `out[${node.layer}]["${key}"]  # [n_heads, dest, src]`,
    ].join("\n"),
    note,
  };
}

/** Every other point: one address through the capture. */
function activation(node: Located, load: string, form: Form): Reading {
  if (form === "await") {
    return {
      code: [
        "from interp_engine import Address, load_model",
        "",
        load,
        "await model.warmup()",
        `ids = model.to_tokens("${PROMPT}")[0].tolist()`,
        `point = ${addressCall(node)}`,
        "acts = await model.capture(ids, [point])",
        "acts[point]  # [pos, ...]",
      ].join("\n"),
      note: "The method returns a dict keyed by Address and drops the batch axis, so it is indexed with the address itself rather than a string.",
    };
  }

  return {
    code: [
      "from interp_engine import Address, capture, load_model",
      "",
      load,
      `ids = model.to_tokens("${PROMPT}")`,
      `point = ${addressCall(node)}`,
      "cache = capture(model, ids, [point])",
      "cache[point]  # [batch, pos, ...]",
    ].join("\n"),
  };
}
