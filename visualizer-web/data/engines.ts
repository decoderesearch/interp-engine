/**
 * How each stack spells a canonical point.
 *
 * Transcribed from `interp_engine/mappers.py` and `docs/ENGINE_HOOK_MAPPINGS.md`.
 * Adding a fourth stack means adding one entry to `ENGINES`; nothing else in
 * the app knows how many there are.
 *
 * The one thing to preserve here: `transformerlens` is trait-aware. TL's
 * block-level `hook_mlp_out` is the sublayer's residual *contribution*, which
 * on a sandwich-norm family is a different tensor from the raw module output
 * that carries the same name everywhere else. `mappers.tlens_hook_to_point`
 * takes a model for exactly this reason, and the docs call getting it wrong
 * "the mapping mistake that produces a plausible-looking wrong number".
 */

import type { EngineId, TraitId } from "@/lib/types";

export interface EngineName {
  /** The name in this stack, or null where the stack has no name for it. */
  text: string | null;
  /** Why there is no name. Only set when `text` is null. */
  reason?: string;
  /** A name exists but means something worth knowing. */
  caveat?: string;
}

export interface Engine {
  id: EngineId;
  label: string;
  /** Short label for the segmented control, where space is tight. */
  short: string;
  format(ctx: FormatContext): EngineName;
}

export interface FormatContext {
  point: string;
  layer: number | null;
  stream: number | null;
  traits: Set<TraitId>;
}

// --- interp-engine ---------------------------------------------------------

/** `format_address` from `interp_engine/address.py`, in TypeScript. */
export function formatAddress(
  point: string,
  layer: number | null,
  stream: number | null,
): string {
  const parts = [point];
  if (layer !== null) parts.push(String(layer));
  if (stream !== null) parts.push(`stream-${stream}`);
  return parts.join(".");
}

/** The three coordinates an address carries, whether or not it has them all. */
export interface Address {
  point: string;
  layer: number | null;
  stream: number | null;
}

/**
 * `formatAddress` read backwards, for an address that arrived as text.
 *
 * Shape only. Whether `risd_mid` is a point is a question for `pointSpec`, and
 * whether `resid_mid.3` is a point *this* architecture has on *that* layer is
 * one only the graph can answer.
 */
export function parseAddress(address: string): Address | null {
  const parts = address.split(".");
  const point = parts.shift();
  if (!point) return null;

  let layer: number | null = null;
  if (parts[0] !== undefined && /^\d+$/.test(parts[0])) {
    layer = Number(parts.shift());
  }

  let stream: number | null = null;
  const streamed = parts[0]?.match(/^stream-(\d+)$/);
  if (streamed) {
    stream = Number(streamed[1]);
    parts.shift();
  }

  // Anything left over is not an address this app ever wrote, and neither is a
  // stream coordinate on a point with no layer — the trunk points that carry
  // one are all inside the loop.
  if (parts.length > 0 || (stream !== null && layer === null)) return null;
  return { point, layer, stream };
}

// --- TransformerLens -------------------------------------------------------

/** `_POINT_TO_TLENS`, emitted as `blocks.{layer}.{suffix}`. */
const POINT_TO_TLENS: Record<string, string> = {
  resid_pre: "hook_resid_pre",
  resid_post: "hook_resid_post",
  resid_mid: "hook_resid_mid",
  mlp_in: "mlp.hook_in",
  mlp_out: "mlp.hook_out",
  attn_in: "hook_attn_in",
  attn_out: "attn.hook_out",
  mlp_out_post: "hook_mlp_out",
  attn_out_post: "hook_attn_out",
  z: "attn.hook_z",
  value: "attn.hook_v",
  attn_probs: "attn.hook_pattern",
  attn_scores: "attn.hook_attn_scores",
  mlp_act: "mlp.hook_post",
  mlp_pre: "mlp.hook_pre",
  mlp_pre_linear: "mlp.hook_pre_linear",
  expert_indices: "mlp.hook_expert_indices",
  q_norm_in: "attn.q_norm.hook_in",
  q_norm_out: "attn.q_norm.hook_out",
  k_norm_in: "attn.k_norm.hook_in",
  k_norm_out: "attn.k_norm.hook_out",
};

/** `UNMAPPED_TLENS`, with the reason each one is declared rather than guessed. */
const UNMAPPED_TLENS: Record<string, string> = {
  attn_gate: "TransformerLens models no family with a gated attention output.",
  embeddings:
    "TL's hook_embed fires pre-positional and pre-scaling, so it is a different tensor.",
  final_norm: "ln_final.hook_normalized is outside the blocks.{i}. namespace.",
  lm_head: "TL returns logits rather than hooking the unembed.",
  router_logits: "TL hooks the softmax over all experts, not the logits.",
  expert_weights:
    "TL's hook_expert_weights is pre-top-k, over every expert, so it is a different tensor.",
};

const TLENS_CAVEATS: Record<string, string> = {
  z: "Shaped per head there, [batch, pos, head, d_head], against a flattened tensor here.",
  value:
    "Also fires before heads are repeated for GQA, but shaped [batch, pos, kv_head, d_head].",
  mlp_act:
    "The name collides with mlp_out_post but the tensor does not: this is d_mlp wide, not d_model.",
  mlp_pre:
    "Do not translate by weight name — TL's W_in is HF's up_proj, the other branch.",
};

function tlensName(ctx: FormatContext): EngineName {
  const { point, layer, traits } = ctx;

  const unmapped = UNMAPPED_TLENS[point];
  if (unmapped) return { text: null, reason: unmapped };

  const suffix = POINT_TO_TLENS[point];
  if (!suffix)
    return { text: null, reason: "No TransformerLens hook names this tensor." };

  if (layer === null)
    return { text: null, reason: "This point has no layer to address." };

  const text = `blocks.${layer}.${suffix}`;
  const sandwich = traits.has("sandwich_norms");

  // Without a post-sublayer norm the raw output and the contribution are one
  // tensor, so TL's two names are interchangeable and it is worth saying so.
  if (!sandwich && (point === "mlp_out" || point === "attn_out")) {
    const other = point === "mlp_out" ? "hook_mlp_out" : "hook_attn_out";
    return {
      text,
      caveat: `blocks.${layer}.${other} is the same tensor here. Turn on sandwich norms and it stops being.`,
    };
  }
  if (sandwich && (point === "mlp_out_post" || point === "attn_out_post")) {
    return {
      text,
      caveat:
        "Fires after the post-sublayer norm on this family, so it is not the raw module output.",
    };
  }

  const caveat = TLENS_CAVEATS[point];
  return caveat ? { text, caveat } : { text };
}

// --- nnterp / nnsight ------------------------------------------------------

/** `_NNSIGHT_TO_POINT`, inverted. These six accessors are the whole per-layer map. */
const POINT_TO_NNSIGHT: Record<string, string> = {
  resid_pre: "layers_input",
  resid_post: "layers_output",
  mlp_in: "mlps_input",
  mlp_out: "mlps_output",
  attn_out: "attentions_output",
  attn_in: "attentions_input",
};

/** Whole-forward quantities, which are not indexed by layer. */
const NNSIGHT_GLOBAL: Record<string, string> = {
  embeddings: "token_embeddings",
  final_norm: "ln_final.output",
  lm_head: "logits",
};

const NNSIGHT_CAVEATS: Record<string, string> = {
  attn_probs:
    "Only reachable through a source patch, not a standardized accessor.",
  attn_out:
    "Always the raw module output, with no sandwich-norm awareness anywhere in nnsight or nnterp.",
  mlp_out:
    "Always the raw module output, with no sandwich-norm awareness anywhere in nnsight or nnterp.",
};

function nnterpName(ctx: FormatContext): EngineName {
  const { point, layer } = ctx;

  const global = NNSIGHT_GLOBAL[point];
  if (global) return { text: global };

  if (point === "attn_probs" && layer !== null) {
    return {
      text: `attention_probabilities[${layer}]`,
      caveat: NNSIGHT_CAVEATS.attn_probs,
    };
  }

  const accessor = POINT_TO_NNSIGHT[point];
  if (!accessor || layer === null) {
    if (point === "resid_mid") {
      return {
        text: null,
        reason:
          "No accessor. Reachable only as the pre-MLP norm's .input, under that module's HF name.",
      };
    }
    return {
      text: null,
      reason:
        "nnterp's accessor set is closed, and it has no accessor for this tensor.",
    };
  }

  const caveat = NNSIGHT_CAVEATS[point];
  const text = `${accessor}[${layer}]`;
  return caveat ? { text, caveat } : { text };
}

// --- registry --------------------------------------------------------------

export const ENGINES: Engine[] = [
  {
    id: "interp-engine",
    label: "interp-engine",
    short: "engine",
    format: ({ point, layer, stream }) => ({
      text: formatAddress(point, layer, stream),
    }),
  },
  {
    id: "transformerlens",
    label: "TransformerLens",
    short: "tlens",
    format: tlensName,
  },
  {
    id: "nnterp",
    label: "nnterp",
    short: "nnterp",
    format: nnterpName,
  },
];

export const DEFAULT_ENGINE_ID: EngineId = "interp-engine";

const BY_ID = new Map(ENGINES.map((e) => [e.id, e]));

export function engine(id: EngineId): Engine {
  const found = BY_ID.get(id);
  if (!found) throw new Error(`unknown engine: ${id}`);
  return found;
}

/**
 * Whether the name the toggle is currently showing contains `term`, which the
 * caller has already trimmed and lowercased.
 *
 * Matched against the whole name rather than the shortened diagram label, so
 * `blocks.5` finds one layer's TransformerLens hooks. A point the current stack
 * has no name for can never match, which is the honest answer: you are
 * searching that stack's vocabulary, not the union of all three.
 */
export function nameMatches(
  id: EngineId,
  ctx: FormatContext,
  term: string,
): boolean {
  const { text } = engine(id).format(ctx);
  return text !== null && text.toLowerCase().includes(term);
}
