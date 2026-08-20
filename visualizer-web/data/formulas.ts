/**
 * How each point is derived, in terms of the points before it.
 *
 * Written against what `interp-engine` states rather than what a generic
 * transformer diagram would say, so the trait-dependent cases are real ones:
 * `z` is post-gate on Qwen3-Next, the attention sink is a denominator column
 * rather than a term in the scores, and the MoE routing conventions are read
 * off the router instead of being reimplemented.
 *
 * Notation: `@` is a matmul, `⊙` is elementwise, `W_x` is a weight matrix, and
 * a bare lowercase name is a module (`input_layernorm`, `router`, `act`).
 */

import { ALL_POINTS, pointSpec } from "@/data/points";
import { trait as traitSpec } from "@/data/traits";
import type { PointName, Trait, TraitId } from "@/lib/types";

export interface Formula {
  expr: string;
  /** One line of context: what an operator is, or where the expression stops. */
  note?: string;
}

export interface FormulaContext {
  traits: Set<TraitId>;
  layer: number | null;
  stream: number | null;
  /** Whether this layer's MLP is a sparse block. */
  isMoe: boolean;
}

export function formulaFor(
  point: PointName,
  ctx: FormulaContext,
): Formula | null {
  const has = (t: TraitId) => ctx.traits.has(t);
  const { layer } = ctx;

  switch (point) {
    case "embeddings":
      return {
        expr: "embeddings = W_E[token_ids]",
        note: "The embedding module's own output. Gemma scales it by √d_model afterwards, inside the block.",
      };

    case "resid_pre":
      if (layer === 0) return { expr: "resid_pre.0 = embeddings" };
      return {
        expr: `resid_pre.${layer} = resid_post.${(layer ?? 1) - 1}`,
        note: "The same tensor under two names: one block's output is the next one's input.",
      };

    case "attn_in":
      return {
        expr: "attn_in = input_layernorm(resid_pre)",
        note: "The norm's own scale is an intermediate of its arithmetic, not a boundary, so it is not a point.",
      };

    case "q_norm_in":
      return {
        expr: "q_norm_in = W_Q @ attn_in",
        note: has("gated_attn_out")
          ? "W_Q is double width here; its second half per head is attn_gate."
          : has("fused_qkv")
            ? "Sliced out of the fused QKV projection with the family's exact layout."
            : undefined,
      };

    case "q_norm_out":
      return {
        expr: "q_norm_out = q_norm(q_norm_in)",
        note: "RoPE runs after this, inline in the attention forward, so it is not a point of its own.",
      };

    case "k_norm_in":
      return {
        expr: "k_norm_in = W_K @ attn_in",
        note: has("gqa") ? "n_kv_heads wide, not n_heads." : undefined,
      };

    case "k_norm_out":
      return { expr: "k_norm_out = k_norm(k_norm_in)" };

    case "value":
      return {
        expr: "value = W_V @ attn_in",
        note: has("fused_qkv")
          ? "There is no standalone v_proj: this is sliced out of the fused projection with the family's exact layout."
          : undefined,
      };

    case "attn_scores": {
      const q = has("qk_norm") ? "q_norm_out" : "W_Q @ attn_in";
      const k = has("qk_norm") ? "k_norm_out" : "W_K @ attn_in";
      const softcap = has("logit_softcapping") ? " (softcapped)" : "";
      const mask = has("sliding_window") ? "window_mask" : "causal_mask";
      return {
        expr: `attn_scores = RoPE(${q}) @ RoPE(${k})^T / √head_dim + ${mask}${softcap}`,
        note: has("attn_sinks")
          ? "The full pre-softmax value, including the additive mask. The sink is not in it: a sink is a column in the softmax denominator, not a term in the scores."
          : has("sliding_window")
            ? "The mask is the layer's own: the local layers see back only one window, and the recompute path has to read the width off the checkpoint rather than assume it."
            : "The full pre-softmax value: the scale, any score softcap, and the additive causal mask, so masked positions are large and negative rather than zero.",
      };
    }

    case "attn_probs":
      return has("attn_sinks")
        ? {
            expr: "attn_probs = softmax([attn_scores, sink])[…, :n_keys]",
            note: "The learned sink adds a column to the denominator, so rows do not sum to 1. Never renormalize them.",
          }
        : { expr: "attn_probs = softmax(attn_scores)" };

    case "z":
      return has("gated_attn_out")
        ? {
            expr: "z = (attn_probs @ value) ⊙ σ(attn_gate)",
            note: "Post-gate, which is exactly why the identity probs @ value == z is false on this family.",
          }
        : {
            expr: "z = attn_probs @ value",
            note: has("gqa")
              ? "value's heads are repeated out to the query head count first, so this is n_heads wide where value is n_kv_heads."
              : "The identity every layout check here rests on, which is why a family that breaks it is worth flagging.",
          };

    case "attn_gate":
      return {
        expr: "attn_gate = W_gate @ attn_in",
        note: "The raw projection. σ(·) is applied inline in the block, so there is no module to read it from.",
      };

    case "attn_out":
      return { expr: "attn_out = W_O @ z" };

    case "attn_out_post":
      return has("sandwich_norms")
        ? {
            expr: "attn_out_post = post_attention_layernorm(attn_out)",
            note: "The norm between the module output and the residual add is what makes these two different tensors.",
          }
        : {
            expr: "attn_out_post = attn_out",
            note: "Nothing sits between the module output and the residual add here, so the contribution is the raw output.",
          };

    case "resid_mid":
      return {
        expr: has("residual_multipliers")
          ? "resid_mid = resid_pre + α · attn_out_post"
          : "resid_mid = resid_pre + attn_out_post",
        note: has("residual_multipliers")
          ? "α is the family's residual multiplier, so the raw output and the contribution differ by a constant factor."
          : "Read off the pre-MLP norm's input rather than reconstructed from the two terms.",
      };

    case "mlp_in":
      return has("parallel_attn_mlp")
        ? {
            expr: "mlp_in = pre_mlp_norm(resid_pre)",
            note: "A parallel block feeds the MLP the layer input, not a mid-residual — there is no resid_mid to norm.",
          }
        : { expr: "mlp_in = pre_mlp_norm(resid_mid)" };

    case "mlp_pre":
      return has("gated_mlp")
        ? {
            expr: "mlp_pre = W_gate @ mlp_in",
            note: "HF's gate_proj. Do not translate by weight name: TransformerLens' W_in is HF's up_proj, the other branch.",
          }
        : { expr: "mlp_pre = W_in @ mlp_in" };

    case "mlp_pre_linear":
      return { expr: "mlp_pre_linear = W_up @ mlp_in" };

    case "mlp_act":
      return {
        expr: has("gated_mlp")
          ? "mlp_act = act(mlp_pre) ⊙ mlp_pre_linear"
          : "mlp_act = act(mlp_pre)",
        note: "act is whichever activation the checkpoint configures, not a fixed one.",
      };

    case "router_logits":
      return {
        expr: "router_logits = W_router @ mlp_in",
        note: "Element 0 of the router's own output tuple, not recomputed.",
      };

    case "expert_indices":
      return {
        expr: "expert_indices = topk(score(router_logits), k).indices",
        note: "score and the top-k convention differ per checkpoint, so both are read off the router rather than reimplemented. Integer-valued: this is where the model stops being continuous.",
      };

    case "expert_weights":
      return {
        expr: "expert_weights = renorm(topk(score(router_logits), k).values)",
        note: "Whether the softmax runs before or after selection, and whether the survivors are renormalized, is per checkpoint — every convention yields k weights summing to 1, so a guess would be plausible and silently wrong.",
      };

    case "mlp_out":
      if (!ctx.isMoe) return { expr: "mlp_out = W_down @ mlp_act" };
      return {
        expr: has("shared_experts")
          ? "mlp_out = shared(mlp_in) + Σ_i expert_weights[i] · expert[expert_indices[i]](mlp_in)"
          : "mlp_out = Σ_i expert_weights[i] · expert[expert_indices[i]](mlp_in)",
        note: "Tapped at the whole block, so it already includes any always-on shared expert.",
      };

    case "mlp_out_post":
      return has("sandwich_norms")
        ? { expr: "mlp_out_post = post_feedforward_layernorm(mlp_out)" }
        : {
            expr: "mlp_out_post = mlp_out",
            note: "Nothing sits between the module output and the residual add here.",
          };

    case "resid_post":
      return {
        expr: has("parallel_attn_mlp")
          ? "resid_post = resid_pre + attn_out_post + mlp_out_post"
          : "resid_post = resid_mid + mlp_out_post",
        note: has("parallel_attn_mlp")
          ? "Both sublayers read the layer input, so there is no mid-residual to add through."
          : "Equivalently resid_pre + attn_out_post + mlp_out_post, the invariant that catches a sandwich norm nobody has inspected yet.",
      };

    case "final_norm":
      return { expr: "final_norm = norm(resid_post of the last layer)" };

    case "lm_head":
      return {
        expr: "lm_head = W_U @ final_norm",
        note: has("logit_softcapping")
          ? "Gemma's c·tanh(logits/c) softcap is not applied by this module, so a lens has to apply it explicitly afterwards."
          : undefined,
      };

    // --- DeepSeek-V4 hyper-connections ---
    case "resid_streams":
      return {
        expr: "resid_streams = stack(resid_pre.stream-0 … resid_pre.stream-n)",
      };

    case "attn_stream_collapse":
      return {
        expr: "attn_stream_collapse = Σ_s attn_stream_mix[s] · resid_pre.stream-s",
        note: "This, and not any one stream, is what a steering vector or an SAE wants on a hyper-connection model.",
      };

    case "attn_stream_write":
      return {
        expr: "attn_stream_write.stream-s = attn_stream_mix[s] · attn_out",
      };

    case "attn_stream_mix":
      return {
        expr: "attn_stream_mix = hyper_connection_weights(resid_streams)",
      };

    case "mlp_stream_collapse":
      return {
        expr: "mlp_stream_collapse = Σ_s mlp_stream_mix[s] · resid_mid.stream-s",
      };

    case "mlp_stream_write":
      return {
        expr: "mlp_stream_write.stream-s = mlp_stream_mix[s] · mlp_out",
      };

    case "mlp_stream_mix":
      return {
        expr: "mlp_stream_mix = hyper_connection_weights(resid_streams)",
      };

    default:
      return null;
  }
}

/**
 * Which points read differently under one trait set than the other.
 *
 * Several traits change no geometry at all — fused QKV, attention sinks, logit
 * softcapping — so toggling them looks like nothing happened even though the
 * arithmetic behind two or three points just changed. This is what the diagram
 * pulses, and what a trait's hover card lists, so the effect is findable
 * instead of hidden a hover away.
 *
 * Compared at a mid-stack layer, since layer 0's `resid_pre` reads from the
 * embeddings rather than from a previous block and would differ for a reason
 * that has nothing to do with traits.
 */
export function formulaChanges(
  before: Set<TraitId>,
  after: Set<TraitId>,
): Set<PointName> {
  const changed = new Set<PointName>();
  for (const spec of ALL_POINTS) {
    const one = render(spec.name, before);
    const other = render(spec.name, after);
    if (one !== other) changed.add(spec.name);
  }
  return changed;
}

/**
 * What one active trait does to one point, said from the point's side.
 *
 * The pairs here are exactly the ones where `formulaFor` reads differently with
 * the trait than without it, so the diagram's ripple, a trait's "Rewrites" row
 * and this text all describe the same set. Anything a trait does *structurally*
 * — adding a point, refusing it, merging it into its neighbour — is left out,
 * because the popover already says so in the alias and refusal boxes.
 */
const TRAIT_IMPACTS: Partial<
  Record<TraitId, Partial<Record<PointName, string>>>
> = {
  gqa: {
    k_norm_in:
      "The key projection is n_kv_heads wide rather than n_heads, so this is narrower than the queries it will be scored against.",
    z: "The value's heads are repeated out to the query head count before the product, so z is wider than value.",
  },
  qk_norm: {
    attn_scores:
      "The scores are built from the normed queries and keys, so q_norm_out and k_norm_out are the inputs rather than the raw projections.",
  },
  gated_attn_out: {
    q_norm_in:
      "q_proj is double width here. Only the first half per head is the query; the second half is attn_gate, and halving the flat vector mixes the two.",
    z: "Taken after the gate, so probs @ value reconstructs the pre-gate value and no longer equals this tensor.",
  },
  attn_sinks: {
    attn_scores:
      "The sink stays out of the scores. It is an extra column in the softmax denominator, not a term in the product.",
    attn_probs:
      "That extra column joins the denominator, so a row sums to less than 1. Renormalizing it would delete the effect.",
  },
  sliding_window: {
    attn_scores:
      "The local layers add a window mask instead of the plain causal one, and a recompute has to read the width off the checkpoint rather than assume it.",
  },
  fused_qkv: {
    q_norm_in:
      "There is no q_proj module; the queries are sliced out of the shared projection.",
    value:
      "There is no v_proj module either. The value is sliced out with the family's exact layout, and slicing it the other way mixes keys into the values.",
  },
  sandwich_norms: {
    attn_out_post:
      "A post-attention norm sits between the module output and the residual add, which is what makes this a different tensor from attn_out.",
    mlp_out_post:
      "Likewise a post-feedforward norm, so mlp_out is no longer what reaches the residual stream.",
  },
  logit_softcapping: {
    attn_scores:
      "Gemma-2 softcaps the scores inside eager attention, so the cap is already here.",
    lm_head:
      "The final softcap is not applied by this module, so logits a lens can compare need c·tanh(·/c) on top.",
  },
  residual_multipliers: {
    resid_mid:
      "The block scales the sublayer's output before adding it, so the contribution and the raw output differ by a constant.",
  },
  gated_mlp: {
    mlp_pre:
      "This is the gate branch, HF's gate_proj. The up branch is its own point, and TransformerLens names the two weights the other way round.",
    mlp_act:
      "The activation runs on the gate branch and is multiplied into the up branch, rather than applied to one projection.",
  },
  moe: {
    mlp_out:
      "The block sums the experts the router selected instead of running one down projection, which is also why the neuron-basis points below it refuse.",
  },
  shared_experts: {
    mlp_out:
      "An always-on expert runs alongside the routed ones. Tapping the block rather than mlp.experts is what keeps its contribution in.",
  },
  parallel_attn_mlp: {
    mlp_in:
      "The MLP reads the layer input, so this is a normed resid_pre — there is no resid_mid for it to norm.",
    resid_post:
      "Both sublayers write to the same residual, so their contributions land together rather than in sequence.",
  },
};

export interface TraitImpact {
  trait: Trait;
  text: string;
}

/**
 * The active traits that shaped how this point reads, and how.
 *
 * `requires` is folded in mechanically: a point that only exists under a trait
 * is worth attributing even when the trait does nothing to its arithmetic.
 */
export function traitImpacts(
  points: PointName[],
  traits: Set<TraitId>,
  isMoe: boolean,
): TraitImpact[] {
  const impacts = new Map<TraitId, string>();

  for (const point of points) {
    for (const id of pointSpec(point)?.requires ?? []) {
      if (traits.has(id))
        impacts.set(id, `This point only exists under ${traitSpec(id).label}.`);
    }
  }

  for (const id of traits) {
    // The MoE pair is per layer: a hybrid stack's dense prefix runs an ordinary
    // MLP even though the trait is on.
    if ((id === "moe" || id === "shared_experts") && !isMoe) continue;
    for (const point of points) {
      const text = TRAIT_IMPACTS[id]?.[point];
      if (text) impacts.set(id, text);
    }
  }

  return [...impacts].map(([id, text]) => ({ trait: traitSpec(id), text }));
}

function render(point: PointName, traits: Set<TraitId>): string {
  const formula = formulaFor(point, {
    traits,
    layer: 1,
    stream: null,
    isMoe: traits.has("moe"),
  });
  return formula ? `${formula.expr}||${formula.note ?? ""}` : "";
}
