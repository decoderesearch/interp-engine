/**
 * The canonical point table, transcribed from `interp_engine/points.py`.
 *
 * Order is forward order, and it is load-bearing: the diagram lays points out
 * left to right in exactly this sequence. `scope`, `width`, `vllm` and
 * `vllmNote` mirror the `PointSpec` rows upstream; `role`, `description` and
 * the trait gating are this app's additions.
 *
 * The `vllm` column is transcribed from the engine's **main branch**, not from
 * the 1.0.1 wheel this repo pins and captures against. Sixteen points are
 * hookable there and refused by 1.0.1 — the QK-norm four, `embeddings`,
 * `attn_in`, `mlp_act`, `router_logits`, `final_norm`, and all seven
 * hyper-connection rows — and `attn_scores` became a recompute rather than a
 * refusal. So a card here can show a vLLM call that
 * raises on the released engine, and the two only converge when the engine
 * releases and the pin moves. Nothing checks this: the transcription is by
 * hand, and the version it describes is a fact about this comment.
 */

import type { PointSpec } from "@/lib/types";

export const POINTS: PointSpec[] = [
  {
    name: "embeddings",
    scope: "global",
    width: "d_model",
    role: "global",
    description: "The embedding module's output, before any block runs.",
    vllm: "hooks",
  },

  // --- per layer, in the order the forward pass reaches them ---
  {
    name: "resid_pre",
    scope: "layer",
    width: "d_model",
    role: "resid",
    description: "The residual stream as the block receives it.",
    vllm: "hooks",
  },
  {
    name: "attn_in",
    scope: "layer",
    width: "d_model",
    role: "attn",
    description: "The normed residual entering attention.",
    vllm: "hooks",
    mergedWith: "resid_pre",
    mergedUnder: "no_pre_attn_norm",
  },
  {
    name: "q_norm_in",
    scope: "layer",
    width: "heads",
    role: "attn",
    description: "Query projection output, before QK-norm.",
    vllm: "hooks",
    requires: ["qk_norm"],
  },
  {
    name: "q_norm_out",
    scope: "layer",
    width: "heads",
    role: "attn",
    description: "Queries after QK-norm, gain multiply included, before RoPE.",
    vllm: "hooks",
    requires: ["qk_norm"],
  },
  {
    name: "k_norm_in",
    scope: "layer",
    width: "kv_heads",
    role: "attn",
    description:
      "Key projection output, before QK-norm. KV-head wide under GQA.",
    vllm: "hooks",
    requires: ["qk_norm"],
  },
  {
    name: "k_norm_out",
    scope: "layer",
    width: "kv_heads",
    role: "attn",
    description: "Keys after QK-norm. KV-head wide under GQA.",
    vllm: "hooks",
    requires: ["qk_norm"],
  },
  {
    name: "value",
    scope: "layer",
    width: "kv_heads",
    role: "attn",
    description:
      "The value tensor attention multiplies the pattern by, before heads are repeated for GQA. The value projection's output on most families; the output of the norm after it on a family that norms its values (Gemma 4).",
    vllm: "hooks",
    refusedBy: {
      mla: "Latent attention expands the compressed key/value latent inside the forward, so no module's output is the value. Capture the latent itself, or use z, which exists either way.",
    },
  },
  {
    name: "attn_scores",
    scope: "layer",
    width: "scores",
    role: "attn",
    description: "Pre-softmax attention scores.",
    vllm: "recompute",
    vllmNote:
      "no module boundary holds the pre-softmax scores on either backend, and the paged kernel never forms the matrix at all; rebuilt alongside attn_probs from the same captured post-RoPE q/k, which is the tensor the softmax is taken over",
  },
  {
    name: "attn_probs",
    scope: "layer",
    width: "scores",
    role: "attn",
    description: "The attention pattern, after softmax.",
    vllm: "recompute",
    vllmNote:
      "fused paged attention never materializes the probabilities; rebuilt from captured post-RoPE q/k with the checkpoint's own window, softcap and sinks reapplied",
  },
  {
    name: "z",
    scope: "layer",
    width: "heads",
    role: "attn",
    description:
      "Per-head attention output, before the output projection. Flattened.",
    vllm: "hooks",
  },
  {
    name: "attn_gate",
    scope: "layer",
    width: "heads",
    role: "attn",
    description:
      "The raw double-width gating projection. Its presence breaks the identity probs @ value == z.",
    vllm: "none",
    vllmNote:
      "unimplemented: q_proj is a real module on both trees; double width, so rank-sliced under TP",
    requires: ["gated_attn_out"],
  },
  {
    name: "attn_out",
    scope: "layer",
    width: "d_model",
    role: "attn",
    description: "The attention module's raw output, on every architecture.",
    vllm: "hooks",
  },
  {
    name: "attn_out_post",
    scope: "layer",
    width: "d_model",
    role: "attn",
    description:
      "Attention's contribution to the residual stream. Distinct from the raw output only where a post-sublayer norm sits between them.",
    vllm: "hooks",
    mergedWith: "attn_out",
    distinctUnder: "sandwich_norms",
  },
  {
    name: "resid_mid",
    scope: "layer",
    width: "d_model",
    role: "resid",
    description:
      "The residual stream between the two sublayers, read off the pre-MLP norm's input rather than reconstructed.",
    vllm: "hooks",
    refusedBy: {
      parallel_attn_mlp:
        "A parallel block adds both sublayers to the same residual, so there is no tensor between them.",
    },
  },
  {
    name: "mlp_in",
    scope: "layer",
    width: "d_model",
    role: "mlp",
    description: "The normed residual entering the MLP.",
    vllm: "hooks",
  },
  {
    name: "mlp_pre",
    scope: "layer",
    width: "neurons",
    role: "mlp",
    description:
      "Pre-activation neurons. The gate projection's output on a gated MLP.",
    vllm: "none",
    vllmNote:
      "unreachable as a module boundary on vLLM: it fuses gate_proj and up_proj into one gate_up_proj, so neither branch is a module output — and being d_mlp wide it is also interleaved per rank under TP, so rank 0's slice cannot be reassembled on its own",
    refusedBy: {
      moe: "The neuron basis is not a single tensor on a sparse layer; the expert bank is fused.",
    },
  },
  {
    name: "mlp_pre_linear",
    scope: "layer",
    width: "neurons",
    role: "mlp",
    description:
      "The up projection's output, the branch the gate multiplies into.",
    vllm: "none",
    vllmNote: "see mlp_pre; gated MLPs only",
    requires: ["gated_mlp"],
    refusedBy: {
      moe: "The neuron basis is not a single tensor on a sparse layer; the expert bank is fused.",
    },
  },
  {
    name: "mlp_act",
    scope: "layer",
    width: "neurons",
    role: "mlp",
    description:
      "Post-activation neurons, the down projection's input. This is the neuron basis SAEs are trained on.",
    vllm: "hooks",
    refusedBy: {
      moe: "The neuron basis is not a single tensor on a sparse layer; the expert bank is fused.",
    },
  },
  {
    name: "router_logits",
    scope: "layer",
    width: "routing",
    role: "route",
    description: "The router's scores over every expert, before top-k.",
    vllm: "hooks",
    requires: ["moe"],
  },
  {
    name: "expert_weights",
    scope: "layer",
    width: "routing",
    role: "route",
    description:
      "The k selected weights, renormalized as the checkpoint does it, in the router's ranking order.",
    vllm: "none",
    vllmNote:
      "unreachable: the top-k happens inside the FusedMoE kernel, which takes the logits and returns the combined output with the selection never materialized",
    requires: ["moe"],
  },
  {
    name: "expert_indices",
    scope: "layer",
    width: "routing",
    role: "route",
    description: "Which experts each token was routed to. Integer-valued.",
    vllm: "none",
    vllmNote:
      "see expert_weights; integer-valued, so it is the one point that is not a differentiable activation",
    requires: ["moe"],
  },
  {
    name: "mlp_out",
    scope: "layer",
    width: "d_model",
    role: "mlp",
    description: "The MLP module's raw output, on every architecture.",
    vllm: "hooks",
    refusedBy: {
      dense_mlp_beside_experts:
        "Here the routed experts sit beside the dense MLP rather than replacing it, and the block sums the two branches, so the MLP module's output is half the feed-forward — at the full d_model width, with nothing about it to notice. Read mlp_out_post, which is downstream of the sum.",
    },
  },
  {
    name: "mlp_out_post",
    scope: "layer",
    width: "d_model",
    role: "mlp",
    description:
      "The MLP's contribution to the residual stream. Distinct from the raw output only where a post-sublayer norm sits between them.",
    vllm: "hooks",
    mergedWith: "mlp_out",
    distinctUnder: "sandwich_norms",
  },
  {
    name: "resid_post",
    scope: "layer",
    width: "d_model",
    role: "resid",
    description: "The residual stream as the block hands it on.",
    vllm: "hooks",
  },

  // --- global, after the last block ---
  {
    name: "final_norm",
    scope: "global",
    width: "d_model",
    role: "global",
    description: "The trunk's final norm, applied to the last residual stream.",
    vllm: "hooks",
  },
  {
    name: "lm_head",
    scope: "global",
    width: "vocab",
    role: "global",
    description:
      "The bare unembed. TransformerLens returns logits instead of hooking this.",
    vllm: "none",
    vllmNote:
      "unreachable as a bare unembed: vLLM's compute_logits can fold scaling and softcapping, so what it returns is not this point",
  },
];

/**
 * The hyper-connection points, from `HYPER_CONNECTION_POINTS` in `points.py`.
 * These are not variations on a shared point — that is what the `stream`
 * coordinate is for — but tensors with no counterpart on a conventional trunk.
 *
 * `requires` is the whole of the gate, upstream as well: `points_for` keys these
 * on the trunk it found streams on, never on the architecture name. A second
 * family shipping the shape would get these rows without anything here being
 * edited, whatever it calls the modules behind them.
 *
 * All seven are served on vLLM, and none of them by a module hook: five are
 * locals of a decoder layer's forward, reached by wrapping the fused mHC kernel
 * calls, and two are elements of the layer's return. `hooks` is still the right
 * value — upstream's `HOOKS` means the worker returns the point itself, however
 * it got hold of it — but it is why these rows moved without a version bump, and
 * `docs/ENGINE_HOOK_MAPPINGS.md` is where the mechanism per row is written.
 */
export const HYPER_CONNECTION_POINTS: PointSpec[] = [
  {
    name: "resid_streams",
    scope: "layer",
    width: "streams",
    role: "resid",
    description:
      "All parallel residual streams at once, before the block reads them.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "attn_stream_collapse",
    scope: "layer",
    width: "d_model",
    role: "attn",
    description:
      "The single vector attention actually reads, collapsed from the streams. This is what a steering vector or SAE wants.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "attn_stream_write",
    scope: "layer",
    width: "streams",
    role: "attn",
    description: "Attention's output distributed back across the streams.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "attn_stream_mix",
    scope: "layer",
    width: "streams",
    role: "attn",
    description:
      "The learned mixing weights across streams on the attention side.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "mlp_stream_collapse",
    scope: "layer",
    width: "d_model",
    role: "mlp",
    description:
      "The single vector the MLP actually reads, collapsed from the streams.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "mlp_stream_write",
    scope: "layer",
    width: "streams",
    role: "mlp",
    description: "The MLP's output distributed back across the streams.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
  {
    name: "mlp_stream_mix",
    scope: "layer",
    width: "streams",
    role: "mlp",
    description: "The learned mixing weights across streams on the MLP side.",
    vllm: "hooks",
    requires: ["multi_residual_streams"],
  },
];

export const ALL_POINTS: PointSpec[] = [...POINTS, ...HYPER_CONNECTION_POINTS];

const BY_NAME = new Map(ALL_POINTS.map((p) => [p.name, p]));

export function pointSpec(name: string): PointSpec | undefined {
  return BY_NAME.get(name);
}

/**
 * Why a point's vLLM path is limited, with any `see <other>` followed —
 * `reason()` from `points.py`, which resolves the indirection rather than
 * printing it. The table writes `see mlp_pre` so that four QK-norm rows do not
 * restate one paragraph; a reader looking at one point is owed the paragraph.
 *
 * Undefined where upstream has no note, which is the points vLLM serves whole.
 */
export function vllmReason(name: string): string | undefined {
  const note = BY_NAME.get(name)?.vllmNote;
  if (!note?.startsWith("see ")) return note;
  const [referent, extra] = splitOnce(note.slice("see ".length), ";");
  const target = BY_NAME.get(referent.trim())?.vllmNote;
  if (target === undefined) return note;
  return `as ${referent.trim()}: ${target}${extra ? `; ${extra.trim()}` : ""}`;
}

function splitOnce(text: string, separator: string): [string, string] {
  const at = text.indexOf(separator);
  return at === -1
    ? [text, ""]
    : [text.slice(0, at), text.slice(at + separator.length)];
}
