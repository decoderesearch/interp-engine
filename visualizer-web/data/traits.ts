/**
 * Architectural traits: the axes along which real models differ in ways that
 * move, add or remove a hook point.
 *
 * Most of these mirror a field on interp-engine's `Quirks` dataclass in
 * `arch.py`, which is the engine's own list of what it has to branch on. The
 * rest are facts it derives from the config instead of storing, so do not expect
 * every id here to be a field name you can look up upstream.
 */

import type { Trait, TraitGroup, TraitId } from "@/lib/types";

export const TRAITS: Trait[] = [
  // --- attention ---
  {
    id: "gqa",
    label: "GQA",
    group: "attention",
    description:
      "Grouped-query attention: several query heads share one key/value head, so the key and value tensors are narrower than the query tensor.",
    exampleModels: [
      "meta-llama/Llama-3.1-8B",
      "Qwen/Qwen3-8B",
      "google/gemma-3-12b-it",
    ],
  },
  {
    id: "mla",
    label: "Latent attn",
    group: "attention",
    description:
      "Multi-head latent attention factors the query and key/value projections through a low-rank bottleneck, so there is no single q_proj or v_proj module to hook.",
    exampleModels: ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"],
  },
  {
    id: "qk_norm",
    label: "QK-norm",
    group: "attention",
    description:
      "A normalization applied to queries and keys before RoPE. Adds four capture points, per head on some families and flat on others.",
    exampleModels: [
      "Qwen/Qwen3-8B",
      "google/gemma-3-12b-it",
      "allenai/Olmo-3-1025-7B",
    ],
  },
  {
    id: "gated_attn_out",
    label: "Gated attn out",
    group: "attention",
    description:
      "A sigmoid gate on the attention output, projected at double width. Its presence means probs @ value no longer equals z.",
    exampleModels: ["Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3.5-9B-Base"],
  },
  {
    id: "attn_sinks",
    label: "Attn sinks",
    group: "attention",
    description:
      "A learned per-head sink absorbs probability mass, so the captured attention pattern does not sum to one and must never be renormalized.",
    exampleModels: ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
  },
  {
    id: "sliding_window",
    label: "Sliding window",
    group: "attention",
    description:
      "Most layers attend only to a local window, with full-context layers interleaved at a fixed ratio. The attention pattern's shape differs by layer.",
    exampleModels: [
      "google/gemma-3-12b-it",
      "mistralai/Mistral-7B-v0.3",
      "openai/gpt-oss-20b",
    ],
    minLayers: 4,
  },
  {
    id: "hybrid_linear_attn",
    label: "Hybrid linear",
    group: "attention",
    description:
      "Some layers replace softmax attention with a linear-attention or state-space mixer. Every attention point is refused on those layers.",
    exampleModels: [
      "Qwen/Qwen3-Next-80B-A3B-Instruct",
      "LiquidAI/LFM2-8B-A1B",
      "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
    ],
    minLayers: 4,
  },
  {
    id: "fused_qkv",
    label: "Fused QKV",
    group: "attention",
    description:
      "Query, key and value share one projection matrix, so there is no standalone v_proj module and the value tensor has to be sliced back out with the family's exact layout.",
    exampleModels: [
      "openai-community/gpt2",
      "microsoft/Phi-3-mini-4k-instruct",
      "bigscience/bloom-560m",
    ],
  },

  // --- norms ---
  {
    id: "sandwich_norms",
    label: "Sandwich norms",
    group: "norms",
    description:
      "A norm after each sublayer as well as before it. This is the one trait that changes what a TransformerLens name means: hook_mlp_out is the post-norm contribution here and the raw module output everywhere else.",
    exampleModels: [
      "google/gemma-2-9b-it",
      "google/gemma-3-12b-it",
      "allenai/Olmo-3-1025-7B",
    ],
  },
  {
    id: "no_pre_attn_norm",
    label: "Post-norm only",
    group: "norms",
    description:
      "No pre-attention norm at all, so attention reads the raw residual stream and attn_in is the same tensor as resid_pre.",
    exampleModels: ["allenai/OLMo-2-0425-1B", "allenai/Olmo-3-1025-7B"],
    implies: ["sandwich_norms"],
  },
  {
    id: "logit_softcapping",
    label: "Logit softcap",
    group: "norms",
    description:
      "A tanh squashing applied to the final logits, and on some families to the attention scores too. A lens has to apply it explicitly or its numbers will not match the model's.",
    exampleModels: ["google/gemma-2-9b-it", "google/gemma-2-27b"],
  },
  {
    id: "residual_multipliers",
    label: "Residual mult",
    group: "norms",
    description:
      "Each sublayer's contribution is scaled by a constant before it rejoins the residual stream, so the raw module output and the contribution differ by a factor.",
    exampleModels: [
      "ibm-granite/granite-3.3-2b-instruct",
      "openbmb/MiniCPM3-4B",
    ],
  },

  // --- mlp ---
  {
    id: "gated_mlp",
    label: "Gated MLP",
    group: "mlp",
    description:
      "A SwiGLU-style MLP with separate gate and up projections multiplied together. Without it there is no mlp_pre_linear point to capture.",
    exampleModels: [
      "meta-llama/Llama-3.1-8B",
      "Qwen/Qwen3-8B",
      "mistralai/Mistral-7B-v0.3",
    ],
  },
  {
    id: "moe",
    label: "MoE",
    group: "mlp",
    description:
      "A router picks a few experts per token out of many. Adds three routing points and removes the neuron-basis points, because the expert bank is fused into one tensor.",
    exampleModels: [
      "Qwen/Qwen3-30B-A3B",
      "mistralai/Mixtral-8x7B-Instruct-v0.1",
      "openai/gpt-oss-20b",
    ],
    minLayers: 2,
  },
  {
    id: "shared_experts",
    label: "Shared experts",
    group: "mlp",
    description:
      "One or more experts run on every token regardless of routing, alongside the routed ones.",
    exampleModels: [
      "deepseek-ai/DeepSeek-V3",
      "Qwen/Qwen3-Next-80B-A3B-Instruct",
    ],
    implies: ["moe"],
  },

  // --- residual ---
  {
    id: "parallel_attn_mlp",
    label: "Parallel blocks",
    group: "residual",
    description:
      "Attention and the MLP both read the same input and both add to the same residual stream, so there is no resid_mid between them.",
    exampleModels: ["CohereLabs/c4ai-command-r-v01", "microsoft/phi-2"],
    conflicts: ["sandwich_norms"],
  },
  {
    id: "multi_residual_streams",
    label: "Hyper-connections",
    group: "residual",
    description:
      "The trunk carries several residual streams side by side, so no single tensor between blocks is the residual stream. Every resid point needs a stream coordinate to name one.",
    exampleModels: ["deepseek-ai/DeepSeek-V4"],
    minLayers: 2,
  },
];

export const TRAIT_GROUPS: { id: TraitGroup; label: string }[] = [
  { id: "attention", label: "Attention" },
  { id: "norms", label: "Norms" },
  { id: "mlp", label: "MLP" },
  { id: "residual", label: "Residual" },
];

const BY_ID = new Map(TRAITS.map((t) => [t.id, t]));

export function trait(id: TraitId): Trait {
  const found = BY_ID.get(id);
  if (!found) throw new Error(`unknown trait: ${id}`);
  return found;
}

/**
 * Apply a trait set's own `implies` and `conflicts` until it stops changing.
 * Turning on a trait wins over anything it conflicts with, so selecting a
 * post-norm architecture cannot leave a stale parallel-block toggle behind.
 */
export function resolveTraits(selected: Iterable<TraitId>): Set<TraitId> {
  const out = new Set<TraitId>(selected);
  let changed = true;
  while (changed) {
    changed = false;
    for (const id of [...out]) {
      for (const implied of trait(id).implies ?? []) {
        if (!out.has(implied)) {
          out.add(implied);
          changed = true;
        }
      }
    }
  }
  for (const id of [...out]) {
    for (const clash of trait(id).conflicts ?? []) {
      if (out.has(clash) && out.has(id)) out.delete(clash);
    }
  }
  return out;
}

/** The layer-count floor the active traits impose. */
export function minLayersFor(
  traits: Set<TraitId>,
  windowRatio: number,
): number {
  let floor = 2;
  for (const id of traits) {
    floor = Math.max(floor, trait(id).minLayers ?? 2);
  }
  // A 5:1 interleave needs six layers before a single global layer appears.
  if (traits.has("sliding_window")) floor = Math.max(floor, windowRatio + 1);
  return floor;
}
