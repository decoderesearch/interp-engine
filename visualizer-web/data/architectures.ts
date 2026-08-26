/**
 * Architecture presets. Keyed by HF architecture class, matching the ids in the
 * validator's `comparison/sweep_architectures.json` — except where one class
 * covers two wirings, which so far is Gemma 4 alone: its 26B declares the same
 * `Gemma4ForConditionalGeneration` as the dense SKUs and switches the experts on
 * with a config flag. That preset takes a synthetic `id` and names the real
 * class in `hfClass`, so the URL key stays stable and the class stays recorded.
 *
 * Selecting one only sets traits. Layer, head and neuron counts stay wherever
 * the user put them, except where a trait imposes a floor — the diagram is a
 * schematic of how a family is wired, not a scale model of a checkpoint.
 *
 * Model ids are full HF repo ids. `displayModel` strips the owner for the UI,
 * but the id in the data stays whole so it can be looked up.
 */

import type { Architecture } from "@/lib/types";

export const CUSTOM_ARCHITECTURE_ID = "__custom__";

const FAMILIES: Architecture[] = [
  {
    id: "LlamaForCausalLM",
    label: "Llama",
    released: "2023-02",
    significance:
      "Llama settled the modern decoder into one shape — RMSNorm, rotary embeddings and a gated SwiGLU MLP — and released the weights, so everyone after it was either copying that block or explaining why not. It put GPT-3-class quality on hardware researchers actually own, and made its own layout the thing every interpretability tool assumes by default.",
    traits: ["gqa", "gated_mlp"],
    exampleModels: [
      "meta-llama/Llama-3.1-8B",
      "meta-llama/Llama-3.3-70B-Instruct",
      "HuggingFaceTB/SmolLM3-3B",
    ],
    note: "The plain modern decoder, and the shape most tooling assumes.",
  },
  {
    id: "Llama4ForConditionalGeneration",
    label: "Llama 4",
    released: "2025-04",
    significance:
      "Llama 4 moved the family to sparse routing with a shared expert always on, and interleaved chunked local attention with global layers that carry no positional encoding at all. Dropping position from the global layers is what let it claim context lengths far past anything it trained on, without the extrapolation collapse that usually follows.",
    traits: ["gqa", "gated_mlp", "moe", "shared_experts", "sliding_window"],
    exampleModels: [
      "meta-llama/Llama-4-Scout-17B-16E-Instruct",
      "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    ],
    note: "Chunked local attention with un-RoPE'd global layers interleaved, and the text decoder nested under a multimodal wrapper.",
  },
  {
    id: "Qwen2ForCausalLM",
    label: "Qwen2.5",
    released: "2024-09",
    significance:
      "Qwen2.5 changed almost nothing about the block and spent its effort on data and scale instead, shipping a complete ladder from 0.5B to 72B wired identically at every rung. That sameness is the contribution: it is the family to reach for when an experiment has to hold architecture fixed and vary only size.",
    traits: ["gqa", "gated_mlp"],
    exampleModels: ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
  },
  {
    id: "Qwen3ForCausalLM",
    label: "Qwen3",
    released: "2025-04",
    significance:
      "Qwen3 normalized queries and keys before RoPE, so attention logits cannot grow without bound and the loss spikes that plagued large-batch training stop happening. It bought training stability at almost no inference cost, and it is now standard enough that a new family without QK-norm is the surprising one.",
    traits: ["gqa", "gated_mlp", "qk_norm"],
    exampleModels: ["Qwen/Qwen3-8B", "Qwen/Qwen3-32B", "Qwen/Qwen3-1.7B"],
  },
  {
    id: "Qwen3MoeForCausalLM",
    label: "Qwen3 MoE",
    released: "2025-04",
    significance:
      "Qwen3 MoE kept the dense block untouched and swapped only the MLP for a bank of 128 experts routed eight at a time, proving the two choices are independent. It let a 235B model serve at roughly the cost of a 22B one, which is what made sparse routing the default for open flagships rather than a specialty.",
    traits: ["gqa", "gated_mlp", "qk_norm", "moe"],
    exampleModels: ["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B"],
  },
  {
    id: "Qwen3NextForCausalLM",
    label: "Qwen3-Next",
    released: "2025-09",
    significance:
      "Qwen3-Next replaced three of every four attention layers with a gated delta net and put a sigmoid gate on what attention does return, leaving only a quarter of the stack quadratic in sequence length. Long context got dramatically cheaper to serve — and three layers in four stopped having an attention pattern to look at.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "gated_attn_out",
      "hybrid_linear_attn",
      "moe",
      "shared_experts",
    ],
    exampleModels: ["Qwen/Qwen3-Next-80B-A3B-Instruct"],
    note: "Three linear-attention layers for every softmax one.",
  },
  {
    id: "Qwen3_5ForConditionalGeneration",
    label: "Qwen3.5",
    released: "2026-02",
    significance:
      "Qwen3.5 took the hybrid trunk out of a one-off preview and made it the whole family's default, native multimodal from the start, with layer 0 itself a linear mixer rather than attention. It made long agentic runs affordable at every size — and it is why reading a model's architecture off layer 0 now reports the wrong answer.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "gated_attn_out",
      "hybrid_linear_attn",
    ],
    exampleModels: [
      "Qwen/Qwen3.5-9B-Base",
      "Qwen/Qwen3.5-27B",
      "Qwen/Qwen3.6-27B",
    ],
    note: "Qwen3-Next's hybrid trunk carried into a dense family: layer 0 is a gated delta net, so the first softmax layer is not layer 0 and anything read off layer 0 reports the wrong architecture.",
  },
  {
    id: "Qwen3_5MoeForConditionalGeneration",
    label: "Qwen3.5 MoE",
    released: "2026-02",
    significance:
      "Qwen3.5 MoE stacks sparse routing on top of the hybrid trunk, activating 17B of 397B parameters per token while three quarters of its layers avoid quadratic attention entirely. The two savings compound rather than overlap, which is how it reaches frontier benchmarks at something close to mid-size dense serving cost.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "gated_attn_out",
      "hybrid_linear_attn",
      "moe",
    ],
    exampleModels: ["Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.6-35B-A3B"],
    note: "Renormalizes the router weights unconditionally, where Qwen3-MoE only does so under norm_topk_prob.",
  },
  {
    id: "MistralForCausalLM",
    label: "Mistral",
    released: "2023-09",
    significance:
      "Mistral brought sliding-window attention into a mainstream open model, letting most layers see only a local span instead of the whole sequence. The KV cache stops growing once the window fills, which made long inputs affordable and got a 7B model beating 13B ones on most benchmarks.",
    traits: ["gqa", "gated_mlp", "sliding_window"],
    exampleModels: [
      "mistralai/Mistral-7B-v0.3",
      "mistralai/Mistral-Small-24B-Instruct-2501",
    ],
  },
  {
    id: "MixtralForCausalLM",
    label: "Mixtral",
    released: "2023-12",
    significance:
      "Mixtral was the first sparse mixture of experts anyone outside a large lab could download and run, routing every token to two of eight experts. It matched models several times its activated size, which is what turned sparsity from an interesting result into the assumed design for open flagships.",
    traits: ["gqa", "gated_mlp", "sliding_window", "moe"],
    exampleModels: ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
  },
  {
    id: "Gemma2ForCausalLM",
    label: "Gemma 2",
    released: "2024-06",
    significance:
      "Gemma 2 put a norm on both sides of every sublayer and softcapped the logits and attention scores with a tanh, taming the outliers that destabilize small models. It made distillation into 2B and 9B models work far better than training them directly — and it is the one change that redefines what TransformerLens' hook_mlp_out actually holds.",
    traits: [
      "gqa",
      "gated_mlp",
      "sandwich_norms",
      "sliding_window",
      "logit_softcapping",
    ],
    exampleModels: [
      "google/gemma-2-9b-it",
      "google/gemma-2-2b",
      "google/gemma-2-27b",
    ],
    note: "The family where TransformerLens' hook_mlp_out stops meaning the raw module output.",
  },
  {
    id: "Gemma3ForCausalLM",
    label: "Gemma 3",
    released: "2025-03",
    significance:
      "Gemma 3 traded softcapping for QK-norm and pushed the interleave to five sliding-window layers for every full-context one. Only a sixth of the stack pays the quadratic cost, which is how a 27B model reaches 128K context on a single GPU.",
    traits: ["gqa", "gated_mlp", "qk_norm", "sandwich_norms", "sliding_window"],
    exampleModels: [
      "google/gemma-3-12b-it",
      "google/gemma-3-4b-it",
      "google/gemma-3-270m",
    ],
  },
  {
    id: "Gemma4ForConditionalGeneration",
    label: "Gemma 4",
    released: "2026-04",
    significance:
      "Gemma 4 stopped treating a layer as a copy of its neighbours: attention width varies by layer type, and on the edge models a layer past a cutoff reuses an earlier layer's keys and values rather than projecting its own. Cutting that much KV cache and that many parameters is what fits an effective-2B model into about a gigabyte on a phone, at the cost of twenty of its thirty-five layers having no value tensor at all.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "sandwich_norms",
      "sliding_window",
      "logit_softcapping",
    ],
    exampleModels: [
      "google/gemma-4-E2B",
      "google/gemma-4-E4B",
      "google/gemma-4-31B",
    ],
    note: "Softcapping is back after Gemma 3 dropped it. Head width varies by layer, and two things this diagram does not draw: E2B and E4B reuse an earlier layer's keys and values past a cutoff (20 of E2B's 35 layers have no v_proj), while the 31B instead sets attention_k_eq_v, so its full-attention layers use the key projection as the value and have no v_proj either.",
  },
  {
    id: "Gemma4Moe",
    hfClass: "Gemma4ForConditionalGeneration",
    label: "Gemma 4 MoE",
    released: "2026-04",
    significance:
      "Gemma 4's 26B does not swap its MLP for an expert bank the way every other sparse family does — it keeps the dense MLP and runs 128 routed experts alongside it, summing the two branches. Eight experts fire per token for 3.8B active out of 26B, and because the dense branch stays, a sparse layer here still has a real neuron basis to read.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "sandwich_norms",
      "sliding_window",
      "logit_softcapping",
      "moe",
      "dense_mlp_beside_experts",
    ],
    exampleModels: ["google/gemma-4-26B-A4B-it"],
    note: "The one family here whose class does not identify its wiring: it declares Gemma4ForConditionalGeneration like the dense SKUs and turns the experts on with enable_moe_block. Every layer is sparse, layer 0 included, and each keeps its neuron basis — which is what the Dense MLP beside experts trait draws. One thing the diagram still gets wrong: mlp_out is shown as the whole feed-forward when it is only the dense half, because the routed branch is a sibling of layer.mlp rather than inside it, and the two are summed after.",
  },
  {
    id: "Gemma4UnifiedForConditionalGeneration",
    label: "Gemma 4 Unified",
    released: "2026-04",
    significance:
      "Gemma 4's 12B drops the vision and audio encoders entirely and projects raw image and audio input straight into the token stream with linear maps. Removing two towers takes a whole class of modality-specific plumbing out of the model, and leaves one dense trunk that reads every input the same way.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "sandwich_norms",
      "sliding_window",
      "logit_softcapping",
    ],
    exampleModels: ["google/gemma-4-12B", "google/gemma-4-12B-it"],
    note: "Encoder-free: same trunk as the dense Gemma 4, reached through Gemma4UnifiedForConditionalGeneration instead. The difference is all in front of the decoder, so the diagram is the dense one.",
  },
  {
    id: "Olmo3ForCausalLM",
    label: "OLMo 3",
    released: "2025-11",
    significance:
      "OLMo 3 went fully post-norm, dropping the pre-attention norm so attention reads the residual stream raw, and published the data, code and intermediate checkpoints alongside the weights. The reproducibility is the point: it is the strongest model whose entire recipe can be inspected and rerun rather than inferred.",
    traits: [
      "gqa",
      "gated_mlp",
      "qk_norm",
      "sandwich_norms",
      "no_pre_attn_norm",
    ],
    exampleModels: ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-1125-32B"],
    note: "Fully post-norm: attention reads the raw residual stream.",
  },
  {
    id: "GptOssForCausalLM",
    label: "gpt-oss",
    released: "2025-08",
    significance:
      "gpt-oss gave every head a learned sink that soaks up probability mass when there is nothing worth attending to, and shipped MXFP4 weights so a 120B model fits on one card. The sink holds quality steady out to very long context — and it means a captured attention pattern no longer sums to one, so renormalizing it silently corrupts the result.",
    traits: ["gqa", "gated_mlp", "moe", "attn_sinks", "sliding_window"],
    exampleModels: ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
  },
  {
    id: "LagunaForCausalLM",
    label: "Laguna",
    released: "2026-04",
    significance:
      "Laguna gates the attention output through its own projection under a softplus rather than packing the gate into a double-width query matrix, and was trained from scratch for agentic coding instead of adapted to it. At 3B activated parameters it holds its own on SWE-bench against far larger models, which puts a real coding agent on one local GPU.",
    traits: ["gqa", "gated_mlp", "gated_attn_out", "sliding_window", "moe"],
    exampleModels: ["poolside/laguna-xs.2", "poolside/laguna-xs-2.1"],
    note: "Gates the attention output from a separate g_proj under a softplus, rather than packing the gate into a double-width q_proj the way Qwen does.",
  },
  {
    id: "Phi3ForCausalLM",
    label: "Phi-3",
    released: "2024-04",
    significance:
      "Phi-3 made the case that curriculum beats scale, training on heavily filtered textbook-grade and synthetic data rather than more of the web. It packed roughly GPT-3.5 ability into 3.8B parameters and made on-device models credible, which shifted a lot of the field's effort from bigger to cleaner.",
    traits: ["gqa", "gated_mlp", "fused_qkv", "sliding_window"],
    exampleModels: ["microsoft/Phi-3-mini-4k-instruct"],
  },
  {
    id: "DeepseekV3ForCausalLM",
    label: "DeepSeek V3",
    released: "2024-12",
    significance:
      "DeepSeek V3 pushed the key/value projections through a low-rank latent bottleneck and paired that with many fine-grained experts plus an always-on shared one. Multi-head latent attention cuts the KV cache by roughly an order of magnitude, and the combination trained a 671B model for a few million dollars — but it leaves no v_proj module for tooling to hook.",
    traits: ["mla", "gated_mlp", "moe", "shared_experts"],
    exampleModels: ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1"],
  },
  {
    id: "DeepseekV4ForCausalLM",
    label: "DeepSeek V4",
    released: "2026-04",
    significance:
      "DeepSeek V4 runs several residual streams side by side and learns how each block reads from and writes to them, instead of the single stream every transformer has carried since 2017. It improves gradient flow through very deep stacks — and it invalidates the assumption underneath most interpretability tooling, that there is one residual stream to name.",
    traits: [
      "mla",
      "gated_mlp",
      "moe",
      "shared_experts",
      "multi_residual_streams",
    ],
    exampleModels: ["deepseek-ai/DeepSeek-V4"],
    note: "Hyper-connections: no single tensor between blocks is the residual stream.",
  },
  {
    id: "CohereForCausalLM",
    label: "Cohere Command",
    released: "2024-03",
    significance:
      "Cohere Command runs attention and the MLP in parallel off the same input rather than one after the other, halving the sequential depth of every block. It buys throughput at scale with no measured quality cost, and it removes resid_mid entirely, since there is no moment between the two sublayers to name.",
    traits: ["gqa", "gated_mlp", "parallel_attn_mlp"],
    exampleModels: [
      "CohereLabs/c4ai-command-r-v01",
      "CohereLabs/c4ai-command-r-plus",
    ],
  },
  {
    id: "GraniteForCausalLM",
    label: "Granite 3",
    released: "2024-10",
    significance:
      "Granite 3 scales every sublayer's contribution, the embeddings and the logits by explicit constants, making the residual stream's magnitude a design parameter rather than an accident of depth. The multipliers let one hyperparameter recipe transfer across model sizes — and they mean a module's raw output is no longer the contribution it actually makes.",
    traits: ["gqa", "gated_mlp", "residual_multipliers"],
    exampleModels: [
      "ibm-granite/granite-3.3-2b-instruct",
      "ibm-granite/granite-3.0-1b-a400m-base",
    ],
  },
  {
    id: "Starcoder2ForCausalLM",
    label: "StarCoder 2",
    released: "2024-02",
    significance:
      "StarCoder 2 kept a plain GELU MLP with no gate while everything around it moved to SwiGLU, and trained on The Stack v2 with per-repository licence provenance and an opt-out. It made a strong code model whose training data can actually be audited, which is why it keeps turning up in work that needs to say where the data came from.",
    traits: ["gqa"],
    exampleModels: ["bigcode/starcoder2-3b", "bigcode/starcoder2-7b"],
    note: "A plain GELU MLP, so no gate branch to capture.",
  },
  {
    id: "GPT2LMHeadModel",
    label: "GPT-2",
    released: "2019-02",
    significance:
      "GPT-2 established that a decoder-only transformer trained only to predict the next token picks up translation, summarization and question answering as a side effect, with no task-specific heads. It set the template every model here still follows, and being small, open and endlessly replicated is why most interpretability results are demonstrated on it first.",
    traits: ["fused_qkv"],
    exampleModels: ["openai-community/gpt2", "openai-community/gpt2-xl"],
    note: "Old, but still the workhorse of interpretability research.",
  },
  {
    id: "GPTJForCausalLM",
    label: "GPT-J",
    released: "2021-06",
    significance:
      "GPT-J paired rotary position embeddings with a parallel attention-and-MLP block, in the first credible fully-open answer to GPT-3, trained by a volunteer collective rather than a lab. Rotary embeddings won and are now universal, and GPT-J itself became the model early circuit-finding and knowledge-editing work was built on.",
    traits: ["parallel_attn_mlp"],
    exampleModels: ["EleutherAI/gpt-j-6b"],
    note: "The first widely used parallel block, and unlike its GPT-NeoX sibling it keeps q, k and v as three separate projections.",
  },
  {
    id: "GPTNeoXForCausalLM",
    label: "Pythia / GPT-NeoX",
    released: "2022-02",
    significance:
      "Pythia was built to be studied rather than deployed: eight sizes trained on identical data in identical order, with 154 checkpoints saved along the way. It made training dynamics measurable for the first time, and it is still the standard testbed for any question of the form 'when during training does this appear'.",
    traits: ["parallel_attn_mlp", "fused_qkv"],
    exampleModels: [
      "EleutherAI/pythia-70m-deduped",
      "EleutherAI/pythia-6.9b",
      "EleutherAI/gpt-neox-20b",
    ],
    note: "The checkpoint ladder most training-dynamics work runs on. Its fused QKV is interleaved per head, so slicing it in thirds silently mixes queries into the value tensor.",
  },
];

/**
 * Kept grouped by family above, where a new preset is written next to the one
 * it was copied from, and sorted here, where the reader is looking a name up
 * rather than studying a taxonomy. Numeric collation keeps Gemma 2 ahead of
 * Gemma 3, and case-insensitive collation keeps `gpt-oss` beside `GPT-2`.
 */
export const ARCHITECTURES: Architecture[] = [...FAMILIES].sort((a, b) =>
  a.label.localeCompare(b.label, "en", { sensitivity: "base", numeric: true }),
);

const BY_ID = new Map(ARCHITECTURES.map((a) => [a.id, a]));

export function architecture(id: string): Architecture | undefined {
  return BY_ID.get(id);
}

/** Months since year zero, which is only ever used to sort by. */
function releaseMonth(arch: Architecture): number {
  const [year, month] = arch.released.split("-").map(Number);
  return year * 12 + (month - 1);
}

export function releaseYear(arch: Architecture): number {
  return Number(arch.released.slice(0, 4));
}

const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/** `2023-02` reads as `Feb 2023`. */
export function formatRelease(arch: Architecture): string {
  const [year, month] = arch.released.split("-").map(Number);
  return `${MONTHS[month - 1]} ${year}`;
}

/** The same families in release order, which is the order the slider walks. */
export const BY_RELEASE: Architecture[] = [...ARCHITECTURES].sort(
  (a, b) => releaseMonth(a) - releaseMonth(b),
);

export const DEFAULT_ARCHITECTURE_ID = "Qwen3ForCausalLM";

/** `meta-llama/Llama-3.3-70B-Instruct` renders as `Llama-3.3-70B-Instruct`. */
export function displayModel(hfId: string): string {
  const slash = hfId.indexOf("/");
  return slash === -1 ? hfId : hfId.slice(slash + 1);
}

/**
 * The architecture whose trait set is exactly the one given, if any. Used to
 * snap the picker back to a named family when the toggles happen to match one.
 *
 * Families can share a trait set — Llama and Qwen2.5 differ in nothing this
 * vocabulary can express, and neither do Gemma 4 and Gemma 4 Unified, whose
 * whole difference sits in front of the decoder — so `preferred`, the one the
 * user actually picked, wins over list order for as long as its traits still
 * hold. Without it, choosing the second of a pair would silently rename itself
 * to the first.
 */
export function matchArchitecture(
  traits: Set<string>,
  preferred?: string,
): string {
  const picked = preferred === undefined ? undefined : architecture(preferred);
  if (picked && sameTraits(picked, traits)) return picked.id;
  for (const arch of ARCHITECTURES) {
    if (sameTraits(arch, traits)) return arch.id;
  }
  return CUSTOM_ARCHITECTURE_ID;
}

function sameTraits(arch: Architecture, traits: Set<string>): boolean {
  return (
    arch.traits.length === traits.size &&
    arch.traits.every((t) => traits.has(t))
  );
}
