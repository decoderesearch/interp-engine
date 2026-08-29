# Model architecture quirks

> Every structural fact about a model family that the engine cannot get by asking the model.
> `arch.py` resolves module paths programmatically and reads dims and quirks from the HF
> `config`, so **most families need no code here at all**. This page is what is left over:
> the traps where inspection returns a shape-correct wrong answer, and the reason each
> vocabulary and table in `facts.py` looks the way it does.

**Read it in this order, and stop early.** Most of this page is the part you only need once
something looks wrong, so it is organized by _when_ you need it rather than as one list.

| you are…                                                 | read                                                                                                                                              |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| wondering where a per-model fact is allowed to live      | [the file map](#where-model-specific-config-lives), then [what resolves for free](#resolution-what-happens-with-no-code-changes)                  |
| holding a wrong number, or a model with an unusual block | [structural facts inspection cannot see](#structural-facts-inspection-cannot-see) — one section per trap, each with the invariant that detects it |
| trying to trust the result                               | [proving it](#proving-it-invariants-and-ground-truth): the invariants that a plausible-looking guess cannot satisfy                               |
| about to serve it on vLLM                                | [the fused-backend section](#if-it-will-be-served-on-vllm). Three of its four failures are silent                                                 |
| adding a _hook point_ rather than a model                | not this page: `interp_engine/points.py` is the table, and `tests/test_points_registry.py` is what fails if you half-add one                      |

## Where model-specific config lives

The engine's default answer to "how does this model work?" is **ask the model**: `arch.py` resolves
module paths by inspection, and chat structure comes from the tokenizer's own chat template. So
model-specific config is deliberately small and confined to the files below. If you're adding a
model, glance here first — in most cases nothing needs editing.

Model facts live in `facts.py` and **not** per backend. `arch.py` (eager, binds a live HF tree) and
`vllm_capture/_tree.py` / `vllm_backend.py` (vLLM worker tree and config-only client) are thin adapters
over it. Derive a fact per backend instead and a family with unusual nesting gets fixed on one while
staying broken on the other. If you find yourself adding a name or a dim to one backend, it belongs
in `facts.py` instead.

| file                       | what's in it                                                                                                                                                                                                                                                                                                                                            | when you edit it                                                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `facts.py`                 | **The single source of truth for model facts**, read by _both_ backends: the candidate attribute names for each structural role (trunk / layers / embed / final norm / attention / MLP), every config-derived dim, the per-layer window & linear-attention predicates, and the per-backend tables (`EAGER_QKV_LAYOUTS`, `ALWAYS_PARALLEL_BLOCK_ARCHS`). | When a family names a structural role something new, or has a structural fact that differs **per backend**. Anything you add here is automatically seen by eager _and_ vLLM — which is the point.   |
| `arch.py` → `KNOWN_QUIRKS` | _Structural_ facts config can't express, keyed on `config.architectures[0]`. Mostly `note` text now: attention sinks are the only remaining behavioural flag here.                                                                                                                                                                                      | Only for a structural gotcha; softcaps / `layer_types` / tied embeddings are read from `config` automatically — never hardcode those. See [Adding a quirk](#adding-a-quirk-known_quirks-in-archpy). |
| `attn_config.py`           | Which config fields the vLLM attention recompute applies (`CONSUMED`), which provably can't matter (`BENIGN`, with reasons), and which make it refuse.                                                                                                                                                                                                  | When a new model trips the tripwire — file the field in one of the three. See [the three terms](#the-three-terms-the-vllm-attention-recompute-must-reapply).                                        |
| `points.py`                | The canonical hook points as data: scope, width, vLLM support and the reason for each limit. Consumers (the vLLM served-point gate, the width guard, the reshape set, the docs footnotes) derive from it.                                                                                                                                               | **Never for a new model** — a point is architecture-independent by construction. Only when adding a _point_, and then `tests/test_points_registry.py` tells you what else needs updating.           |
| `chat_conventions.py`      | How a model structures what it **generates**: harmony markers (gpt-oss), reasoning delimiters (`REASONING_TAGS`, e.g. `<think>`/`</think>`), and generic turn-end tokens.                                                                                                                                                                               | Only when a model introduces a _new_ reasoning delimiter pair — append one `ReasoningTags` entry. Models reusing `<think>` need no change.                                                          |
| `chat_compose.py`          | Consumes the table above to turn a generation back into messages (`compose_assistant_turns`), normalizing every family's reasoning to `<think>…</think>` in the message content.                                                                                                                                                                        | Never for a new model — it reads `chat_conventions.py`. Only to change the _wire shape_ clients receive.                                                                                            |
| `chat_formatters.py`       | `CODE_CHAT_FORMATS`: the few architectures that ship **no** `chat_template` and define their prompt format in Python instead (DeepSeek-V4), mapped to a loader that imports the checkpoint's own encoder.                                                                                                                                               | Only when a model has no chat template at all. A model with a template needs no entry — that is the other 99%, and the default path below.                                                          |

Two deliberate non-entries:

- **Prompt-side chat structure needs no per-family entry, in either direction.** Going _in_,
  `Tokenize.message_spans` renders through the model's real chat template and derives per-token
  role/section by diffing renders, so every family (ChatML, Gemma, Llama, harmony) works without
  per-family code. Coming back _out_, `chat_compose.py` reads only the generated text and lets the
  caller supply the prompt messages it already holds, so no code ever parses a rendered prompt back
  into messages. Between them they replaced the frontend's per-family state machines and the
  inference app's per-model response parsers.

  `CODE_CHAT_FORMATS` is the one exception, and it is narrow on purpose: it holds only checkpoints
  that publish **no** template, where there is no render to diff and the format exists solely as
  code. Even then the entry is a _loader_, not a description — the format itself is read from the
  encoder shipped beside the weights, so the table names which architectures are affected and
  never what their prompts look like. Adding a model that _has_ a template here would be the
  mistake the paragraph above is about.
- **Nothing keys on a model-name _substring_.** Both tables above key on architecture class or on
  added-token-vocab membership, i.e. on **capability**. A substring match (`"llama-3" in
name_or_path`) is the anti-pattern this structure exists to avoid: a substring is a claim about a
  _family_, so it fires on every finetune and re-upload whose name happens to contain it, and it
  fires silently — the wrong row is used and the numbers come back wrong rather than absent.

  An **exact repo id** is a different thing and is allowed where the fact really is about one
  published checkpoint rather than a family — "this checkpoint's own bf16 arithmetic explains this
  measurement" is a claim about those weights and nothing else. What an id match buys is bounded in
  the other direction, though: it cannot fire
  on a mirror, a local copy or a revision of the same weights, so it fits facts that can only be
  _observed_ on the checkpoint you measured, and it is the wrong tool for anything inspection can
  answer. When in doubt, ask the model — `EagerModel`'s native-vs-bundled code choice
  (`resolve_trust_remote_code`) is the worked example: four checkpoints needed fixing and it names
  none of them, because `transformers` can be asked which of the two cases a checkpoint is in and
  the answer then covers mirrors too.

## Resolution: what happens with no code changes

### How `arch.py` resolves a model (so you know when to leave it alone)

`resolve_arch(model, config)` returns an `ArchSpec` of **live `nn.Module` handles** into the
loaded model. Because hooks attach to the _real_ submodules, `transformers` applies every
architecture detail (RoPE, RMSNorm offset, embed scaling, softcaps, masks) inside `forward()`
for free. Resolution is:

- **Trunk**: BFS through `model` / `language_model` / `text_model` / `transformer` / `gpt_neox`
  / `decoder` for the first module holding a `layers` / `h` / `blocks` list. Handles plain
  decoder-only, `model.model` double-nesting, and multimodal `*ForConditionalGeneration` (text
  stack under `model.language_model`, next to vision/audio towers).
- **embed / final_norm / lm_head**: first matching leaf name (`embed_tokens`/`wte`/…,
  `norm`/`ln_f`/…, `lm_head`/`embed_out`), scoped to the trunk subtree so it never grabs a
  vision tower's norm.
- **dims**: from `config` (or `get_text_config()` for multimodal — top-level dims are `None`
  there). `head_dim` falls back to `d_model // n_heads` but **prefers `config.head_dim`** (Gemma
  sets it explicitly and it is NOT `d_model/n_heads`).

Every candidate name list above comes from `facts.py`, and the vLLM worker walks the _same_ lists
against vLLM's own module tree. So adding a name teaches both backends at once.

If all of the above resolves, you need no code change.

### Adding a spelling to a `facts.py` vocabulary

**A spelling generic enough to name something else needs narrower scoping, not a comment.** Most entries
in these vocabularies say what the module _is_ (`embed_tokens`, `q_proj`, `post_feedforward_layernorm`),
so a collision would take two families using one name for two roles. InternLM2 calls its unembed
`output`, which is also what any number of nested sublayers call themselves — and since the `lm_head`
search falls back to walking the whole trunk for a nested head, the walk would return the first
`something.output` at any depth. It lives in `facts.LM_HEAD_ROOT_ONLY_ATTRS`, consulted only as a direct
child of the model root, rather than in the list the subtree walk uses. Ask which lookups a name is
exposed to before adding it, because unembedding through a `d_model → d_model` matrix raises nothing.

A name gap and an architectural absence look nothing alike, and the difference is what a failed lookup
should tell you: a point that does not resolve must be refused with a `ValueError` that explains why —
a fused `gate_up_proj`, a sparse MLP, a parallel block, MLA with no separable value. An `AttributeError`
means a module we failed to _name_, which is the bug these vocabularies exist to prevent and which
nothing downstream catches: the family loads, the block points work, and one point is silently missing
across a whole architecture. That is how BLOOM, Falcon, OPT, MPT, phi-2 and the Granite-MoE families
came to be uncovered while looking supported.

A fourth of what vLLM serves has no `transformers` class at all (InternLM2/3, ChatGLM, Exaone, MiniCPM,
the Bailing/Pangu/TeleChat families, …); their HF checkpoints ship `trust_remote_code` modeling files
instead, so nothing can build them from a config class. A spelling added for one of those families is an
unchecked claim, and `tests/test_vllm_only_families.py` is what checks it — against a synthetic tree in
the shape the family's own modeling file describes, with the source cited. InternLM2 is the worked
example there.

## Structural facts inspection cannot see

Everything below is a fact that a config either does not carry or carries misleadingly, so it is
measured off a real block instead (`facts.py`'s third category — the live-module predicates). Each
section names the trap, the families it applies to, and the invariant that detects it. You need at
most one of them per model; skip to the one your gate failure points at.

### Adding a quirk (`KNOWN_QUIRKS` in `arch.py`)

`Quirks` is structured data the capture/attention/lens code actually branches on — each field
maps to a real code path, and the `note` says why. Fields:

| field                     | when                                                                                                                                                                                                                                                                                    | who reads it                                                                                                                                                                                                                                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `qkv_layout`              | q/k/v packed in one matrix, no standalone `v_proj`. An **enum, not a bool** — see [Fused QKV layout](#fused-qkv-layout-is-per-backend-and-on-two-families-per-checkpoint). Set via `facts.EAGER_QKV_LAYOUTS`.                                                                           | DFA value split (`split_fused_qkv`, `ArchSpec.v_proj`/`fused_qkv_module`)                                                                                                                                                                                                                                                   |
| `parallel_attn_mlp`       | attention and MLP both read the layer input instead of running in sequence (GPT-NeoX `use_parallel_residual`, Falcon `parallel_attn`, GPT-J/CodeGen/phi-1,2/Cohere by name)                                                                                                             | changes what `mlp_in` _means_ (it is normed `resid_pre`); no `resid_mid` exists, so that point **refuses** and residual-contribution invariants do not apply                                                                                                                                                                |
| `n_residual_streams`      | the trunk carries more than one residual stream (hyper-connections: DeepSeek-V4's `hc_mult`, Motif 3's `mhc_expansion_rate`)                                                                                                                                                            | all three `resid_*` points **refuse**: the tensors between blocks are `(batch, seq, streams, d_model)` and every consumer of a residual assumes one stream — see [More than one residual stream](#more-than-one-residual-stream-hyper-connections)                                                                          |
| `sandwich_norms`          | each sublayer's output is normed before the residual add (Gemma-2/3/4, OLMo-2/3)                                                                                                                                                                                                        | enables `attn_out_post`/`mlp_out_post`; see [Post-sublayer norms](#post-sublayer-sandwich-norms-the-post_attention_layernorm-trap) (structural, detected on a real block)                                                                                                                                                   |
| `gated_attn_out`          | `q_proj` is double width and its second half gates the attention output, so `z` is post-gate (Qwen3-Next, Qwen3.5)                                                                                                                                                                      | enables the `attn_gate` point; **invalidates `probs @ value == z`** and leaves DFA uncorrected — see [Gated attention output](#gated-attention-output-z-is-post-gate-so-probs--value-is-not-z)                                                                                                                              |
| `qk_norm`                 | which axis the in-attention q/k norms normalize — `PER_HEAD` (Qwen3, Gemma-3/4) or `FLAT` (OLMo-2/3), an **enum, not a bool**, measured off the norm's weight                                                                                                                           | enables `q_norm_in`/`q_norm_out` and the `k` pair, and tells a caller how to reshape them: the two conventions differ by a reshape, not a scale (structural, detected on a real block)                                                                                                                                      |
| `moe_layers`              | which layers have a sparse mixture-of-experts MLP — **per layer**, since a dense prefix or every-k-th-sparse pattern is the norm (config-driven)                                                                                                                                        | `mlp_in`/`mlp_out` tap `layer.mlp` regardless, but this gates the points _below_ the block: it enables `router_logits`/`expert_weights`/`expert_indices` and refuses the neuron-basis points (`mlp_pre`, `mlp_pre_linear`, `mlp_act`), whose projections live on the experts. See [MoE](#moe-tap-the-block-not-the-experts) |
| `attn_sinks`              | learned softmax-denominator sink; attention does NOT sum to 1 (gpt-oss)                                                                                                                                                                                                                 | attention capture — **never renormalize**                                                                                                                                                                                                                                                                                   |
| `final_logit_softcapping` | `logits = c·tanh(logits/c)` NOT applied by the real `lm_head` (Gemma)                                                                                                                                                                                                                   | lens applies it explicitly on the eager path (config-driven). **vLLM applies it for you** — see [vLLM `compute_logits` is not a bare unembed](#vllm-compute_logits-is-not-a-bare-unembed)                                                                                                                                   |
| `logit_multiplier`        | a scalar the family's forward multiplies its logits by after `lm_head` — Cohere's `logit_scale`, Granite's `logits_scaling` (a divide), Falcon-H1's `lm_head_multiplier`, LLaDA's `scale_logits` (a bool meaning `1/√d_model`) — normalized to one multiply by `facts.logit_multiplier` | lens applies it explicitly on the eager path (config-driven). **vLLM applies it for you**, and the worker asserts the two numbers agree — see [vLLM `compute_logits` is not a bare unembed](#vllm-compute_logits-is-not-a-bare-unembed)                                                                                     |
| `attn_logit_softcapping`  | attn-score softcap inside eager attention (Gemma-2)                                                                                                                                                                                                                                     | eager attn applies it; recompute path matches (config-driven)                                                                                                                                                                                                                                                               |
| `hybrid_layer_types`      | some layers compute no softmax attention (Qwen3-Next/3.6 linear attention, Jamba/Bamba mamba, LFM2 conv)                                                                                                                                                                                | attention endpoint guards on `layer_types` for **both** backends (config-driven). See [Block types](#block-types-classify-them-do-not-pattern-match-them)                                                                                                                                                                   |
| `sliding_window`          | banded attention on the layers `layer_types` marks `sliding_attention` (gpt-oss 128, Gemma-3 512/1024, Gemma-2 4096)                                                                                                                                                                    | free on eager; the vLLM recompute rebuilds the band — see [the three terms](#the-three-terms-the-vllm-attention-recompute-must-reapply) (config-driven)                                                                                                                                                                     |
| `tied_embeddings`         | `lm_head.weight is embed.weight`                                                                                                                                                                                                                                                        | recorded only on the eager path (we call the real `lm_head`); **vLLM may omit `lm_head` entirely** — see [vLLM unembed / tied embeddings](#vllm-unembed--tied-embeddings)                                                                                                                                                   |

Rules for adding one:

- **Config first.** Softcaps, `layer_types`, `tie_word_embeddings` are read from `config` at
  resolve time and merged over static hints — add them to the config, not the table. Put in
  `KNOWN_QUIRKS` only what no config field exposes (currently just `attn_sinks`).
- **Ask whether the fact is per-backend.** If eager and vLLM would answer differently, it does not
  belong in one shared field at all — see [Fused QKV
  layout](#fused-qkv-layout-is-per-backend-and-on-two-families-per-checkpoint) for the worked example.
- Key on `config.architectures[0]` (the HF class name, e.g. `"GptOssForCausalLM"`).
- When first filling an entry, cross-reference the TransformerLens / circuit-tracer adapter for
  that arch to confirm the structural fact, then write a `note` explaining it. Never leave a bare
  boolean.

### Fused QKV layout is per-backend, and on two families per _checkpoint_

A fused `qkv` projection's output can be sliced into three equal parts on _any_ architecture, so
the wrong split still yields a `value` tensor of the right shape and a plausible magnitude — it is
simply not the model's. It then feeds DFA, which reports confident attribution numbers derived from
noise. Nothing raises, and no shape or normalization check notices. This is why layout is an enum
(`facts.QKVLayout`) rather than a boolean, and why a bare `fused_qkv: True` is the wrong fix.

Three layouts are in use, and they are mutually incompatible:

| layout                     | packing                                                                                         | who                                                                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `CONTIGUOUS_THIRDS`        | `[all_q \| all_k \| all_v]`                                                                     | gpt2's Conv1D `c_attn`, MPT's `Wqkv`, Phi-3's `qkv_proj`, the MQA configurations of GPT-BigCode and Falcon; **every** vLLM `QKVParallelLinear` |
| `PER_HEAD_INTERLEAVED`     | `[h0_q h0_k h0_v \| h1_q h1_k h1_v \| …]`, i.e. `(n_heads, 3, head_dim)`                        | HF's GPT-NeoX, BLOOM, and the non-MQA configurations of GPT-BigCode and Falcon                                                                 |
| `PER_KV_GROUP_INTERLEAVED` | `(n_kv_heads, q_per_kv + 2, head_dim)` — each KV head's queries, then its k row, then its v row | Falcon's `new_decoder_architecture` (40B, 180B, 11B)                                                                                           |

**Falcon and GPT-BigCode pack all three ways from one code path**, chosen by `multi_query` and
`new_decoder_architecture`, so Falcon-7B and Falcon-40B need different splits and a single table entry
would be silently wrong for one of them. Those two are resolved from the config
(`facts._MULTI_QUERY_PACKED_ARCHS`), and the same flag also decides `n_kv_heads`
(`facts.effective_kv_heads`), which the split depends on: Falcon-7B's config says `num_kv_heads: 71`
and it attends with **one**, so taking the field at face value reshapes `z` and `value` into 71 heads
that do not exist. The two answers have to agree or the split lands mid-tensor.

**The same checkpoint needs different splits on the two backends.** vLLM routes every family
through `QKVParallelLinear` and normalizes the checkpoint to match as it loads — its GPT-NeoX
loader transposes `(n_heads, 3, head_dim)` into `(3, n_heads, head_dim)` weight by weight. So HF's
GPT-NeoX is interleaved while vLLM's is contiguous, and a single shared value would hand one
backend the other's answer. Hence `facts.eager_qkv_layout()` and `facts.vllm_qkv_layout()`.

Adding a family: **do not guess from the attribute name.** Verify with the invariant below, which
is layout-agnostic, then record the result in `facts.EAGER_QKV_LAYOUTS`. Under GQA note that k/v
are narrower than q, so contiguous "thirds" are not actually equal thirds. A checkpoint you cannot
download is no excuse to skip the verification: the identity is algebra over one forward pass, so it
holds on a **randomly initialized** model of the same shape built from config defaults, which is how
the Falcon, BLOOM, MPT, Phi-3 and GPT-BigCode rows in `tests/test_qkv_layout.py` are checked. Note
also that a layout entry is not free — recording one for a family with _standalone_ projections
(Phi-3.5-MoE, which shares Phi-3's lineage but not its packing) makes `value` refuse rather than
resolve, because there is then nothing fused to split.

### Block types: classify them, do not pattern-match them

Whether a layer computes softmax attention decides whether `attn_probs`/`z`/`value` exist there, and
**`attn_probs` is indexed by position among attention layers, not by layer number**. So misjudging a
block type does not raise — it returns a _different layer's_ attention.

The **field name needs no normalization**: transformers maps `layers_block_type` onto `layer_types`
via `attribute_map` (Zamba2, Falcon-H1, Nemotron-H), synthesizes it from `full_attn_idxs` (LFM2) and
from `hybrid_override_pattern` (Nemotron-H), and rewrites the legacy `mamba`/`attention` spellings in
configs that call `remap_legacy_layer_types`. Reading `config.layer_types` is enough.

The **values** do need classifying, against `facts.SOFTMAX_ATTENTION_LAYER_KINDS` /
`NO_ATTENTION_LAYER_KINDS`, and not by substring. A `"linear" in kind` test matches only
`linear_attention` and lets every other non-attention spelling through as an ordinary attention
layer: `mamba`/`mamba2` (Jamba, Bamba, Zamba2, Falcon-H1, Granite-4-hybrid), `recurrent`
(RecurrentGemma), `conv` (LFM2), and `mlp`/`moe` (Nemotron-H blocks that are only an MLP). Two
values are easy to get backwards in the other direction: `deepseek_sparse_attention` and
`heavily_compressed_attention` attend fewer keys but **do** have a softmax, so probs exist.

An **unrecognized** value stays permissive — treated as attention, so a new family still loads — and
is reported by `facts.unclassified_layer_kinds` so the attention endpoint refuses and names it
(`attn_config`). Loading must not break on something only the attention path cares about; the
attention path must not guess.

The same applies to the **separate** `mlp_layer_types` field, which says which layers are sparse, and
where the spellings disagree even within one lineage: DeepSeek-V3.2 says `sparse`/`dense` while
DeepSeek-V4 says `moe`/`hash_moe` (`facts.SPARSE_MLP_LAYER_KINDS` / `DENSE_MLP_LAYER_KINDS`). Comparing
against one spelling reads a V4 trunk as entirely dense, which loses the router points on it.

Two block shapes need something other than a name. **Nemotron-H** calls every sublayer `mixer` — the
same attribute is a `NemotronHAttention`, a `NemotronHMLP`, a `NemotronHMoE` or a
`NemotronHMamba2Mixer` depending on the layer — so a name vocabulary cannot separate them and putting
`mixer` in `ATTN_ATTRS` would bind attention points to a state-space recurrence. That one is resolved
by **class** (`facts.mixer_role`), the only place in `facts.py` that does. **Zamba2/Zaya `hybrid`**
layers hold attention one level below the block, under a `shared_transformer` (the block's own
children are `linear`, `mamba_decoder`, `shared_transformer`), so resolution has to descend rather
than read an attribute off the block — see `facts.SUBLAYER_CONTAINER_ATTRS`.

The name `shared_transformer` describes shared _weights_, not a shared module: a 54-layer Zamba2 has
nine hybrid layers holding nine distinct block objects that all reference one `q_proj.weight`, with
per-layer LoRA in `*_adapter_list` on top. Each fires exactly once per forward, so addressing and
capture are per-layer as usual. What repeats is a per-layer _weight_ readout, and
`facts.has_tied_attention_weights` flags that.

### Per-layer attention dims, and layers with no `value` at all

Almost every family has one `head_dim` and one `n_kv_heads` for the whole model. **Gemma-4 breaks
both**, so `ArchSpec` exposes per-layer accessors and the flat `head_dim` is a trap on that family:

- **`head_dim_for_layer(layer)`** — a `full_attention` layer uses `config.global_head_dim` (512 on
  E2B) while a `sliding_attention` layer uses `config.head_dim` (256). The top-level number is wrong
  by exactly 2x on over a third of the layers, and reshaping `z`/`value` into `(n_heads, head_dim)`
  with it mis-splits them without raising. `per_head_value` uses the per-layer value.
- **`is_kv_shared_layer(layer)` / `kv_source_layer(layer)`** — from `num_hidden_layers -
num_kv_shared_layers` onward (layer 15 of 35 on E2B) a layer reuses the keys/values of the last
  non-shared layer **of its own type** and is built with no `k_proj`/`v_proj` at all. 20 of E2B's 35
  layers have no value projection. `value` there is not hard to find, it is produced elsewhere, so
  capture refuses and names the source layer to read instead.

Both are checked against the real checkpoint on all 35 layers (`tests/test_per_layer_attn_dims.py`),
because the whole point is that the config's top-level numbers disagree with the modules. The vLLM
client carries `global_head_dim` and `first_kv_shared_layer` in its dims dict and resolves them
through the same shared predicates (`vllm_backend.head_dim_for_layer`, `kv_shared_source_layer`) —
including in the attention recompute, which reshapes each layer's q/k/v by that layer's own head dim
and counts its heads by dividing the captured width rather than trusting the model-level count.

`k_norm` follows `k_proj`: a KV-shared layer has neither, and its **`q_norm` is unaffected**. Presence
is therefore asked per side (`facts.has_qk_norm(attn, "q")`), because the pair-wise question answers
"no" for such a layer and refusing the query norm on the key norm's absence declines a tensor the model
plainly computes.

- **`kv_heads_for_layer(layer)`** — the kv-head count moves with the head width, and in the opposite
  direction: 4 against 16 on the 31B, 2 against 8 on the 26B, 1 against 8 on the 12B. transformers
  5.15 states this per layer (`per_layer_config`); below that the family spells it
  `num_global_key_value_heads`, which `Gemma4TextAttention` applies to a layer only when
  `attention_k_eq_v` is set **and** the layer is not sliding — so `ArchSpec` carries the flag beside
  the count rather than reading the count on its own.

`attention_k_eq_v` is on for the 26B, 31B and 12B, and off for E2B and E4B. It is a second, separate
way for a layer to have no `v_proj`, and it does not overlap with KV sharing: those three SKUs set
`num_kv_shared_layers: 0`, so **every** one of their `full_attention` layers is built with
`v_proj = None` and takes the key projection's output as the value instead — `v_norm(k_proj(h))`
against a key of `rope(k_norm(k_proj(h)))`, so the two are the same projection read through
different norms, not the same tensor.

**The third answer that layer needed is the norm, and it applies to the whole family.**
`ArchSpec.v_proj` still returns `None` there, which is the signal it uses for "fused QKV, the caller
must split", on a family whose `fused_qkv` is false — so `value` no longer asks it first.
`ArchSpec.value_module` does, and it returns the `v_norm` whenever the family has one
(`facts.ATTN_VALUE_NORM_ATTRS`). Two things that buys, and the first would apply even without
`attention_k_eq_v`:

- `Gemma4TextAttention.forward` runs `value_states = self.v_norm(value_states)` on **every** layer
  that projects its own KV, so the projection's output is a norm short of the tensor the attention
  pattern multiplies. `value` was off by that norm on the sliding layers too, which do have a
  `v_proj`.
- vLLM has no value projection to compare against at all — its `Gemma4Attention` splits one fused
  `qkv_proj`, with the K weights loaded into the V slot on the `k_eq_v` layers — but it runs the same
  `v_norm` on the V slice. So this is the only boundary at which the two engines hold the same tensor,
  which is what makes the point checkable rather than merely servable. `_tree._value_module` is the
  vLLM side, and it prefers the norm in the same order for the same reason.

**Reading a norm costs a rank, and the point does not pay it.** A norm over `head_dim` can only be
given the per-head view, so both engines hand `v_norm` a `(…, n_kv_heads, head_dim)` tensor where a
`v_proj` would have produced the flat one — and `value` would then be 4-D on this family and 3-D on the
next, against a `Width.HEADS` declaration and `run_with_cache`'s `[batch, seq, width]`. Flattened back
on both sides (`hooks.flat_per_head`, `vllm_capture._hooks.flat_value`), including on the way *into* a
steer, so a vector measured on a capture of `value` is the shape the write expects; the module gets its
own rank back before the attention reshapes it. Eagerly the head count is checked rather than assumed,
because the other 4-D layout in circulation is `[batch, heads, pos, head_dim]` — transformers norms
after the head transpose on ten families — and flattening that one puts the head count where every
reader expects the sequence.

**Off Gemma-4, `value` on vLLM is one third of a packed matrix.** Every family with a fused vLLM
implementation goes through `QKVParallelLinear`, whose output is `[q | k | v]` on the last axis, so an
output hook there returns all three under the value's name at three times the width.
`_tree.value_span` measures the q and k widths off the projection's own rank-local geometry
(`num_heads`, `num_kv_heads`, `head_size`, each divided by the TP size in its `__init__`) and
`_hooks.value_columns` applies it on both install paths, refusing a geometry it can read only in part —
every wrong offset into a packed matrix yields a right-shaped tensor of another projection's heads. On
the per-request path the narrowing happens *before* steering, so a steer on `value` leaves the same
matrix's q and k alone.

The `k_eq_v` layers stay the interesting case for a *reader*: `value` there is `v_norm(k_proj(h))`
where the key is `rope(k_norm(k_proj(h)))`, so the two points come off one projection through
different norms and only one of them has RoPE applied. `ArchSpec.is_k_eq_v_layer` answers which layers
those are, gated on `not sliding` the way the modeling code gates it — the flag is model-wide and the
structure is not.

Two more ways the value side can differ from the query side, both of which produce a correctly shaped
and completely wrong per-head split if you reshape by `head_dim`:

- **`value_head_dim_for_layer(layer)`** — the value head is not always as wide as the q/k head.
  MiMo-V2 uses 64 for q/k and 128 for v (`config.v_head_dim`); the DeepSeek MLA families use 192 and 128. `value` and `z` are `n_heads * v_head_dim` wide on those, so `per_head_value` and
  `head_contributions` split by this and not by `head_dim_for_layer`.
- **`value_scale(layer)`** — MiMo-V2 multiplies `v_proj`'s output by `v_scale` (1/√2) before attention
  reads it, so the projection's output is not the value the attention consumed. The `value` _point_
  stays the module's output, because that is what a hook on a module means; `per_head_value` applies
  the scale, so it is the DFA path that is right. Read off the module rather than the config, since
  that is where the number is bound.

Both are why `per_head_value` refuses instead of reshaping when a layer's width does not equal
`n_kv_heads * v_head_dim`: with powers of two everywhere the wrong split usually divides. Inkling is
the family that trips it — it sizes its sliding layers from `swa_num_attention_heads` / `swa_head_dim`,
so the model-level head count describes only its full-attention layers, and per-layer _head counts_
are not something this engine models.

A whole class of attention has **no `value` to hook anywhere**: multi-head _latent_ attention
(DeepSeek-V2/V3/V3.2/V4, MiniCPM3, GLM-MoE-DSA) keeps a compressed kv latent and expands it inside
the forward, so no module output is the value and there is nothing to split either. `value` refuses
and says so, and `z` still exists. Treat "this family has no capturable value" as a fact to pin
somewhere, not one to discover twice: a _new_ latent-attention family should fail a check rather than
quietly lose DFA.

### Gated attention output: `z` is post-gate, so `probs @ value` is not `z`

Qwen3-Next and Qwen3.5 make `q_proj` **double width** and split its output per head into the query
and a gate, then apply `attn_output * sigmoid(gate)` on the way into `o_proj`:

\[ z = (\text{probs} \cdot \text{value}) \odot \sigma(\text{gate}) \]

This **breaks the ground-truth identity** used everywhere else here. `probs @ value == z` is what
validates fused-QKV splits and DFA, and on these models it is false — by a positive per-element
factor in (0, 1), so the result keeps its shape and a plausible magnitude. If you are adding one of
these families and the QKV invariant fails, this is why; check `quirks.gated_attn_out` before
concluding the split is wrong.

The live consequence is for DFA. It computes per-source contributions as `probs @ value` projected
onto a `hook_z` encoder direction, but a `hook_z` SAE for one of these models is trained on the
_post-gate_ `z`, so the attribution runs through a quantity the SAE never saw. **This is currently
uncorrected** — there are no attention SAEs for these models yet, so the engine exposes the gate and
flags the model rather than silently changing DFA:

- `quirks.gated_attn_out` — detected from the module, not an architecture list, because the projection
  _is_ the mechanism. Read off a **softmax** attention layer, which on Qwen3.5 is not layer 0 (it
  alternates three `linear_attention` GatedDeltaNet blocks, which have no `q_proj` at all, to one
  `full_attention` block).
- capture point `attn_gate` + `capture.attn_out_gate(model, cache, layer)` → `sigmoid(gate)` as
  `[batch, pos, n_heads, head_dim]`. The raw projection is **per-head interleaved**, the same trap as
  fused QKV: halving the flat vector mixes queries into the gate.

**A second shape of the same thing.** Afmoe and Laguna keep the gate in a _separate_ projection inside
the attention module (`gate_proj`, `g_proj`) rather than packing it into `q_proj`, and apply a sigmoid
and a softplus to it respectively. Both are `gated_attn_out`, and there `attn_gate` resolves to that
projection's output — no split needed. `capture.attn_out_gate` covers only the packed layout, because
the activation is applied _inline in the block_ and there is no module to read it from. Since the
activation is not readable from a module, checking that the projection the engine found is really the
gate means supplying the family's activation externally and reconstructing `z` with it.

### Sublayers that add the residual themselves

BLOOM and MPT hand the residual **into** the sublayer: `BloomMLP.forward(hidden_states, residual)` ends
in `dropout_add(down_proj_out, residual)`, and BLOOM's attention does the same. So the module's output
is the block's running stream, not what the sublayer computed — `mlp_out` on BLOOM would be
`resid_post`, and `attn_out` would be `resid_mid`. Right shape, plausible magnitude, a whole residual
stream away from the tensor an SAE was trained on.

`facts.sublayer_adds_the_residual(module)` reads the **forward's signature**: a sublayer cannot add a
residual it was not given. Where it holds, `mlp_boundary`/`attn_boundary` return the _projection's_
output (`down_proj`, `o_proj`), which is the contribution before the add. Nothing else in the engine
branches on it.

### Sublayer output multipliers

A few families scale each sublayer's output on the way into the residual:

\[ \text{resid_mid} = \text{resid_pre} + m \cdot \text{attn_out} \]

Granite (`config.residual_multiplier`, **0.22** on the 3.x checkpoints), HyperCLOVAX, Falcon-H1
(attention side only), and MiniCPM3 — whose multiplier is _derived_ (`scale_depth / sqrt(n_layers)`) and
so is read off the decoder layer's own `residual_scale` rather than from any config field.

Since `attn_out_post` / `mlp_out_post` are _defined_ as the residual contribution, capture multiplies
them by `quirks.residual_multipliers`; the raw `attn_out` / `mlp_out` stay the module outputs. Every
default is 1.0, so a random-weights test cannot see this — a real checkpoint is the only place it
appears, so any test covering it has to set one explicitly.

**On vLLM the multiply happens outside the module the hook is on**, in the decoder layer's own
`forward` (`hidden_states = residual + hidden_states * self.residual_multiplier`), so a hook on
`self_attn`/`mlp` sees the unscaled tensor and the same point means two different things on the two
backends. `vllm_capture._tree.capture_scale` corrects it at _collect_ — not in the hook, which also
writes steering back into the forward — and Gemma's embedding scale is the same story in the other
direction: HF applies `sqrt(d_model)` inside `Gemma3TextScaledWordEmbedding.forward` while vLLM applies
it in `embed_input_ids` around `self.embed_tokens(...)`, so eager's `embeddings` hook sees it and
vLLM's does not.

Both are constant factors, which is the hardest kind of error to notice: the tensor is finite,
right-shaped, and perfectly correlated with the truth, so cosine similarity stays 1.0 and only the
magnitudes are wrong. The cross-engine comparison scored 38 such cells green before it gained a
magnitude gate to go with its cosine one.

The correction, when something needs it: the gate depends only on the **destination** position and
DFA fixes one destination row, so scale that row's encoder direction by the gate
(`w_per_head * gate[dest]`) instead of re-forming the product. Exact, and free.

### MoE: tap the block, not the experts

MoE changes what is _inside_ the MLP, not where the MLP is, so **a mixture-of-experts model needs
no capture changes**. `mlp_in` / `mlp_out` tap `layer.mlp`, which is the complete block — router,
routed experts, and any always-on shared expert — consuming and producing `d_model`. Resolution
deliberately never descends to `mlp.experts`, which would silently drop the shared expert's
contribution on every family that has one (DeepSeek-V3, Qwen2/3-MoE, Qwen3-Next).

Two things to know:

- **A sparse block returns a tuple** `(hidden_states, router_scores)` where a dense one returns a
  bare tensor. `hooks.extract_hidden` takes element 0 generically, so this is already handled — but
  it is pinned by a test, because capturing the router scores instead would give a `[tokens,
n_experts]` tensor that is plausible rather than obviously wrong.
- **"Is this an MoE model" is the wrong question.** Most are only partly sparse, so the fact is per
  layer (`quirks.moe_layers`, and `ModelFacts.is_moe_layer`). Two config idioms cover the families
  in transformers, and `facts.is_moe_layer` mirrors both branch expressions exactly rather than
  inferring from the loaded model: a **dense prefix** (`layer >= first_k_dense_replace`; DeepSeek-V3,
  Mistral-4, dots1, GLM-4.5) or **every k-th layer sparse with a dense opt-out list** (`(layer + 1)
% decoder_sparse_step == 0 and layer not in mlp_only_layers`; Qwen2/3-MoE, Qwen3-Next,
  Qwen3-VL/Omni). Expert counts have four live spellings (`num_local_experts`, `num_experts`,
  `n_routed_experts`, `moe_num_experts`), all read, and the top-k has four of its own
  (`num_experts_per_tok`, `experts_per_token`, `moe_topk`, `top_k_experts`). A third idiom, an
  explicit `mlp_layer_types` pattern, wins over both — see the spelling warning in [block
  types](#block-types-classify-them-do-not-pattern-match-them).
- **Gemma-4 breaks the first sentence of this section.** Its sparse layers do not swap the MLP for an
  expert bank: `Gemma4TextDecoderLayer` builds `self.mlp` on every layer and, where
  `enable_moe_block` is set, hangs `self.router` and `self.experts` **beside** it on the block,
  summing the two branches (`post_ffn_norm_1(mlp(x)) + post_ffn_norm_2(experts(x))`, both reading the
  pre-feedforward residual). Three consequences, and `facts.dense_mlp_beside_experts` is the fact they
  hang off. A sparse layer keeps a real neuron basis, so `mlp_pre`/`mlp_act` must stay served there —
  which they are, because the LongCat guard makes the `mlp_pre` refusal require a router **on the MLP
  module**, and Gemma-4's is a sibling. The parameter count of a sparse layer includes the dense MLP,
  so the usual `(n_layers - n_sparse) * dense_mlp` undercounts (the 26B by ~0.5B). And `mlp_out` is
  **not** the whole feed-forward here, so it is refused — see below.

Which experts fired _is_ capturable, one level down: `router_logits` / `expert_weights` /
`expert_indices` are three elements of the router module's own output tuple, resolved by
`arch.moe_router` (`gate` on Mixtral/Qwen/OLMoE/DeepSeek, `router` on gpt-oss). Four things to keep
in mind if you touch that path:

- **The order of that tuple is per family.** It is `(logits, weights, indices)` on Mixtral, Qwen,
  OLMoE, DeepSeek and gpt-oss, and exactly reversed on GraniteMoe, GraniteMoeShared and
  GraniteMoeHybrid, whose `GraniteMoeTopKRouter` returns `(top_k_index, top_k_weights,
router_logits)`. `facts.ROUTER_OUTPUTS` holds the exceptions; `facts.assert_routing_shapes` runs on
  every capture, because reading the tuple backwards yields tensors that are the wrong width under
  the right name rather than an error.

- **Do not reimplement the top-k.** Read it. The conventions are mutually incompatible and all yield
  `k` weights summing to 1, so a wrong guess is plausible and silent: Mixtral softmaxes then selects
  then renormalizes unconditionally, Qwen3-MoE renormalizes only under `norm_topk_prob` (default
  `False`), Qwen3.5-MoE always does and has no such field, gpt-oss selects on the raw logits and
  softmaxes only the survivors, DeepSeek-V3 scores with a sigmoid, masks all but the best expert
  _groups_ using an `e_score_correction_bias` it then discards, and scales the result.
- **The name `gate` is used twice in one block.** `MOE_ROUTER_ATTRS` matches exact attribute names so
  it cannot reach Qwen3-Next's `shared_expert_gate` (a 1-wide sigmoid for the shared expert) or a
  dense MLP's `gate_proj`. A prefix match here would capture the wrong tensor on both.
- **A fused kernel can leave the router module unused.** transformers' MXFP4 path for gpt-oss replaces
  the _block's_ forward and routes inline (`F.linear` on `self.router.weight`, then a Triton top-k), so
  the router is in the tree, correct, and never called. `arch.moe_router` detects the replaced forward
  and refuses with the remedy (`Mxfp4Config(dequantize=True)`); more generally, `run_with_cache` now
  raises when a resolved module did not run, rather than returning a short cache. `router_logits` is the
  exception, and it does not need the remedy: `mlp_forward` returns `(routed_out, router_logits)`, so the
  point resolves to the _block's_ `output:1` instead (`arch.inline_routing_logits`) and is bit-identical
  to the router's own linear. Which index holds them is allowlisted per replacement
  (`facts.INLINE_ROUTING_FORWARDS`) because the un-replaced block returns `router_scores` at that same
  index — a four-wide top-k against the logits' 32 — and gpt-oss carries a second, unverified kernel
  swap (`@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`) whose layout nothing here has confirmed.
  `expert_weights` and `expert_indices` have no boundary on that path even so, and are _rebuilt_ from
  those logits by `run_with_cache` (`interp_engine.moe_routing`) — the one place a routing convention is
  reimplemented, allowed because the derivation is a top-k on an already-captured tensor and because it
  is verified against `GptOssTopKRouter` on the real checkpoint rather than read off the source. They
  still refuse from `resolve_point`, which answers with addresses and has none to give.

The MLP's neuron basis (`mlp_pre`, `mlp_pre_linear`, `mlp_act`) is refused on a sparse layer, since
the projections live on the experts — often as one fused 3-D parameter per bank, with no per-expert
module at all. Gemma-4 is the exception noted above, and the refusal already lets it through for the
right reason rather than by luck: it is gated on the router being found *on the MLP module*, the same
guard that keeps LongCat's shortcut MoE from suppressing its two real feed-forwards.

**Three things follow on Gemma-4's 26B from that one fact** — the routed branch is a sibling of
`layer.mlp` rather than a part of it, so the *block* owns both halves of the feed-forward:

- **`mlp_out` is refused**, on both backends. It taps `layer.mlp`, which here returns the dense branch
  alone: correctly shaped `[tokens, d_model]`, at the right positions, missing the experts entirely,
  and with nothing about it to notice — both engines build the same tree, so a cross-engine sweep
  agrees on the same half rather than catching it. The summed tensor has no module boundary to serve
  instead (the sum is a local of the block's `forward`), and there is already a point that means it:
  `mlp_out_post` is the `post_feedforward_layernorm`'s output, downstream of the sum, and the one for
  which `resid_post == resid_mid + mlp_out_post` holds. `ArchSpec.mlp_is_half_the_feed_forward` is the
  question both refusals ask. Nothing else moves: `mlp_in`, `mlp_pre`, `mlp_pre_linear` and `mlp_act`
  are the dense branch's own internals and mean what they mean everywhere else.
- **The router is looked for on the block as well as on the MLP** (`facts.moe_router_owner`). Asking
  only `layer.mlp` left all three routing points unresolvable on the 26B. Presence alone is not enough
  now that a decoder layer is searched: `facts.moe_router_attr` requires a callable submodule, because
  a block can hold a plain flag under one of these names and the caller is about to install a forward
  hook on whatever comes back.
- **`router_logits` is read one module deeper than the tuple.** `Gemma4TextRouter.forward` returns
  `(router_probabilities, top_k_weights, top_k_index)`: element 0 matches the default tuple *order*
  while being a softmax over all 128 experts, and is exactly as wide as the logits over the same bank,
  so neither `ROUTER_OUTPUTS` nor a width check can see it — and the block discards it. So the point
  reads the router's own `proj` output, which is what that softmax consumed and what vLLM's
  `Gemma4Router.forward` returns outright, leaving the two engines comparable
  (`facts.ROUTER_LOGITS_SUBMODULE`, plus a `ROUTER_OUTPUTS` row naming slot 0 for what it is).
  `assert_routing_shapes` also rejects a `router_logits` that is non-negative everywhere and sums to 1
  per token, which is the family-agnostic version of the same catch. Elements 1 and 2 are the weights
  and indices the block really routes with, and are read where every other family's are.

In our shipping set only gpt-oss-20b is MoE (32 experts, top-4, no shared expert, every layer
sparse). Its 3D batched expert weights matter only to code reading MLP weight matrices, and the
engine reads none — it hooks activations.

### A block with no MLP _module_, and a block with no MLP at all

Two shapes where `mlp_module` finds nothing, wanting opposite responses.

**OPT and XGLM inline their projections**: `fc1` and `fc2` hang on the decoder layer itself, so the
MLP exists but no module's input and output are `mlp_in` and `mlp_out`. Those two points resolve to
the projections' boundaries instead — the same two tensors — via `ArchSpec.mlp_boundary`, and the
neuron basis works normally. What does **not** resolve there is `resid_mid`, and the refusal is the
interesting part: OPT's pre-MLP norm is called `final_layer_norm` (which is also what its _trunk_
calls the model's final norm) and `do_layer_norm_before` decides whether it runs before the MLP or
after it, so falling back to `fc1`'s input would hand back the normed value under the residual's name.

**A state-space block has no feed-forward at all**: a `MambaBlock` is a norm and a mixer. There the
MLP points do not exist, and the refusal says that rather than reporting a missing submodule, because
"this architecture has no such part" and "we could not find the part" send a reader to completely
different places. Same for the attention points, which are refused from `layer_types` before any
lookup happens.

### More than one residual stream (hyper-connections)

Every canonical point in this engine except three is a submodule's input or output, and those three —
`resid_pre`, `resid_mid`, `resid_post` — additionally assume there is _one_ residual stream, shaped
`(batch, seq, d_model)`, that each block reads and adds to. That assumption is what the logit lens, a
steering vector, an SAE trained on `resid_post`, and the decomposition
`resid_pre + attn_out_post + mlp_out_post == resid_post` are all built on.

Manifold-constrained hyper-connections (mHC) break it. DeepSeek-V4 shipped them first and Motif 3
followed. The trunk carries several streams stacked as `(batch, seq, streams, d_model)`; each block
**collapses** them into one sequence with learned per-token weights, runs its sublayer on that, and
scatters the output back across the streams through a Sinkhorn-projected doubly-stochastic matrix. So
the block's input is a stack of streams, its output is a different stack, and the sublayer outputs
never add into any single tensor.

**This is the one gap in the audit that would otherwise be silent.** The block resolves fine; its
activations merely have an extra axis, `d_model` is still last, and anything that broadcasts over the
extra one returns a shaped and wrong answer instead of raising. So `facts.residual_streams` reads the
count off the config, `Quirks.n_residual_streams` carries it, and the three residual points refuse
with the count and the shape in the message. Everything else on the family — `attn_out`, `mlp_out`,
the neuron basis, `final_norm`, `lm_head` — is a submodule boundary and works as usual.

Each family spells the count its own way, so `facts._RESIDUAL_STREAM_FIELDS` holds one name per
family: `hc_mult` on DeepSeek-V4, `mhc_expansion_rate` on Motif 3.

If you add a family with this shape, add its field name there — and check whether that family also
ships a **switch**. Motif 3 does: `mhc_enabled` sits beside `mhc_expansion_rate`, and a config with
the rate still at 4 and the flag off carries one stream, so reading the count alone would refuse
`resid_post` on an ordinary trunk. Those names go in `_RESIDUAL_STREAM_SWITCHES`, where only an
explicit `False` disables — an absent flag is not a disabled one, or DeepSeek-V4 (which has no flag)
would read as single-stream, which is exactly the silent wrong answer this section exists to prevent.

**Also add its module layout**, to `facts.HYPER_CONNECTION_LAYOUTS`. The count decides which points
_exist_ — the seven mHC rows are gated on it, not on the architecture name, because `MotifForCausalLM`
is Motif 2.6B and Motif 2-12.7B as well as Motif 3 and only the last has the trunk. The layout decides
where each of them _is_, and the two families disagree by exactly enough to be dangerous: `attn_hc` /
`ffn_hc` return `(post, comb, collapsed)` while `mhc_attn` / `mhc_ffn` return `(h_pre, h_post, h_res)`,
so the same tuple index means the write coefficients on one family and the mixing matrix on the other,
at the same shape and dtype. A layout row is also the only place that can say a quantity is not a
module output at all: Motif 3's collapsed vector is its pre-sublayer norm's input, since the block
applies the coefficients rather than the module returning the result.

Working in this basis needs those points rather than the count, and the count is only ever asked to be
right about being \>1 — it backs a refusal. Note that the collapse _weights_ are a distinct quantity
from the collapsed _vector_ and only Motif 3 returns them (`h_pre`); no canonical point names them yet,
so that one is a dotted module path away.

**The layout table describes eager, and vLLM's DeepSeek-V4 fits none of its two shapes.** Both rows
assume the mHC quantities live on a per-block *module* — either returned by it or fed to a norm it
owns. On vLLM's NVIDIA tree there is no such module: `hc_attn_{fn,base,scale}` and
`hc_ffn_{fn,base,scale}` are flat parameters on the decoder layer itself, two of the quantities come
back as elements of the layer's own return tuple, and the other five are locals of its forward that
reach no boundary at all. All seven are served, and the second five by wrapping the mHC kernel calls
(`vllm_capture.mhc`) rather than by hooking anything. Three consequences worth stating, all measured on
DeepSeek-V4-Flash at vLLM 0.26.0:

- The collapse _weights_ are not a module output there; they are not an output at all. A consumer
  working in this basis can still have everything it needs, because they are derivable from those flat
  parameters and the stream stack — which is what serving `*_stream_collapse` does.
- The collapsed _vector_ is not reachable on that tree even by a dotted path, because `attn_norm` /
  `ffn_norm` are fused into the mHC kernel and only the **normed** collapse ever crosses a boundary —
  the engine's `attn_in`/`mlp_in`. A third layout shape — "not materialized" — is what would be needed
  to describe it. The point is therefore rebuilt rather than read, the only mHC row whose value is
  arithmetic. The AMD and XPU trees apply the norm separately and so behave like the eager layouts,
  where it is an ordinary module output.
- **The stream stack is off by one sublayer at the layer boundary**, which is the trap this section
  exists to flag, because the tensor there has the shape a `resid_streams` check would look for. vLLM
  defers each sublayer's write into the next sublayer's kernel, so what the layer returns is the stack
  the MLP read — `resid_mid` in stream form — and the block's own output stack is a local of the *next*
  layer's first kernel. A consumer that takes `output:1` for `resid_streams` gets a real tensor from one
  sublayer earlier, on every layer, with nothing in its shape to say so. `resid_streams` is read one
  layer downstream for exactly that reason, off the kernel that does the scatter — and off the model's
  own closing `mhc_post` call for the last layer, which no layer completes.

Four of the seven are capture-only, and that is a fact about them rather than about a backend: the
per-stream write weights and the Sinkhorn-normalized mixing matrix are the hyper-connection's own
parameters, and an additive edit leaves a doubly stochastic matrix neither stochastic nor a mixture of
anything. The other three are activations on the residual trunk and can be written under vLLM, each by
a mechanism the fused kernels force rather than by a hook — a steer of a collapse arrives as the
difference it makes to the norm fused in above it, and a steer of the stream stack re-runs the fused
call's second half so that the collapse computed inside that same call sees the edit, for the steered
rows only, since that half does not reproduce the fused kernel bit for bit. A `stream` coordinate says
which row of the stack to write. See [ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md#steering-the-three-mhc-points-that-are-activations).

See [ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) for the measurements and the per-vendor split.

### Post-sublayer (sandwich) norms: the `post_attention_layernorm` trap

**`post_attention_layernorm` means two different things depending on the family, and getting it
wrong misidentifies every Llama-family model.** Never key sandwich-norm detection on that name.

| family                                   | `post_attention_layernorm` is…                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| Llama, Qwen, Mistral (2 norms per block) | the **pre-MLP** norm: applied to the _residual_, after `resid = resid + attn_out`    |
| Gemma-2/3/4 (4 norms per block)          | a true **post-sublayer** norm: applied to the attention _output_, before it is added |
| OLMo-2/3                                 | a true post-sublayer norm, and there is **no pre-attention norm at all**             |
| GLM-4, HyperCLOVAX (4 norms per block)   | the **pre-MLP** norm again — these post-norm attention under a _different_ name      |

The reliable discriminator is the presence of the **MLP-side post-norm**
(`facts.POST_MLP_NORM_ATTRS`: `post_feedforward_layernorm`, or Afmoe's `post_mlp_layernorm`). It
exists exactly on the families that have real post-sublayer norms (Gemma-2/3/4, OLMo-2/3, Afmoe) and
not on Llama-shaped blocks, and it is structural rather than a name coincidence. Do not key on "is it
Gemma" either: Gemma-1 has none of these norms and VaultGemma _removes_ them.

**Being a sandwich block is not enough to settle the ambiguous name, though.** GLM-4 post-norms its
attention as `post_self_attn_layernorm` and still uses `post_attention_layernorm` for the pre-MLP norm,
exactly as Llama does; HyperCLOVAX numbers its pair `post_norm1`/`post_norm2` and does the same. So the
vocabulary is split by whether a spelling is ambiguous (`AMBIGUOUS_POST_ATTN_NORM_ATTRS`) or means only
one thing (`POST_ATTN_NORM_UNAMBIGUOUS_ATTRS`): when an unambiguous sibling claims the attention side,
`post_attention_layernorm` is free and `pre_mlp_norm_attr` accepts it again. Reading GLM-4 as
Gemma-shaped bound `attn_out_post` to the _next_ sublayer's normed input and left `resid_mid` on the
normed residual — two wrong tensors of the right shape, on one of the most-used families in the list.

Not every such module is a norm, either: Inkling runs a short **convolution** over each sublayer's
output before the add (`attn_sconv`, `mlp_sconv`). Structurally that is the same fact — this module's
output, not the sublayer's, is what reaches the residual — so those spellings live in the same lists,
and only the name of the concept is imprecise.

Which makes that one list the most expensive place to miss a spelling in this file. A missing name
there does not just lose `mlp_out_post`: detection fails, the whole block re-reads as pre-norm, and
`post_attention_layernorm` is handed back as the pre-MLP norm — so `resid_mid` becomes the _attention
output_ and `attn_out_post` silently equals `attn_out`. Afmoe shipped both wrong for exactly this
reason (see `test_a_sandwich_norm_the_mlp_side_does_not_name_is_not_read_as_a_pre_norm_block`).

Consequences for capture:

- `mlp_out` / `attn_out` are the **raw submodule outputs** on every architecture. That is what SAEs
  are trained on, and it matches nnsight (`mlps_output` is `LayerAccessor(self, "mlp", OUTPUT)`,
  the raw output, with no sandwich-norm awareness anywhere in nnsight or nnterp).
- `mlp_out_post` / `attn_out_post` are the **residual contributions** — a hook on the post-norm
  module, so `resid_post == resid_pre + attn_out_post + mlp_out_post` holds. On an architecture
  with no post-norm they alias the raw points (same hook, same tensor, no extra clone), so a caller
  wanting the composing quantity never branches on the family.
- `resid_mid` reads the same trap from the other side, and so keys on the same signal
  (`facts.pre_mlp_norm_attr`). It is the **input to the pre-MLP norm**, which is
  `post_attention_layernorm` on a Llama-shaped block but `pre_feedforward_layernorm` on Gemma-2/3/4 —
  where hooking the ambiguously named module instead would return a tensor from _before_ the
  attention add. OLMo-2/3 have no pre-MLP norm at all, so there it aliases `mlp_in`.
- **That alias is gated on the post-MLP norm, and not on failing to find a pre-MLP one.** "This block
  has no pre-MLP norm" and "this block's pre-MLP norm is spelled something we don't know" are the same
  observation, and aliasing in the second case returns `mlp_in` — the residual _after_ normalization —
  under the residual's name, at no point raising. Five families were doing that (`pre_ff_layernorm` on
  Bamba/Falcon-H1/Jamba, `feedforward_layernorm` on Apertus, `ffn_norm` on LFM2), so the alias now
  requires a post-MLP norm and the refusal prints what the block calls its norms. Verify a new spelling
  against the family's `forward` before adding it: the name is not proof of position, and OPT's
  `final_layer_norm` moves either side of the MLP depending on `do_layer_norm_before`.
- **A block with one sublayer has no `resid_mid` at all**, which is not the same as a parallel block
  having none. Nemotron-H interleaves single-sublayer blocks (each layer is an attention, _or_ an MLP,
  _or_ a Mamba2 mixer) and calls the one norm `norm`, so both candidate answers are wrong there:
  the norm's input is `resid_pre` and the MLP's input is that normalized.
- But the question is whether the block **mixes positions**, not whether it _attends_ — the two come
  apart on a hybrid trunk. LFM2's `conv` layers run a short causal convolution where attention would
  go and are otherwise ordinary sequential blocks (mix, add, `ffn_norm`, feed-forward, add), so
  `resid_mid` is their `ffn_norm`'s input like anywhere else, even though `attn_out` correctly refuses
  for want of an attention module. `Arch.has_position_mixer` is the predicate, and it counts a short
  convolution (`facts.POSITION_CONV_ATTRS`) alongside attention and the state-space mixers. Keying it
  on the attention module instead cost the whole point on LFM2-8B-A1B, whose sampled layers were all
  convs; vLLM captured a `resid_mid` there that the eager reference declined to, and the tensor was a
  15–107% relative distance from the `resid_pre` the refusal implied it would be.
- TransformerLens reports something **different** on these models: its `hook_mlp_out` fires _after_
  the post-norm, so it is the residual contribution. TL v3 gets there by hand — each of its 139
  architecture adapters names the roles explicitly (`gemma2.py` maps `ln1_post` →
  `post_attention_layernorm` and `ln2_post` → `post_feedforward_layernorm`, while `llama.py` maps
  plain `ln2` → `post_attention_layernorm`), and `BlockBridge` then rewrites `hook_mlp_out` to
  `ln2_post.hook_out` only when an `ln2_post` was declared. So TL never has to infer this; a human
  did it per family. That is robust but is the per-family-code approach this engine avoids, which
  is why we use the structural signal plus a numeric check instead.
- Expect cos ≈ 0.2–0.4 between our `mlp_out` and TL's on Gemma. That is a convention difference,
  not a bug: `mlp_out` is defined here as the raw module output, and `mlp_out_post` is the residual
  contribution TL's block-level hook reports. Compare like against like before concluding anything.
- OLMo-2/3 are a free pass on this specific discrepancy — both TL implementations use the raw
  pre-norm value there, so all engines agree. But note their `mlp_in` is the _raw residual_ rather
  than a normed value, since they have no pre-attention norm.

## Proving it: invariants and ground truth

### Verify numerically, don't trust the attribute name

Both traps above share a failure mode: a wrong answer that is shape-correct and quiet. So the
check that matters is an invariant computed from the model's own tensors, which cannot be satisfied
by a plausible-looking guess.

- **`value` / DFA — reconstruct `z`.** `z` (the input to the attention output projection) _is_ the
  concatenated per-head attention output, so a correct per-head `value` must satisfy
  `einsum("bhqk,bkhd->bqhd", probs, value) == z` to floating-point equality. That is also exactly
  the computation DFA performs, so agreement is a statement about DFA and not just about reshaping.
  Load in **float32** for this: an `auto` fp16 checkpoint carries ~1e-4 error, which is larger than
  the gap you are trying to detect. `tests/test_qkv_layout.py` does this for both layouts, and also
  asserts the _other_ layout fails — otherwise the test would pass on a model with one head. One
  exception: on a `gated_attn_out` model the identity is genuinely false and the gate has to be
  applied first — see [Gated attention
  output](#gated-attention-output-z-is-post-gate-so-probs--value-is-not-z).
- **Residual contributions — `resid_post == resid_pre + attn_out_post + mlp_out_post`**. This catches a
  sandwich-norm misidentification on a family nobody has inspected, a sublayer that adds the residual
  itself, and an unapplied multiplier. It holds on a parallel block too (both sublayers read
  `resid_pre`); what does not exist there is `resid_mid`.
- **Neuron basis — `down_proj(mlp_act) == mlp_out`.** Catches a neuron basis resolved to the wrong
  projection of a gated MLP.
- Only after an invariant is green should you record the fact in `facts.py`.

**These three hold on any model of the family, trained or not**, so a randomly-initialised model
shrunk to two layers checks them for free — no download, no allocation worth counting. Each identity
must either hold or be _refused with a reason_: a sparse block has no single down projection, MLA has
no value, a Mamba layer has no attention. A mismatch is a bug, and so is an exception from inside the
capture that names nothing.

Two things a randomly-initialised model cannot show you, both worth knowing when you add a family:

- **Anything a trained config turns on.** Every multiplier defaults to 1.0, so Granite's 0.22 is
  invisible on defaults. Those cases are set explicitly by their own tests.
- **A layer type the shrunk config does not produce.** A 2-layer Jamba is all Mamba, so its attention
  points are refused rather than checked. Real hybrids interleave, and a real checkpoint is the only way
  to exercise both.

### Ground truth: how to know you got it right

- **Parity tests** (`tests/`, run on CI): `test_parity_gpt2.py` is the CPU golden gate;
  `test_new_models_gpu.py` is reference-free self-consistency for large models (decoding the last
  layer's residual through the real `final_norm`+`lm_head` must reproduce the model's true
  next-token argmax — this validates the arch map end-to-end; gpt-oss additionally checks
  attention-sink rows don't sum to 1); `test_sliding_window_attn.py` pins the vLLM recompute's
  band and sink terms, including a CPU run _past_ gemma-3-270m-it's 512-token window on a layer
  that is actually banded.
- **The strongest check is an independent implementation** on real weights, point by point. That is
  not something this repo runs: nothing here loads another framework, and the parity gate above is
  deliberately the one exception (TransformerLens, gpt2, one golden file). When a number from
  another stack disagrees with ours, read [Post-sublayer (sandwich)
  norms](#post-sublayer-sandwich-norms-the-post_attention_layernorm-trap) first — a hook that fires
  after the post-sublayer norm reports a _different tensor_, not a wrong one, and that convention
  difference accounts for most apparent mismatches on Gemma-shaped models.
- **Expected non-issues:** in bf16 the top SAE feature index is stable but the exact top value / L0
  can wobble by ~1 near the activation threshold — that is bf16 noise, not a mismatch.

### Gotchas we actually hit (checklist)

- **`attn_implementation="eager"` is required** to read attention _probabilities_ (SDPA/flash
  don't materialize them). The server forces this for `EagerModel`.
- **`head_dim ≠ d_model/n_heads`** on Gemma (GQA + explicit `head_dim`): the attention-output /
  `z` width is `n_heads*head_dim`, not `d_model`. Attention-SAE code must use `head_dim`.
- **Embedding scaling** (Gemma multiplies embeddings by `sqrt(d_model)`): free when you hook real
  modules, but if you ever feed embeddings _in_ (NLA `prompt_embeds`), you must apply the scale
  yourself — `resolve_embed_scale` in the NLA path exists for exactly this.
- **Multimodal `*ForConditionalGeneration`**: dims live under `config.text_config` (top-level are
  `None`); the text trunk nests under `model.language_model`. `resolve_arch` handles both; just
  don't assume top-level `num_hidden_layers`. The vLLM worker resolves its _own_ module handles
  and needs the same treatment separately — see [vLLM worker-side module
  resolution](#vllm-worker-side-module-resolution).
- **Attention sinks (gpt-oss)**: never renormalize captured attention to sum to 1.
- **Sliding-window attention (gpt-oss, Gemma-2/3/4)**: banded on the layers `layer_types` marks,
  full on the rest. Free on eager; the vLLM recompute must rebuild the band, and **it is invisible
  on a short prompt** — see [the three terms](#the-three-terms-the-vllm-attention-recompute-must-reapply).
- **Hybrid attention (Qwen3-Next/3.6)**: linear-attention layers have no softmax probs — the
  attention endpoint skips them (the webapp hides those layers).
- **Tied embeddings / MoE**: on the eager path, tied is numerically free (we call the real
  `lm_head`, which shares storage with embed). For MoE confirm the `mlp` submodule output is the
  residual contribution you want as `mlp_out`. **On vLLM, tied embeddings are not free** — see
  [vLLM unembed / tied embeddings](#vllm-unembed--tied-embeddings).
- **dtype**: load native (`--model_dtype auto`). MPS has poor bf16 → the server runs bf16-native
  models in fp16 on MPS (or falls to CPU); see `interp_engine/select.py`.
- **vLLM serving specifics** (only if the model will run on the fused backend): vLLM has its own
  arch-support probe (`select._vllm_supports_arch`); capture hooks must `.clone()` (vLLM
  reuses activation buffers in-place across the fused add+RMSNorm); attention probabilities are
  recomputed off-kernel from post-RoPE q/k/v; `prompt_embeds` forces the legacy V1 runner (needs
  `spawn` + prefix-caching off); and the per-request capture/steer demux reads
  `InputBatch.req_ids` order (vLLM 0.25.1 sorts the batch by token count). These are engine-owned
  in `vllm_backend.py` / `vllm_capture/_demux.py`, but re-validate on a vLLM version bump.
- **The three vLLM lens hazards below.** All per-family: where the worker's module handles live
  (loud, but only once the endpoint runs), where the unembedding weight lives, and what
  `compute_logits` applies on the way out (both silent).

## If it will be served on vLLM

The fused backend re-derives what the eager one gets for free, so this is where a model that passed
every check above can still be wrong. Three of the four things to check fail **silently**.

### The three terms the vLLM attention recompute must reapply

vLLM's fused paged kernel never materializes the softmax, so `recompute_attn_probs`
(`vllm_capture/attn.py`) rebuilds it from post-RoPE q/k captured at `self_attn.attn`. Capturing those
_inputs_ is architecture-agnostic — no per-family q/k/rope/norm code. **The softmax around them is
not.** Three terms the kernel applies are invisible at the hook, and each one is silent when
missed: the result stays a well-formed probability matrix, just the wrong one.

| term                             | source                                                                                                                                             | if you miss it                                                                                                            |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `attn_logit_softcapping`         | config (Gemma-2)                                                                                                                                   | scores uncapped                                                                                                           |
| `sliding_window` + `layer_types` | config (gpt-oss 128, Gemma-3 512/1024, Gemma-2 4096)                                                                                               | queries attend past the band                                                                                              |
| attention sinks                  | **a weight** (`self_attn.sinks`, gpt-oss)                                                                                                          | rows renormalized to 1                                                                                                    |
| the score `scaling` itself       | config — `attention_multiplier` where a family states one (Granite 1/64, HyperCLOVAX), else `query_pre_attn_scalar` (Gemma), else `head_dim**-0.5` | every score off by a constant factor: **8x on Granite**, where the stated multiplier and the inverse square root disagree |

The first two are config-driven and resolved client-side in `read_attn_dims` /
`sliding_window_for_layer`. The sink is a `nn.Parameter`, so no config field exposes it — the
worker reads it next to q/k/v and ships it in the capture payload. The scaling is `facts.attn_scaling`,
and eager never needs it: it reads the module's own `.scaling` (or the value the forward passed), which
is where the derived-versus-stated distinction cannot come up.

**And every dim is per layer.** Gemma-4 widens the head on its `full_attention` layers and changes the
kv-head count with it (16x256 vs 4x512 on the 31B), so the config's top-level triple describes neither
kind of layer. The client reshapes q/k/v by `head_dim_for_layer` / `value_head_dim_for_layer` and counts
heads by dividing the captured width, which is the one reading that cannot disagree with the tensor in
hand — see [Per-layer attention dims](#per-layer-attention-dims-and-layers-with-no-value-at-all).

Two things make this class of bug hard to see, and both are worth knowing before you add a model:

- **The window needs a long prompt.** A band only shows up once a query can reach past it. Every
  inference pod's `--token_limit` except gpt-oss-20b's currently sits _below_ its model's window,
  which hides the term entirely — an accident of configuration we deliberately do not rely on,
  since token limits are meant to grow. `tests/test_sliding_window_attn.py` runs a 600-token
  prompt against gemma-3-270m-it's 512 window for exactly this reason.
- **The window is per layer.** `layer_types` alternates, so banding a `full_attention` layer is
  as wrong as leaving a `sliding_attention` one unbanded, and a check that samples only one kind
  of layer sees neither. `sliding_window_for_layer` is the single place that decides.

**The tripwire.** Because that list of terms is not knowable in advance, `attn_config.py`
inverts the question. Rather than asking "what quirks does this model have?", it asks "does
this config contain anything touching attention that nobody has classified?" — which is finite
and mechanical. Every attention-relevant field sits in one of three places:

| where                                | meaning                                                            |
| ------------------------------------ | ------------------------------------------------------------------ |
| `CONSUMED`                           | read by `read_attn_dims` and applied by the recompute              |
| `BENIGN`                             | looked at, cannot move a probability, **with the reason recorded** |
| a check in `unsupported_attn_config` | recognized and _not_ reproducible — the endpoint refuses           |

Anything else matching the attention field pattern is unclassified, and the attention endpoint
returns 400 rather than serving a plausible-looking pattern that is not the model's. Adding a
model that trips it means reading that field's semantics and filing it — which is the point.
Refusal is scoped to the endpoint, not the model load, so the pod still serves everything else.

Two things to know about its reach. It would have caught the original bug: with `sliding_window`
absent from `CONSUMED`, gpt-oss-20b, gemma-2-2b and gemma-3 all trip (there is a test that
rewinds the table and asserts this). But it **only sees config** — the attention _sink_ is an
`nn.Parameter` that no config field mentions, and nothing in `attn_config.py` would ever have
noticed it. Weight-expressed quirks are covered by reading the weight at capture time and by
parity against eager instead; neither check replaces the other.

### Where else a tripwire belongs

A config tripwire only works at a specific shape of code: **a site where we re-derive by hand
something a real `forward()` would otherwise have done for us.** Everywhere the engine calls the
actual module, quirks are free — `transformers` applies them and we never need to know they
exist. That leaves exactly two places in the engine where a config field can be load-bearing and
unread at the same time:

| site                   | what is re-derived by hand                             | guard                                                                                                                                                                                                                    |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `recompute_attn_probs` | the whole softmax, from captured post-RoPE q/k         | `attn_config.py` classification tripwire                                                                                                                                                                                 |
| `worker_lens_readout`  | what vLLM's `compute_logits` already did to the logits | ask the model (`_worker_applied_softcap`); refuse on `scale` — both only as good as finding the `LogitsProcessor`, which a wrapper nests (see [vLLM worker-side module resolution](#vllm-worker-side-module-resolution)) |

Which guard fits depends on **how the quirk fails**, and the two known Gemma-2 bugs are the
clean contrast:

- **Attention failed by omission** — `sliding_window` was a field we never read. Classification
  is the right answer: enumerate the fields, refuse on an unknown one.
- **The final softcap failed by duplication** — `final_logit_softcapping` was read _correctly_
  and applied _twice_, because vLLM had already applied it. No amount of config classification
  catches that; the field was known and consumed. What catches it is **asking the backend what
  it already did** instead of assuming it did nothing.

So: classification for "did we forget to read something", reconciliation for "did both sides do
the same thing". A site that re-derives across two backends usually needs both, which is why the
lens row above has one of each — plus a flat refusal on `LogitsProcessor.scale`, the sibling
term vLLM applies and eager does not.

**What a tripwire would _not_ have caught.** Two other Gemma-2 differences are worth naming,
because neither is a config-reading bug and pointing a tripwire at them would be theater:

- The **sandwich/post-norm hook** difference (our `hook_attn_out` vs TransformerLens's published
  one on Gemma-2/3, documented under [Post-sublayer (sandwich)
  norms](#post-sublayer-sandwich-norms-the-post_attention_layernorm-trap)) is not a
  field anyone failed to read — the model applies its own post-norm and we hook the real module.
  It is a disagreement about _what the hook means_, which no tripwire can see: only running both
  implementations on the same weights and comparing the numbers surfaces it.
- The attention **sink** is an `nn.Parameter`. No config field mentions it, so `attn_config.py`
  would never have noticed; it is caught by reading the weight at capture time and by parity
  against eager (`tests/test_sliding_window_attn.py`).

That is the general rule: tripwires cover config, parity covers everything expressed in weights
or in convention. Neither replaces the other.

The off-by-one is pinned to transformers' own `sliding_window_overlay`
(`kv_idx > q_idx - sliding_window`, i.e. the band holds `sliding_window` keys _including_ the
query) by comparing against a real model's eager attention rather than re-deriving it. Do not
adjust `causal_window_mask` from memory.

When adding a model that will run on vLLM, run the parity script **on that model**:

```bash
IE_VLLM_GPU_UTIL=0.8 .venv/bin/python scripts/vllm_attn_recompute_check.py --model <hf_id>
```

It picks a banded and a full-attention layer, grows the prompt past the window automatically, and
prints a `rowsum_min` column (below 1 means sink mass, correct on gpt-oss and a bug elsewhere).
Its default `Qwen/Qwen3-0.6B` has no window, no sinks and no softcap, and therefore **cannot fail
any of the three** — the same trap as `vllm_unembed_check.py`'s default, below.

### vLLM worker-side module resolution

Everything the worker does by hand — installing capture hooks, decoding a residual for the lens,
fetching `W_U` rows — starts by finding a module on `worker.model_runner.model`. That object is
**vLLM's** model class, not the HF one, so `arch.py` cannot be reused and the layout is whatever
that family's file in `site-packages/vllm/model_executor/models/` builds.

A multimodal wrapper moves _all_ of it at once. Qwen3.6-27B loads as
`Qwen3_5ForConditionalGeneration`, which holds a vision tower plus a nested `Qwen3_5ForCausalLM`,
so relative to a plain causal LM every handle sits one level deeper:

| handle            | plain causal LM    | multimodal wrapper (Qwen3.5/3.6, Gemma 4) | read by                                  |
| ----------------- | ------------------ | ----------------------------------------- | ---------------------------------------- |
| decoder layers    | `model.layers`     | `language_model.model.layers`             | every capture/steer hook                 |
| final norm        | `model.norm`       | `language_model.model.norm`               | `worker_unembed`, `worker_lens_readout`  |
| `LogitsProcessor` | `logits_processor` | `language_model.logits_processor`         | softcap reconciliation + `scale` refusal |
| `W_U`             | `lm_head.weight`   | `language_model.lm_head.weight`           | `worker_lm_head_rows` (jlens swap/steer) |

Two properties make this worth its own checklist entry:

- **It fails at request time, not load time.** The pod starts, reports the model as served, and
  then each endpoint dies on its own lookup as you exercise it — layers on the first capture,
  the norm on the first lens read-out, `W_U` only on a jlens swap. Fixing one just advances to
  the next, so treat them as one change.
- **One of the four is silent.** A missing `logits_processor` reads as "no softcap applied, unit
  scale", which is exactly the wrong default: it double-caps the logits on a Gemma-style model
  and disarms the `scale` refusal described below. The other three raise.

So none of them enumerate dotted paths per family. `_walk_trunk` in `vllm_capture/_tree.py` walks the
model and its trunk containers (`model` / `language_model` / `text_model` / `transformer` /
`decoder` / `gpt_neox`) breadth-first, outermost match wins, and each resolver picks the first
matching leaf — mirroring `arch._resolve_trunk` on the eager side. The walk never enters
`visual` / `audio_tower`, so it cannot return a vision tower's `blocks` or `norm`; the layer-list
check additionally requires the first element to look like a decoder layer (`self_attn` /
`linear_attn` / `mlp` / …, so hybrid trunks match too).

When adding a family that will be served on vLLM: open its vLLM model file, find whether it is a
wrapper and what it names the child holding the text LM. If that name is new, add it to
`_TRUNK_CONTAINER_ATTRS` — one edit fixes all four lookups — and cover it with a CPU stand-in in
`apps/inference/tests/unit/test_worker_get_layers.py`. No GPU needed: the bug is "attribute in a
different place", not numerics.

### vLLM unembed / tied embeddings

For jlens swap/steer: `VLLMModel.unembed_rows` fetches unembedding directions via
`worker_lm_head_rows` → `_worker_unembed_layer` in `vllm_capture/lens.py`. Under tensor
parallelism each rank returns only the vocab rows it owns; `merge_lm_head_row_payloads`
assembles them on the client (indexing a global id outside a rank's shard would be an
out-of-bounds GPU assert). HF/eager almost always has a top-level `lm_head`; **vLLM does
not always**. Check the family's vLLM model class
(`site-packages/vllm/model_executor/models/<family>.py`):

| vLLM pattern                                                            | Example                                                  | Where `W_U` lives                 |
| ----------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------- |
| Creates `ParallelLMHead` (optionally `.tie_weights(embed_tokens)`)      | Llama, Gemma 3, GPT-2                                    | `model.lm_head.weight`            |
| Creates `embed_out` instead of `lm_head`                                | GPT-NeoX / Pythia                                        | `model.embed_out.weight`          |
| **Never creates `lm_head`**; `compute_logits` uses `model.embed_tokens` | **Gemma 1/2**                                            | `model.model.embed_tokens.weight` |
| Multimodal wrapper; text LM nested under `language_model`               | **Gemma 4**, **Qwen3.5/3.6** `*ForConditionalGeneration` | `language_model.lm_head.weight`   |

`_worker_unembed_weight` resolves this in two ordered passes over `_walk_trunk` (see [vLLM
worker-side module resolution](#vllm-worker-side-module-resolution)): a real `lm_head` /
`embed_out` anywhere in the text stack first, tied `embed_tokens` / `wte` only if there is none.
The order is the point — an untied model whose head nests deeper than its embedding table would
otherwise unembed with the wrong matrix, and produce plausible logits while doing it.

When adding a family that will be served on vLLM:

1. Open that vLLM model file and find `compute_logits` / `__init__` — note which module is
   passed to `LogitsProcessor` (or whether `lm_head` is skipped under `tie_word_embeddings`).
2. Confirm the module is reachable from `_walk_trunk` (it is, unless the wrapper names its text
   child something not in `_TRUNK_CONTAINER_ATTRS`).
3. Cover it with a CPU stand-in in `apps/inference/tests/unit/test_worker_lm_head_rows.py`
   (no GPU needed — the bug is "attribute missing", not numerics).
4. Smoke a jlens **swap** (or any `/v1/lens/prompt` with steer tokens) against the live pod;
   a miss surfaces as `Could not locate unembedding weight on the vLLM model` during
   `_build_steer_deltas`, not during ordinary logit-lens readout (which uses `compute_logits`
   and never indexes `W_U` by row).

### vLLM `compute_logits` is not a bare unembed

The lens decodes a residual by calling `compute_logits`, which is _not_ the fused equivalent of
`lm_head(x)`: it runs the model's `LogitsProcessor`, and that applies `soft_cap` and `scale` from
the config **before returning**. The eager engine's `lm_head` applies neither. So the same
residual decodes to different logits on the two backends, and whatever the caller adds on top
lands on a value that already has it.

This cost us a silent Gemma-2 regression. `worker_lens_readout` applied `final_logit_softcapping`
to logits vLLM had already capped, and capping twice compresses hardest exactly where the logits
are large — it roughly halved the gap between the top candidates, flattening the read-out while
still returning well-formed probabilities with the right answer on top.

| `LogitsProcessor` arg                                                                  | Wired by                                                                                                                                                                     | If the caller applies it too |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `soft_cap=config.final_logit_softcapping`                                              | Gemma 2/3/4, gemma3n, rnj1 — but only Gemma-2's config actually sets a value (30.0)                                                                                          | capped twice                 |
| `scale` (`config.logit_scale`, `1/config.logits_scaling`, `config.lm_head_multiplier`) | Cohere Command-R, Granite / GraniteMoE, Falcon-H1, Nemotron, Solar, Exaone, Whisper, Chameleon. Llama and several others pass `getattr(config, "logit_scale", 1.0)`, a no-op | scaled twice                 |

`worker_lens_readout` asks the loaded model what it already applied (`_worker_applied_softcap`)
and skips its own softcap when vLLM has one, so the softcap column is handled for any family.

**`scale` is now reconciled the same way, from the other side.** The eager lens applies the
multiplier itself, resolved from the same config fields by `facts.logit_multiplier` and carried as the
`logit_multiplier` quirk, so the two backends return the same logits for the same residual. What the
worker still checks is that the two _numbers_ agree: `_assert_applied_logit_scale_agrees` compares
`LogitsProcessor.scale` against our resolved fact and raises when they differ, naming the config field
it read. That is a stronger guard than the "refuse any non-unit scale" tripwire it replaced, which was
right only while the eager path could not match it. The case it still catches is the important one — a
family whose scale vLLM derives from a field we do not read — because the only symptom would otherwise
be a lens off by a constant, which leaves the argmax alone and changes every probability.

Note the ordering that follows from this, pinned by `apply_logit_transform`: multiplier first, then
softcap. No known family sets both, so nothing else would catch it being flipped.

To check a family, run `apps/inference/scripts/vllm_unembed_check.py` **on that family**: it
compares vLLM's `compute_logits` against the eager `decode_residuals` and gives the eager side
the config's softcap so the two are comparable. Its default `Qwen/Qwen3-0.6B` sets neither
knob and therefore cannot catch either failure — an uncapped, unscaled model always passes.

[← back to the interp-engine README](../README.md)
