# Engine hook mappings

Every hook point the three hookable stacks expose, mapped onto each other: interp-engine's canonical
points, TransformerLens' hook names (`HookedTransformer` and `TransformerBridge` share them), and
nnterp's standardized accessors over nnsight. The fused engines (vLLM, SGLang) have no hook names of
their own — vLLM is reached through interp-engine's own points, marked below where it cannot serve one.

This page is only the dictionary. `interp_engine.mappers` does the same translation in code, both
directions, and raises rather than guessing where no equivalent exists.

**`*` marks a point interp-engine can capture eagerly but not under vLLM.** `—` means the stack has no name
for that tensor at all. Both the markers and the list vLLM enforces are derived from one table,
`interp_engine/points.py`, and `tests/test_points_registry.py` fails if this page disagrees with it — so a
missing `*` is a bug in the table or the page, not a judgement call. Note the marker is about _capture_:
`resid_mid` is readable everywhere but writable only where the block does not add before the norm.

## Hook name comparisons

In forward order. `5` stands for any layer index. The **caveat** column is the important one: an empty
cell means the names translate and you get the same tensor, and anything else means they do not, or
that capture is conditional. Each links to the note under the table.

| canonical point                              | interp-engine                                       | TransformerLens                                 | nnterp (nnsight)                                              | caveat                                                                                                    |
| -------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `embeddings`                                 | `cache["embeddings"]`                               | `hook_embed`                                    | `token_embeddings`                                            | [three tensors, three conventions](#embeddings-three-different-tensors)                                   |
| `resid_pre`                                  | `cache["resid_pre", 5]`                             | `blocks.5.hook_resid_pre`                       | `layers_input[5]`                                             |                                                                                                           |
| `attn_in`                                    | `cache["attn_in", 5]`                               | `blocks.5.hook_attn_in` (or `attn.hook_in`)     | `attentions_input[5]`                                         | [what the sublayer was handed, normed or not](#attn_in-is-the-attention-modules-input-not-a-norms-output) |
| `q_norm_in` / `k_norm_in` (QK-norm input)    | `cache["q_norm_in", 5]`                             | `blocks.5.attn.q_norm.hook_in` (bridge only)    | —                                                             | [shape is family-dependent](#qk-norm-shape-is-family-dependent)                                           |
| `q_norm_out` / `k_norm_out` (QK-norm output) | `cache["q_norm_out", 5]`                            | `blocks.5.attn.q_norm.hook_out` (bridge only)   | —                                                             | [shape is family-dependent](#qk-norm-shape-is-family-dependent)                                           |
| `value`                                      | `cache["value", 5]`                                 | `blocks.5.attn.hook_v`                          | —                                                             | [KV heads, and flattened](#addressing-and-shape-which-do-not-translate-for-free)                          |
| `attn_scores` (pre-softmax)                  | `cache["attn_scores", 5]`                           | `blocks.5.attn.hook_attn_scores`                | —                                                             | [not a hook on either backend; masked positions differ](#attention-scores-are-not-a-hook)                 |
| `attn_probs`                                 | `cache["attn_probs", 5]`                            | `blocks.5.attn.hook_pattern`                    | `attention_probabilities[5]`                                  | [conditional on all three stacks](#attention-probabilities-are-conditional)                               |
| `z` (per-head, pre-`W_O`)                    | `cache["z", 5]`                                     | `blocks.5.attn.hook_z`                          | —                                                             | [flattened, not per head](#addressing-and-shape-which-do-not-translate-for-free)                          |
| `attn_gate`                                  | `cache["attn_gate", 5]` `*`                         | —                                               | —                                                             | [the raw double-width projection](#qk-norm-shape-is-family-dependent)                                     |
| `attn_out` (raw module output)               | `cache["attn_out", 5]`                              | `blocks.5.attn.hook_out`                        | `attentions_output[5]`                                        | [raw output, not the contribution](#raw-output-versus-residual-contribution)                              |
| `attn_out_post` (residual contribution)      | `cache["attn_out_post", 5]`                         | `blocks.5.hook_attn_out`                        | —                                                             | [same TL name, different tensor](#raw-output-versus-residual-contribution)                                |
| `resid_mid`                                  | `cache["resid_mid", 5]` (writes refused under vLLM) | `blocks.5.hook_resid_mid`                       | _no accessor_: the pre-MLP norm's `.input`, under its HF name | [captured, not reconstructed](#resid_mid-is-captured-not-reconstructed)                                   |
| `mlp_in`                                     | `cache["mlp_in", 5]`                                | `blocks.5.mlp.hook_in` (or `hook_mlp_in`)       | `mlps_input[5]`                                               | [not `ln2.hook_normalized`, which is pre-gain](#a-norms-hook_normalized-is-not-the-sublayer-input)        |
| `mlp_pre` (pre-activation)                   | `cache["mlp_pre", 5]` `*`                           | `blocks.5.mlp.hook_pre`                         | —                                                             | [the neuron basis, and `up_proj` is ambiguous](#the-mlp-internal-points-are-the-neuron-basis)             |
| `mlp_pre_linear` (gated MLPs only)           | `cache["mlp_pre_linear", 5]` `*`                    | `blocks.5.mlp.hook_pre_linear`                  | —                                                             | [refuses on a plain MLP](#the-mlp-internal-points-are-the-neuron-basis)                                   |
| `mlp_act` (post-activation neurons)          | `cache["mlp_act", 5]`                               | `blocks.5.mlp.hook_post`                        | —                                                             | [`d_mlp` wide; refuses on sparse layers](#the-mlp-internal-points-are-the-neuron-basis)                   |
| `router_logits` (MoE)                        | `cache["router_logits", 5]`                         | — (TL hooks the softmax, not the logits)        | —                                                             | [read off the router, not recomputed](#the-moe-routing-points-are-read-wherever-a-read-is-possible)                    |
| `expert_weights` (MoE)                       | `cache["expert_weights", 5]` `*`                    | — (`hook_expert_weights` is a different tensor) | —                                                             | [in the router's ranking order](#the-moe-routing-points-are-read-wherever-a-read-is-possible)                          |
| `expert_indices` (MoE)                       | `cache["expert_indices", 5]` `*`                    | `blocks.5.mlp.hook_expert_indices`              | —                                                             | [integer-valued](#the-moe-routing-points-are-read-wherever-a-read-is-possible)                                         |
| `mlp_out` (raw module output)                | `cache["mlp_out", 5]`                               | `blocks.5.mlp.hook_out`                         | `mlps_output[5]`                                              | [raw output, not the contribution](#raw-output-versus-residual-contribution)                              |
| `mlp_out_post` (residual contribution)       | `cache["mlp_out_post", 5]`                          | `blocks.5.hook_mlp_out`                         | —                                                             | [same TL name, different tensor](#raw-output-versus-residual-contribution)                                |
| `resid_post`                                 | `cache["resid_post", 5]`                            | `blocks.5.hook_resid_post`                      | `layers_output[5]`                                            |                                                                                                           |
| `final_norm`                                 | `cache["final_norm"]`                               | `ln_final.hook_normalized`                      | `ln_final.output`                                             |                                                                                                           |
| `lm_head`                                    | `cache["lm_head"]` `*`                              | _(logits are returned, not hooked)_             | `logits`                                                      | [a bare unembed, which vLLM does not offer](#the-unembed-and-what-vllm-returns)                           |

A hyper-connection trunk adds seven more points, which are conditional on the family and so are
tabled separately: [the seven mHC points in TransformerLens](#the-seven-mhc-points-in-transformerlens).
On such a trunk `blocks.5.hook_out` is one of them rather than `resid_post`, which is the one row
above that changes meaning there.

### Raw output versus residual contribution

TransformerLens' **block-level** `hook_attn_out` / `hook_mlp_out` are the sublayer's _residual
contribution_, so on a sandwich-norm architecture (Gemma-2/3/4, OLMo-2/3) they fire **after** the
post-sublayer norm. `attn.hook_out` / `mlp.hook_out` are the raw module outputs everywhere. Ours and
nnterp's default to raw, and we spell the other one `*_post`. Same name, different tensor, on those
families — this is the mapping mistake that produces a plausible-looking wrong number.

It has been made here, and what it cost is worth knowing: Neuronpedia's inference server translated
`blocks.4.hook_mlp_out` to raw `mlp_out` for gemma-2-2b's `gemmascope-mlp-16k`, which encodes the SAE
off a tensor it was not trained on. Nothing raised — the SAE's own reconstruction gives it away (FVU
9.8 against 0.26, i.e. worse than predicting the mean, at an L0 of 8 against a declared 85), but the
endpoint simply returned zeros, so the symptom was a whole SAE source whose features never fired on
the very text its published dashboards were topped by. `mlp_out_post` aliases `mlp_out` wherever the
architecture has no post-sublayer norm, on both backends, so **asking for the contribution point is
safe on every family** and is what a consumer of block-level hook names should do when it cannot pass
a model to the mapper. See [PORTING.md](PORTING.md).

### Attention probabilities are conditional

`attn_probs` under vLLM is a recompute, not a hook (fused paged attention never materializes the
matrix), and eagerly it needs `attn_implementation="eager"`. nnterp's accessor is a source patch that
has to be enabled and validated at load, and is unavailable on some architecture/version pairs.

### Embeddings, three different tensors

TL's `hook_embed` is the bare token embedding, _before_ positional embedding and any embed scaling;
ours is the embedding module's output; `resid_pre` at layer 0 is the tensor that actually enters the
trunk on either.

### The unembed and what vLLM returns

Our `lm_head` is the unembed module's output. vLLM's `compute_logits` is not a bare unembed (it can
fold scaling), which is why the point is not offered there — see
[ARCHITECTURE_QUIRKS.md](ARCHITECTURE_QUIRKS.md#vllm-compute_logits-is-not-a-bare-unembed).

### `resid_mid` is captured, not reconstructed

It is the input to the pre-MLP norm, so it is the tensor the model formed rather than a sum you
assembled — which matters because the sum that equals it is `resid_pre + attn_out_post`, not
`resid_pre + attn_out`, on a post-norm family. Where the block has no pre-MLP norm at all (OLMo-2/3) it
aliases `mlp_in`, at no extra cost; on a **parallel** attention/MLP trunk (GPT-J, Falcon, GPT-NeoX with
`use_parallel_residual`) the quantity does not exist and the point raises rather than returning
`resid_pre`. TL3's bridge aliases `hook_resid_mid` to `ln2.hook_in`, i.e. the same boundary by the same
reasoning (and drops the alias on a parallel block, as we refuse it); legacy `HookedTransformer`
computes `resid_pre + attn_out` instead, which agrees with ours to fp32 round-off. On vLLM and SGLang
the norm is _fused_ with the residual add, so the point is the sum of that module's two arguments;
reading it is arch-general, but **writing** it is refused where the block adds before the norm, since
the edit would reach the MLP and not the skip connection.

### QK-norm shape is family-dependent

By a reshape. `q_norm_in`/`q_norm_out` are `[batch, pos, n_heads, head_dim]` on the Qwen3/Gemma-3
convention (`RMSNorm(head_dim)`, applied after the view into heads) and `[batch, pos, n_heads * head_dim]`
on the OLMo-2 one (`RMSNorm(n_heads * head_dim)`, applied before it). `arch.quirks.qk_norm` says which,
measured off the norm's weight. `k_norm_*` has `n_kv_heads` under GQA. On a gated-attention model
(Qwen3.5) the gate has already been split off, so `q_norm_in` is the query alone — unlike `attn_gate`,
which is the raw double-width projection. Where a checkpoint disables QK-norm (`use_qk_norm=False`,
leaving an `nn.Identity`) the points refuse rather than return the unnormalized query.

### A norm's `hook_normalized` is not the sublayer input

TransformerLens splits a norm into `x / scale` and `* w` and fires `hook_normalized` **between** them,
so `blocks.5.ln2.hook_normalized` is the normalized residual _without_ the norm's gain — on a folded
model and an unfolded one alike. HuggingFace's norm does both in one call, so no module outputs that
tensor and `tlens_hook_to_point` refuses the name rather than answering `mlp_in`, which is the same
tensor times the gain. On gemma-2-2b layer 19 the two have a cosine of 0.89, and a Gemma Scope
transcoder reads 287 firing features off the wrong one where its trained input gives 170.

The norm's _boundaries_ are `resid_mid` (input) and `mlp_in` (output), and the engine spans the gap
for you rather than making you reproduce TransformerLens' `RMSNorm.forward` from memory:

```python
from interp_engine import pre_gain_normalized, rms_norm_eps_for_model, run_with_cache, tlens_normalized_hook

address = tlens_normalized_hook("blocks.19.ln2.hook_normalized")  # -> Address("resid_mid", 19)
eps = rms_norm_eps_for_model(model)  # None on a LayerNorm family, where this equation does not apply
cache = run_with_cache(model, tokens, [address])
normalized = pre_gain_normalized(cache[address], eps)
```

`tlens_normalized_hook` returns None for any other hook name, so one branch routes it, and
`rms_norm_eps_for_model` works on both backends — the epsilon is config-derived
(`ModelFacts.rms_norm_eps`), so the vLLM client, which holds no modules, answers it from the same
code as eager. That recompute agrees with `HookedTransformer` to 2e-3 relative on gemma-2, the
residue being the converted weights' own drift rather than the formula. Where you _do_ hold the norm
module and want the gain as well, `capture.rms_norm_parts` returns both parts.

This is a supported path rather than a footnote because artifacts are trained on the tensor: Gemma
Scope's transcoders declare it as their SAELens `hook_name`, circuit-tracer's as their
`feature_input_hook`, and OpenMOSS' Llama-Scope-2 Lorsa reads `ln1` where its transcoders read `ln2`.
Substituting `mlp_in` reads them off the wrong activations. Same reasoning one level down for
`q_norm.hook_normalized`, in the table at the end.

### `attn_in` is the attention module's input, not a norm's output

Which matters on a post-norm family. On a Llama-shaped block the two coincide — the attention reads
`input_layernorm`'s output — but OLMo-2/3 have no pre-attention norm at all, so `attn_in` there is the
_unnormalized_ residual, and equals `resid_pre`. Reading the norm's output instead would be right on
one family and unavailable on the other, which is why both backends take the sublayer's own input.

Under vLLM that costs one wrinkle: Llama, Qwen3 and Gemma-3 call `self.self_attn(positions=...,
hidden_states=...)` by **keyword** while OLMo-2 calls it positionally, so this is the one point whose
worker hook is installed `with_kwargs=True` and reads whichever of the two carried the tensor.

### Attention scores are not a hook

Not on any module, on either backend. `transformers` forms the scores inside a plain function and
returns only the probabilities, and HF's own `output_attentions` is a forward hook on the attention
module reading element 1 of its output tuple — always post-softmax. So the eager point registers a
_wrapping attention implementation_ for the duration of the capture (`ALL_ATTENTION_FUNCTIONS`, the
documented way to add a kernel) and delegates to the checkpoint's own eager function for the output,
leaving the forward bit-identical. It therefore needs `attn_implementation="eager"`, like `attn_probs`.

Under vLLM it rides the `attn_probs` recompute rather than a hook: the scores are the tensor that
recompute takes its softmax over, so both come back from one pass over the same captured post-RoPE
q/k. Masked positions differ between the two backends — the recompute writes `-inf` where HF writes
its dtype's minimum — and both softmax to the same zero without comparing equal. The value is the
full pre-softmax tensor in the same order TL applies it — scaling, then Gemma-2's logit softcap, then
the additive causal/sliding mask — so **masked positions are large and negative, not zero**: HF's dtype
minimum where TL writes `-inf`, which softmax to the same zero but do not compare equal. gpt-oss's
attention _sinks_ do not appear, because a sink is an extra column in the softmax denominator rather
than a term in the scores (which is also why its probability rows do not sum to 1).

### The MLP-internal points are the neuron basis

And `up_proj` means two different things. On a gated MLP `mlp_pre` is `gate_proj`'s output (what the
activation function is applied to) and `mlp_pre_linear` is `up_proj`'s (the branch that is multiplied);
on a plain MLP `mlp_pre` is the single `c_fc`-style projection and `mlp_pre_linear` **refuses** rather
than handing back the same tensor twice. `mlp_act = act_fn(mlp_pre) * mlp_pre_linear` is the
post-activation vector MLP transcoders and neuron dashboards index, and it is the down projection's
_input_ because the activation is applied inline rather than by a submodule. All three are `d_mlp` wide,
not `d_model`. Note TL's weight names cross over (`W_gate` is HF's `gate_proj`, but TL's `W_in` is HF's
`up_proj`), so translate by hook name, not by weight name — which the cross-engine sweep now checks
numerically on all three points, on both TL implementations. Where the two input projections are
**fused** (Phi-3's `gate_up_proj`) both pre-activation points refuse and name the module to slice, while
TL3's bridge splits it for you (`JointGateUpMLPBridge`); `mlp_act` is downstream of the fusion and
unaffected either way. On a **sparse** layer all three refuse: the projections live on the experts, so
there is no per-token neuron vector at the block boundary — and note TL does _not_ refuse there, because
its `MoEBridge` aliases `hook_pre`/`hook_post` to the block's own input and output. Those are `d_model`
wide and are our `mlp_in`/`mlp_out`, so on an MoE model the same two hook names mean something else
entirely.

### The MoE routing points are read wherever a read is possible

Every family's router module returns the whole decision — `(router_logits, expert_weights,
expert_indices)` almost everywhere, and `(expert_indices, expert_weights, router_logits)` on IBM's
three Granite MoE families — so the three points are three elements of one module output, and no
routing convention is reimplemented for them. The order per family is `facts.ROUTER_OUTPUTS`, and
because a swapped reading is shape-plausible (`[tokens, k]` where `[tokens, n_experts]` was meant),
every capture checks that the logits are as wide as the expert bank and the selection is integers. That is deliberate: Mixtral softmaxes then selects then renormalizes,
Qwen3-MoE renormalizes only under `norm_topk_prob`, Qwen3.5-MoE always does and has no such field,
gpt-oss selects on the _raw_ logits and softmaxes only the survivors, and DeepSeek-V3 scores with a
sigmoid and selects within expert groups using a bias it then discards. All of them yield `k` weights
summing to 1, so a wrong guess is plausible and silent. `expert_weights` and `expert_indices` are
`[batch, pos, experts_per_token]` in the router's own ranking order — column 0 is a different expert in
every row — and `capture.expert_assignment` scatters them onto the expert axis when you need rows that
line up. TL's `hook_expert_weights` is **not** this tensor: it fires on the softmax over all experts,
before the top-k, so it is `softmax(router_logits)` and is unmapped for that reason.

One split worth knowing: where a quantizer replaces the MoE block's forward with a fused kernel
(transformers' MXFP4 path for gpt-oss routes inline with `F.linear` and a Triton top-k), the router
module is present and never called. `router_logits` survives that anyway — `mlp_forward` ends
`return routed_out, router_logits`, so the logits the kernel routed on leave the block, and the point
resolves to the block's own `output:1`, bit-identical to `F.linear(hidden, router.weight, router.bias)`.
`expert_weights` and `expert_indices` have no boundary there at all — the selection happens inside the
kernel — so on that path they are **rebuilt** from those logits by `run_with_cache`
(`interp_engine.moe_routing`), and `resolve_point` still refuses them because there is no address to
return. The index the logits come off is allowlisted per replacement (`facts.INLINE_ROUTING_FORWARDS`)
rather than assumed, because `output:1` on the _un-replaced_ gpt-oss block is `router_scores` — the
softmaxed top-4, four wide against the logits' 32.

Rebuilding is permitted under two conditions, both required, and only where a read is impossible:

1. **It costs nothing measurable.** A top-k and a softmax over ~32 values per token, on a tensor the
   pass already captured. No second forward, no module kept alive, no memory past the k-wide result.
2. **Its correctness is verified against the read path on a real checkpoint.** Not argued from the
   modeling source — that is the evidence that cannot tell the conventions above apart. For gpt-oss,
   `tests/test_new_models_gpu.py::test_gpt_oss_20b_rebuilt_routing_matches_the_router_it_could_not_read`
   loads the checkpoint twice, rebuilds the layer-0 decision on the MXFP4 path, reads it off
   `GptOssTopKRouter` on the dequantized path, and asserts the selection is identical — with a negative
   control showing the other ordering would have missed by >0.1.

So `facts.ROUTING_CONVENTIONS` has exactly one row. A family whose router _runs_ is read even though the
derivation would agree, because a read cannot drift when the family changes its convention; and a family
whose router does not run but whose convention nobody has verified here still refuses, rather than
returning a plausible top-k.

### A norm's scale is not a point

Not anywhere in interp-engine, TL's `hook_scale` notwithstanding: it is an intermediate of the norm's
arithmetic, not a boundary. `capture.rms_norm_parts(norm, x)` returns `(scale, gain)` with
`norm(x) == x / scale * gain`, so a scale freeze is `x / scale.detach() * gain` from a captured
`q_norm_in`. `gain` is measured with a probe rather than read off `weight`, because the Llama lineage
applies `weight` and Gemma's and Qwen3.5's apply `1 + weight` on a zero-centered parameter. TL's
`hook_normalized` is between the divide and that multiply, so it is neither of the two points.

### All seven mHC points are served on vLLM, by two mechanisms, and neither is a module hook

The seven multi-head-channel points — `resid_streams`, `attn_stream_collapse` / `mlp_stream_collapse`,
`attn_stream_write` / `mlp_stream_write`, `attn_stream_mix` / `mlp_stream_mix` — exist on a
hyper-connection trunk and so on no other family. `model.points()` adds them when the model it was
loaded from has one, gated on the config's stream count rather than on an architecture name.

`resid_streams` names the whole `(tokens, n_streams, d_model)` stack, and is the point to capture when
what you want is a stream: index the result. That is a different question from the `stream` coordinate
an `Address` carries (`resid_post.5.stream-2`), which qualifies a *residual* point — eager slices it
in-process, and vLLM refuses it for the reason in `residual_basis.vllm_residual_basis`. Only
`resid_pre` / `resid_mid` / `resid_post` take it on a read; a steer takes it on `resid_streams` too,
where it means one row of a stack the worker already holds.

Two families ship the trunk, and eagerly all seven are served on both — but not at the same addresses,
which is what `facts.HYPER_CONNECTION_LAYOUTS` exists to keep straight. DeepSeek-V4's `attn_hc` /
`ffn_hc` return `(post, comb, collapsed)`; Motif 3's `mhc_attn` / `mhc_ffn` return
`(h_pre, h_post, h_res)`, so its write and mix coefficients sit one index later and its collapsed
vector is not returned at all — the block applies the coefficients itself and hands the result to the
pre-sublayer norm, whose input is therefore what `*_stream_collapse` names there.

Under vLLM all seven are served, and *where each one comes from* is worth stating precisely, because
for three of them the obvious address exists, has the right shape, and is the wrong tensor. Two come
off the decoder layer's return tuple; the other five are locals of its forward and come off the mHC
kernel calls themselves (`vllm_capture.mhc`), which is a mechanism with a different blast radius — it
is specific to the NVIDIA tree in a way a module hook would not be.

On the NVIDIA tree vLLM computes mHC as fused TileLang kernels over `nn.Parameter`s held on the
decoder layer, so there is genuinely no hyper-connection module to hook — the `MHCPreOp` / `MHCPostOp`
`CustomOp`s under `model_executor/layers/mhc.py` would be that module, but that tree calls the kernel
functions directly and never instantiates them. (The AMD and XPU trees do instantiate them, which
changes the answer completely; see below.) That is not the same as the tensors being unreachable. Its
decoder layer (`vllm/models/deepseek_v4/nvidia/model.py`, at vLLM 0.26.0) ends:

```python
x = self.ffn(x, input_ids)
return x, residual, post_mix, res_mix
```

So three of the quantities cross a forward boundary as tuple elements, and two of them are what the
engine serves: `post_mix` and `res_mix` are `mlp_stream_write` and `mlp_stream_mix`, read by index
(`vllm_capture._tree.LAYER_RETURN_INDEX`) off a hook on the layer itself.

The third, `residual`, is a `(num_tokens, hc_mult, hidden_size)` stream stack — and it is **not**
`resid_streams`, which is the single most expensive thing on this page to get wrong, because the shape
it would be checked against is exactly right. vLLM defers each sublayer's mHC *post* phase into the
next sublayer's pre-phase kernel, so at the moment the layer returns, attention has been scattered
back across the streams and the MLP has not. That tensor is the stack the MLP *read*: `resid_mid` in
stream form, one sublayer short of the block output `resid_streams` names. Measured rather than argued
— collapsing it at the FFN site reproduces the argument to `self.ffn` to 2.9e-3–4.8e-3, and at the
attention site misses the argument to `self.attn` by 0.31–0.74, which places it strictly between the
two sublayers. The block's own output stack is not late, it is *absent* from every boundary: the MLP's
scatter happens inside the **next** layer's first kernel, fused with that layer's attention collapse.
So that is where `resid_streams` at layer L is read from — the `residual` *output* of layer L+1's first
`mhc_fused_post_pre_tilelang` call, bit-identically, since that call did the scatter. What comes back
that way differs from the layer's own returned stack by 0.22–1.65 and reproduces
`mhc_post(mlp_out, output:1, mlp_stream_write, mlp_stream_mix)` — rebuilt on the client from four
independently captured tensors — to 1.9e-3–3.2e-3, which is the identity that says the deferral was
accounted for rather than merely noticed. The last layer
has no successor to defer into, and its stack is formed by the model rather than a layer, in the
standalone `mhc_post_tilelang` call after the loop; that is the one the capture reads there. The single
place this is refused is when a draft/EAGLE model is attached, because `aux_hidden_state_layers` makes
the model call `mhc_post_tilelang` several more times and which call closed the trunk stops being
decidable — earlier layers are unaffected, since theirs come off their successor's own kernel.

The two collapse points look easier still — they look like the arguments passed to `self.attn` and
`self.ffn`, the same shape of hook as `attn_in`. **That reading is wrong, and it is wrong in the worst
available way: the tensor there has the right shape, the right dtype and the right position.** On the
NVIDIA tree `attn_norm` and `ffn_norm` are fused *into* the mHC pre kernel — the layer passes
`norm_weight=self.attn_norm.weight.data` to `mhc_pre_broadcast` / `mhc_fused_post_pre` and comments
that "attn_norm is fused into mhc_pre_tilelang / mhc_fused_post_pre above" — so the argument the
sublayer receives is the collapse *already normed*. That is the engine's `attn_in` / `mlp_in`, a point
this table already has, and not `*_stream_collapse`, which the engine defines as the value one norm
earlier (`test_the_collapsed_vector_sits_one_norm_before_what_attention_reads` pins it as an exact
identity through the block's own norm).

Measured on `deepseek-ai/DeepSeek-V4-Flash` at layers 0, 21 and 42, the argument to `self.ffn`
matches `RMSNorm(collapse) * ffn_norm.weight` to 1.5e-3–5.5e-3 and differs from the collapse itself by
up to 24× (the attention site, up to 2.1e2). So on that tree the unnormed collapse is never materialized at all: it is an internal value of a
fused kernel, no module boundary holds it, and the norm dropped a per-token scalar so the normed one
cannot be inverted back. It is the one mHC point whose value is therefore *arithmetic* rather than a
tensor vLLM computed: `vllm_capture.mhc.stream_collapse` rebuilds it from the stream stack the kernel
was handed and the layer's flat `hc_{site}_{fn,base,scale}`. That function is the collapse half of
vLLM's own `mhc_pre_torch` — the reference implementation for these kernels — and is pinned against it
exactly, in bf16 at DeepSeek-V4's own epsilons, by a unit test that needs no GPU.

The same deferral is what hides the attention half of the write/mix pair. A layer's first fused call
computes the attention `post_mix` / `res_mix`; its second overwrites both, so only the MLP pair survives
to the return. `attn_stream_write` and `attn_stream_mix` therefore never touch a module boundary, and
are read off that first call instead — outputs 1 and 2 of it, so the values are the kernel's own rather
than a reconstruction. Their MLP counterparts need none of that, which is the asymmetry the two
mechanisms exist to absorb.

The wrapper is one shared installation per worker, refcounted like a hook, rebinding the four kernel
names in the model module's namespace — the same seam the per-request demux already uses on the model
runner, and the reason the patch is on `models/deepseek_v4/nvidia/model` rather than on
`kernels/mhc/tilelang`, where a `from ... import` has already copied the references. Calls are
attributed to a `(layer, site)` by the **identity** of the `fn` parameter they were passed, not by
counting calls, because the same functions also serve microbatches. A point nobody asked for is never
computed, so the collapse recompute costs nothing unless it was requested.

### Steering the three mHC points that are activations

Capture and steering ask different questions of the same address, and here they get different answers.
Four of the seven are the hyper-connection's *coefficients* — the per-stream write weights and the
Sinkhorn-normalized mixing matrix — and an additive edit to a doubly stochastic matrix leaves it neither
stochastic nor a mixture of anything, so there is no intervention there that means what a steer means.
Those four are refused at registration, with that reason. The other three are tensors on the residual
trunk, they are steerable, and neither is steerable the way a module point is: `resid_post` is written by
handing a hook's caller a different tensor, and no hook holds either of these.

**A collapse steer is a delta in normed space.** The collapse is never materialized (the norm is fused
into the same kernel), so what a write has to reach is the *normed* collapse the kernel returns, which is
the sublayer's argument. Substituting `norm(collapse + delta)` outright would work arithmetically and be
wrong in practice: the collapse is a recompute that agrees with the kernel to ~5e-3, so substituting
would impose that whole disagreement on the sublayer even for a delta of zero, making an instrumented
layer quietly different from an uninstrumented one. The write is therefore
`layer_input + (norm(c + delta) - norm(c))`, which is exactly zero when the delta is — bitwise, so an
unsteered request that merely shares a forward with a steered one is untouched.

**A stack steer runs the fused call's second half again.** `resid_streams` at layer L is *formed* inside
layer L+1's first fused call, which then immediately collapses it for L+1's attention. So an edit applied
to what that call returned would reach every later layer and miss the one reader inside the call itself —
a partial intervention under a whole intervention's name. vLLM ships that call's second half separately
and its signature composes, so the wrapper lets the fused call run, hands the stack to the steer, and
then runs `mhc_pre_tilelang` on the edited stack to redo the collapse.

Which half gets re-run, and for whom, is the part that took a checkpoint to settle. Running `post` and
then `pre` in place of the fused call — the obvious decomposition, and bitwise exact at float32 on random
weights, which is what `plans/scripts/verify_dsv4_mhc_decompose.py` measures — disagrees with the fused
kernel by up to 2e-2 relative in bf16 on V4-Flash. That is far more than the reduction-order noise
batching already costs, and it would land on requests that asked for nothing: a steer of coefficient 0
would be detectably different from no steer, and a request merely co-scheduled with a steered one would
move. So the fused call always runs first and is authoritative, the re-run is reached only once a recorder
has returned a *different* tensor, and its output is written back **per row** — the rows whose stack
changed take the re-run and every other row keeps the kernel's own numbers. A layer nobody steered pays
nothing, a steered one pays one extra kernel, and the parameters for it are read off the pre phase's own
signature, so a vLLM whose pre phase grows a parameter the fused call lacks refuses rather than
mis-calling a kernel. The last layer needs none of this: nothing collapses its stack again, so replacing
the model's closing `mhc_post` output *is* the intervention.

Both mechanisms carry the optional `stream` coordinate, which on a write means only that the delta lands
in one row of a stack the worker already holds — a weaker claim than the one a read makes, which is why
vLLM serves it here while still refusing to *read* a single stream through the resid points.

A jlens steer/ablate/swap goes through the same gate and the same sites: its spec carries a `point`
(defaulting to `resid_post`, which is what it has always meant on a conventional trunk) and the
per-request path keys the intervention by site rather than by layer. That is what makes the lens usable
on this family at all — `resid_post` there names the whole stack, which no sublayer reads, so the tensor
a swap wants is a collapse. The global `set_lens_intervention` stays pinned to the decoder layer's output
and refuses a spec that names a point, rather than ignoring it.

### On the AMD and XPU trees they are ordinary modules

This inverts the usual assumption that the NVIDIA path is the best-supported one. `MHCPreOp`,
`MHCPostOp`, `MHCFusedPostPreOp` and `HCHeadOp` in `model_executor/layers/mhc.py` are all `CustomOp`
and therefore `nn.Module`, each with a `forward_native` torch implementation — and
`models/deepseek_v4/amd/model.py` and `xpu/model.py` **do** instantiate them
(`self.mhc_pre = MHCPreOp()`, `self.mhc_post = MHCPostOp()`,
`self.mhc_fused_post_pre = MHCFusedPostPreOp()`). Only the NVIDIA tree calls the kernel functions
directly and never instantiates them.

The consequence is not merely that hooking is easier there. Those trees apply the norm as a *separate*
call after the mHC op — `x = self.attn_norm(x)` on its own line, then `x = self.attn(...)` — so on AMD
and XPU the **unnormed** collapse is a genuine module output, exactly as the engine defines the point,
while on NVIDIA it does not exist at any boundary. A resolver written against one tree is therefore
not just mis-addressed on the others; it is answering a differently-defined question.

What is wired declares itself NVIDIA-only rather than detecting the tree: the kernel wrapper looks for
the four TileLang names in the layer's own module and refuses with that reason when they are absent, so
on AMD or XPU the five kernel points are **unwired, not unreachable** — and cheaper to wire there than
here, since each is an ordinary module output. The two return-tuple points are unaffected: all three
trees end the layer with the same `return x, residual, post_mix, res_mix`.

### What is now measured, and what is still missing

Everything above the AMD/XPU section has been run, not read, on `deepseek-ai/DeepSeek-V4-Flash` under
vLLM 0.26.0 on a single B200 with `enforce_eager` — see `plans/scripts/verify_dsv4_mhc_vllm.py` and
`plans/scripts/compare_dsv4_mhc_eager.py`. The released checkpoint needs no config translation and no
weight shim, because vLLM's V4 was written against DeepSeek's own release schema; the mHC parameters
are stored unquantized, which is what lets the eager comparison read them straight out of the
safetensors and skip the fp8 path entirely.

| quantity | address | agreement with transformers' own module, identical weights |
| --- | --- | --- |
| `mlp_stream_write` | layer `output:2`, `(13, 4, 1)` → squeezed to `(13, 4)` | 6e-4 against `post` |
| `mlp_stream_mix` | layer `output:3`, `(13, 4, 4)` | 4e-4 against `comb` |
| `resid_streams` | layer L+1's first fused call, `output:0`; the closing `mhc_post` for the last layer | the kernel's own tensor. Layer `output:1` is `resid_mid`-in-streams and is **not** this |
| `attn_stream_write` / `attn_stream_mix` | layer L's first fused call, `output:1` / `output:2` | the kernel's own tensors, before the second call overwrites them |
| `*_stream_collapse` | rebuilt from that call's input stack | the collapse half of vLLM's `mhc_pre_torch`, pinned to it exactly; 5e-3 against transformers |

The rows served off the return tuple were then captured a second time through `VLLMModel` itself rather
than by hand — point table, address grammar, per-request demux, payload round trip and the client-side
assertions — and came back **bit-identical** to the hand-registered hooks on the same prompt
(`plans/scripts/verify_dsv4_engine_capture.py`). That is what licenses `VllmSupport.HOOKS` on them:
the tensor the shipped path returns is the one that was compared against transformers, not a
same-shaped neighbour. The same script covers the five kernel points in the same run, and checks the
three things only the kernel mechanism can get wrong: that `resid_streams` at L is the block output
rather than layer L's `output:1` (0.22–1.65 apart, and 1.9e-3–3.2e-3 from the post phase rebuilt
client-side), that each collapse is not the `attn_in`/`mlp_in` one norm after it (0.85–2.1e2 apart),
and that the attention write/mix pair is not the MLP pair its layer overwrote it with (0.08–1.0 apart).

One property of the mix matrix has to be stated exactly, because the loose version of it passes on
every fixture and fails on the real checkpoint: it is **column** stochastic to 1e-6 and only roughly
row stochastic — up to 7e-2 at `hidden_size = 4096`. Sinkhorn alternates the two normalizations and
ends on a column one, and the columns are also the axis the post phase contracts
(`out[..., j, :] = Σᵢ mix[..., i, j] · streams[..., i, :]`), so that axis is both the exact one and the
load-bearing one. The row error grows with width rather than with training, so a fixture narrow enough
to run cheaply hides it completely.

Steering the three activation points is measured on the same checkpoint by
`plans/scripts/verify_dsv4_mhc_steer.py`, which is arranged around the claims that would be expensive to
get wrong: that a coefficient-0 steer leaves every witness *bitwise* unchanged at all three points, that
a steer at a collapse moves that sublayer's own input and nothing before it, and that a steer at
`resid_streams` at layer L moves `attn_in` at L+1 — the reader inside the very call that formed the stack,
which is what distinguishes the mechanism from a plausible-looking one that writes too late. The same
script checks the last layer's stack on the model's closing call, a steer confined to one stream, a request
batched alongside a steered one, the jlens path aimed at a collapse, and what the extra kernel costs.

What is still missing is the AMD and XPU trees, where the five kernel points are unwired and would be
module outputs, and steering the four coefficients — which is not an unimplemented feature but a
refusal, for the reason above. It happens at registration, on the client where the spec is fixable and
again on the worker, because a forward hook cannot refuse without taking the worker down.

The parity that has been established is scoped, and the scope matters: the mHC tensors were compared
against the reference implementation *given the same stream stack*, which settles what each point is.
It does not compare logits between the two engines, which would need eager to hold the model too, so it
says nothing yet about whether the two agree after 43 layers of otherwise different kernels. Everything
is also NVIDIA-verified only. The return-tuple address happens to be vendor-independent — all three of
`models/deepseek_v4/{nvidia,amd,xpu}/model.py` end the layer with the same
`return x, residual, post_mix, res_mix` — but only one tree has been run, and the guard that admits the
points keys on the layer carrying an `hc_ffn_fn` parameter, which all three spell identically.

Everything above is about vLLM's DeepSeek-V4, which is the only mHC implementation upstream vLLM has:
Motif 3 is served through its authors' fork, so for that family the question is not what to hook but
which tree to read at all, and it is not one this engine tracks. On the eager side both are served,
which is where the layouts above come from.

### The seven mHC points in TransformerLens

TransformerLens 3 is the only other stack that names these, and it names all of them. Its DeepSeek-V4
adapter — `model_bridge/supported_architectures/deepseek_v4.py`, registered for
`DeepseekV4ForCausalLM` — wraps each hyper-connection module in a bridge that hooks all three of its
outputs separately rather than collapsing them, so each of the seven has a counterpart.
`HookedTransformer` has none of them and nnterp has no accessor for any, which is why they are absent
from the table at the top of this page rather than present with two empty columns.

| canonical point        | interp-engine                      | TransformerLens 3            | nnterp |
| ---------------------- | ---------------------------------- | ---------------------------- | ------ |
| `resid_streams`        | `cache["resid_streams", 5]`        | `blocks.5.hook_out`          | —      |
| `attn_stream_collapse` | `cache["attn_stream_collapse", 5]` | `blocks.5.attn_hc.hook_out`  | —      |
| `mlp_stream_collapse`  | `cache["mlp_stream_collapse", 5]`  | `blocks.5.mlp_hc.hook_out`   | —      |
| `attn_stream_write`    | `cache["attn_stream_write", 5]`    | `blocks.5.attn_hc.hook_post` | —      |
| `mlp_stream_write`     | `cache["mlp_stream_write", 5]`     | `blocks.5.mlp_hc.hook_post`  | —      |
| `attn_stream_mix`      | `cache["attn_stream_mix", 5]`      | `blocks.5.attn_hc.hook_comb` | —      |
| `mlp_stream_mix`       | `cache["mlp_stream_mix", 5]`       | `blocks.5.mlp_hc.hook_comb`  | —      |

Three of those rows are easy to get wrong by eye, and `interp_engine.mappers` encodes all three.

**`blocks.5.hook_out` is model-dependent, exactly as `hook_mlp_out` is.** TransformerLens' `BlockBridge`
aliases `hook_resid_post` onto `hook_out`, so the two names are one tensor on every conventional trunk.
The V4 bridge clears the aliases and declares `hook_out_is_single_residual_stream = False`, which makes
that same name the block's whole `(batch, pos, hc_mult, d_model)` stack — and deletes `hook_resid_post`,
which it omits for the reason this engine refuses the bare residual points on such a trunk. Both
readings are `d_model` in the last axis, so a string-only translation resolves and is wrong by a rank.
This is the second of the page's two model-aware axes, and the remedy is the same one:

```python
from interp_engine import load_model, tlens_hook_to_point

model = load_model("Qwen/Qwen3-1.7B")
tlens_hook_to_point("blocks.5.hook_out", model)  # -> Address("resid_post", 5)
```

On a hyper-connection trunk the same call returns `Address("resid_streams", 5)`. Without a model it
answers `resid_post`, the reading that is right everywhere else.

**A hyper-connection's `hook_out` is the pre-norm vector**, which is what makes it `*_stream_collapse`
rather than `attn_in`: the block computes `post, comb, collapsed = self.attn_hc(streams)` and only then
calls `self.self_attn(self.input_layernorm(collapsed))`, so the norm is downstream of the hook. The two
differ by exactly one norm and share a shape.

**`mlp_hc` is TransformerLens' name for the module HuggingFace calls `ffn_hc`.** The attention site
agrees on `attn_hc`; only the FFN one is renamed, so the obvious spelling is unresolvable in one
direction and silently the wrong site in the other.

The two mHC hooks that carry a stack *in* are refused rather than mapped, and they are in the table
below with the reason. Neither has a canonical point, and both have `resid_streams`' shape.

## Hooks TransformerLens has that we do not name

Reachable in interp-engine — where it is reachable — through the open-set escape hatch: any dotted module
path captures that module's **output** (`cache["model.layers.5.self_attn.q_proj"]`).

| TransformerLens                                                     | what it is                                                      | why we don't map it                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `blocks.5.ln1.hook_scale`, `blocks.5.ln2.hook_scale`                | the RMS/LN scale factor                                         | It is an intermediate of the norm's arithmetic, not a module boundary, so there is nothing to hook. `ln2`'s boundaries are `resid_mid` (input) and `mlp_in` (output); `capture.rms_norm_parts` recovers the scale between them                                                                                                                                          |
| `blocks.5.ln1.hook_normalized`, `blocks.5.ln2.hook_normalized`      | the normalized residual, before the norm's gain                 | Not a module boundary either: TL fires it between the divide and the weight multiply. Unmapped but **not unsupported** — `mappers.tlens_normalized_hook` gives the point to capture and `capture.pre_gain_normalized` the arithmetic; see [the note above](#a-norms-hook_normalized-is-not-the-sublayer-input), which matters because real transcoders are trained here |
| `blocks.5.attn.q_norm.hook_scale`, `.hook_normalized`               | inside the QK-norm                                              | Same reason one level down: `q_norm_in`/`q_norm_out` are the boundaries and `rms_norm_parts` is the decomposition. TL's `hook_normalized` also sits between the divide and the gain multiply, so it is neither of those two points                                                                                                                                      |
| `blocks.5.attn.hook_q`, `.hook_k`                                   | per-head q/k, pre-RoPE                                          | Reachable under other names rather than unmapped: on a QK-norm family they are `q_norm_in` (pre-norm) and `q_norm_out` (post-norm, pre-RoPE), per head where the family is; elsewhere a dotted path to `self_attn.q_proj` / `k_proj`, flattened                                                                                                                         |
| `blocks.5.attn.hook_rot_q`, `.hook_rot_k`                           | q/k after RoPE                                                  | No such tensor crosses a module boundary: `transformers` applies RoPE inside the attention forward                                                                                                                                                                                                                                                                      |
| `blocks.5.attn.hook_result`                                         | per-head output _after_ `W_O`                                   | It is `n_heads` times the size of `z`, which is why TL gates it behind a flag too. Computed on demand instead: `capture.head_contributions(model, cache, 5)` applies the split `W_O` to a captured `z`                                                                                                                                                                  |
| `blocks.5.hook_q_input`, `.hook_k_input`, `.hook_v_input`           | per-head writable inputs (`use_split_qkv_input`)                | A TransformerLens parameterization rather than a place in the HF forward — the tensor exists only because TL splits the input per head to make it writable                                                                                                                                                                                                              |
| `blocks.5.hook_attn_in`, `blocks.5.hook_mlp_in`                     | writable sublayer inputs (`use_attn_in`)                        | Already named: `attn_in` / `mlp_in` are the same tensors, and writing to them needs no separate hook here                                                                                                                                                                                                                                                               |
| `blocks.5.mlp.hook_mid`                                             | post-activation, pre-LayerNorm, on a SoLU model                 | No HF checkpoint in scope has an activation-internal norm, so the point would never resolve                                                                                                                                                                                                                                                                             |
| `blocks.5.mlp.hook_gate`                                            | one expert's SwiGLU gate, at `blocks.5.mlp.experts.3.hook_gate` | **Not the router**, despite the name, so mapping it to anything routing-shaped would be wrong. Our `mlp_pre` is the same tensor on a dense MLP; per expert it is not a module output at all on the families that fuse their expert banks                                                                                                                                |
| `blocks.5.mlp.hook_expert_weights`                                  | the softmax over _all_ experts, pre-top-k                       | It is a different tensor from our `expert_weights` (post-top-k, `k` wide), and mapping two different tensors to one name is the failure this table exists to prevent. Write it as `softmax(cache["router_logits", 5])`, exact for the softmax-routed families                                                                                                           |
| `blocks.5.attn_hc.hook_in`                                          | the stream stack entering the block                             | A real tensor of `resid_streams`' shape, and the *previous* block's: the same stack one layer index lower, and at layer 0 the embedding stack, which is no block's output. Capture `resid_streams` at layer 4 instead of naming this at layer 5                                                                                                                         |
| `blocks.5.mlp_hc.hook_in`                                           | the stream stack between the sublayers                          | Attention has written back and the MLP has not, so it is `resid_mid` in stream form — one sublayer short of `resid_streams`, whose shape it shares. This engine refuses `resid_mid` on a hyper-connection trunk rather than naming this, so there is nothing to map it to; it is the same tensor vLLM's decoder layer returns, [as above](#all-seven-mhc-points-are-served-on-vllm-by-two-mechanisms-and-neither-is-a-module-hook) |
| `hook_tokens`, `hook_pos_embed`                                     | token ids, positional embeddings                                | The ids are the caller's own input, and a RoPE family has no positional-embedding module to hook                                                                                                                                                                                                                                                                        |
| `hook_cross_attn_in`, `hook_cross_attn_out`, `hook_resid_mid_cross` | encoder-decoder cross attention                                 | Out of scope: decoder-only models here                                                                                                                                                                                                                                                                                                                                  |
| `hook_nsp_out`, `hook_pooler_out`, `hook_token_type_embed`          | BERT-family heads                                               | Out of scope: no classification or pooling heads here                                                                                                                                                                                                                                                                                                                   |

## nnterp accessors that are not hook points

`layers[5]`, `attentions[5]`, `mlps[5]` return the _modules_ (so `.output`, `.input`, and assignment for
steering); `logits`, `next_token_probs`, `input_ids`, `attention_mask` are whole-forward quantities; and
`skip_layer(5)` / `steer(...)` are interventions rather than reads. interp-engine's equivalents of the
interventions live in `interp_engine.steer`.

## Addressing and shape, which do not translate for free

|                                     | interp-engine                                                                                                                                                                                                   | TransformerLens                                                                                               | nnterp                                                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| how a point is addressed            | `(point, layer)` tuple into `cache`                                                                                                                                                                             | one dotted string per hook                                                                                    | accessor indexed by layer, inside a trace                              |
| per-head tensors                    | flattened: `z` is `[batch, pos, n_heads * head_dim]`. The QK-norm points are the exception, being per head wherever the family's norm is; `capture.head_contributions` returns `[batch, pos, n_heads, d_model]` | per-head: `[batch, pos, head, d_head]`                                                                        | n/a                                                                    |
| `value` head count                  | `v_proj`'s own output, so `[batch, pos, n_kv_heads * head_dim]` on a GQA checkpoint                                                                                                                             | `hook_v` fires _before_ `repeat_interleave`, so also KV heads — but shaped `[batch, pos, n_kv_heads, d_head]` | n/a                                                                    |
| `attn_probs` / `attn_scores` layout | `[batch, n_heads, query, key]` (HF `output_attentions`)                                                                                                                                                         | `[batch, head, query, key]`                                                                                   | as HF (probs only)                                                     |
| MoE routing layout                  | `[batch, pos, ...]`, restored from the `[batch * pos, ...]` the router returns so every point in a cache indexes alike                                                                                          | `[batch, pos, ...]`                                                                                           | n/a                                                                    |
| how many points per pass            | any number, one forward                                                                                                                                                                                         | any number, one forward                                                                                       | one point per trace in our adapter (an "out of order envoy" otherwise) |
| the point set                       | **open**: unknown names fall through to a dotted module path                                                                                                                                                    | closed: the hooks the components define                                                                       | closed: the standardized accessors                                     |

## Where this lives in code

| what                                                                                                                           | where                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| name translation, both directions, raising where there is no equivalent                                                        | `interp_engine/mappers.py`                                             |
| the canonical point resolver (module + input/output, per architecture)                                                         | `interp_engine/model.py` (`resolve_point`)                             |
| which points vLLM's worker hooks can serve                                                                                     | `interp_engine/vllm_capture/_tree.py` (`HOOK_CAPTURE_POINTS`)          |
| `attn_probs` off-kernel recompute for vLLM                                                                                     | `interp_engine/vllm_capture/attn.py` (`recompute_attn_probs`)          |
| the pre-softmax scores, and the registry swap that reaches them                                                                | `interp_engine/attn_scores.py`                                         |
| per-architecture module paths and quirks (sandwich norm, pre-MLP norm, gated attention, QK-norm shape, MLP gating, the router) | `interp_engine/facts.py`, `interp_engine/arch.py`                      |
| a norm's scale and gain, for the parts that are not boundaries                                                                 | `interp_engine/capture.py` (`rms_norm_parts`)                          |
| per-head residual contributions, and dense expert assignments                                                                  | `interp_engine/capture.py` (`head_contributions`, `expert_assignment`) |
