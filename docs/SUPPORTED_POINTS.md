# Supported points

The 34 canonical points, and whether each backend can serve one. The eager backend captures every
point below; the vLLM backend serves 28 of them, 2 of those by recompute rather than by a hook.

**Every point name below opens that point on the diagram** at [interp-engine.org](https://interp-engine.org),
on an architecture that actually has it — a router on a sparse family, `attn_out_post` on a sandwich-norm
one, the stream points on DeepSeek-V4 — and on a layer that family draws. Coming from another engine?
[ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) defines each point and translates it to TransformerLens,
nnsight and nnterp.

| point                                                 | width                   | eager | vLLM | notes                                                                                                                                                                                                                                     |
| ----------------------------------------------------- | ----------------------- | :---: | :--: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`embeddings`][embeddings]                            | `d_model`               |  ✅   |  ✅  | trunk-level, so addressed with no layer index; distinct from `resid_pre` at layer 0 only where the trunk adds positional embeddings or scales the embedding                                                                               |
| [`resid_pre`][resid_pre]                              | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`attn_in`][attn_in]                                  | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`q_norm_in`][q_norm_in] / [`q_norm_out`][q_norm_out] | `n_heads * head_dim`    |  ✅   |  ✅  | head-sharded, so single-GPU only                                                                                                                                                                                                          |
| [`k_norm_in`][k_norm_in] / [`k_norm_out`][k_norm_out] | `n_kv_heads * head_dim` |  ✅   |  ✅  | head-sharded, so single-GPU only                                                                                                                                                                                                          |
| [`value`][value]                                      | `n_heads * head_dim`    |  ✅   |  ✅  | head-sharded, so single-GPU only                                                                                                                                                                                                          |
| [`attn_scores`][attn_scores]                          | `n_heads * query * key` |  ✅   |  ♻️  | no module boundary holds the pre-softmax matrix on **either** backend; vLLM rebuilds it from captured post-RoPE q/k                                                                                                                       |
| [`attn_probs`][attn_probs]                            | `n_heads * query * key` |  ✅   |  ♻️  | fused paged attention never materializes the probabilities; same recompute                                                                                                                                                                |
| [`z`][z]                                              | `n_heads * head_dim`    |  ✅   |  ✅  | head-sharded, so single-GPU only                                                                                                                                                                                                          |
| [`attn_gate`][attn_gate]                              | `n_heads * head_dim`    |  ✅   |  ❌  | unimplemented — a real module on both trees                                                                                                                                                                                               |
| [`attn_out`][attn_out]                                | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`attn_out_post`][attn_out_post]                      | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`resid_mid`][resid_mid]                              | `d_model`               |  ✅   |  ✅  | capture works everywhere; _steering_ it is refused on families where vLLM adds the residual before the norm                                                                                                                               |
| [`mlp_in`][mlp_in]                                    | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`mlp_pre`][mlp_pre]                                  | `d_mlp`                 |  ✅   |  ❌  | unreachable — vLLM fuses `gate_proj` and `up_proj` into one `gate_up_proj`, so neither branch is a module output                                                                                                                          |
| [`mlp_pre_linear`][mlp_pre_linear]                    | `d_mlp`                 |  ✅   |  ❌  | as `mlp_pre`; gated MLPs only                                                                                                                                                                                                             |
| [`mlp_act`][mlp_act]                                  | `d_mlp`                 |  ✅   |  ✅  | neuron-sharded, so single-GPU only                                                                                                                                                                                                        |
| [`router_logits`][router_logits]                      | `n_experts`             |  ✅   |  ✅  | replicated gate, so it survives tensor parallelism                                                                                                                                                                                        |
| [`expert_weights`][expert_weights]                    | `n_experts`             |  ✅   |  ❌  | unreachable — the top-k happens inside the FusedMoE kernel, which returns the combined output with the selection never materialized                                                                                                       |
| [`expert_indices`][expert_indices]                    | `n_experts`             |  ✅   |  ❌  | as `expert_weights`                                                                                                                                                                                                                       |
| [`mlp_out`][mlp_out]                                  | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`mlp_out_post`][mlp_out_post]                        | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`resid_post`][resid_post]                            | `d_model`               |  ✅   |  ✅  |                                                                                                                                                                                                                                           |
| [`final_norm`][final_norm]                            | `d_model`               |  ✅   |  ✅  | trunk-level, so addressed with no layer index; runs over every position, not just the ones being decoded                                                                                                                                  |
| [`lm_head`][lm_head]                                  | `vocab_size`            |  ✅   |  ❌  | unreachable as a bare unembed — vLLM's `compute_logits` folds scaling and softcapping in, so hooking it returns something other than `W_U @ x`                                                                                            |
| [`resid_streams`][resid_streams]                      | `n_residual_streams`    |  ✅   |  ✅  | the block's own output stack. On vLLM this is **not** the stack the decoder layer returns, which is a sublayer earlier — read off the next layer's first kernel instead. Steerable                                                        |
| [`attn_stream_collapse`][attn_stream_collapse]        | `d_model`               |  ✅   |  ✅  | the one `d_model` vector attention reads, so the tensor an SAE or a steering vector wants on such a trunk. One norm before `attn_in`, and on vLLM that norm is fused into the kernel, so this is recomputed rather than hooked. Steerable |
| [`mlp_stream_collapse`][mlp_stream_collapse]          | `d_model`               |  ✅   |  ✅  | the same for the FFN, one norm before `mlp_in`                                                                                                                                                                                            |
| [`attn_stream_write`][attn_stream_write]              | `n_residual_streams`    |  ✅   |  ✅  | the per-stream weights attention's output is written back with. Capture-only: a coefficient rather than an activation, so a steer is refused rather than unimplemented                                                                    |
| [`attn_stream_mix`][attn_stream_mix]                  | `n_residual_streams`    |  ✅   |  ✅  | the doubly-stochastic matrix that remixes the streams after that write. Capture-only, as `attn_stream_write`; both are overwritten before the layer returns and come off its first kernel call                                            |
| [`mlp_stream_write`][mlp_stream_write]                | `n_residual_streams`    |  ✅   |  ✅  | the MLP's counterpart, and one of the two mHC points that reach a module boundary at all. Capture-only                                                                                                                                    |
| [`mlp_stream_mix`][mlp_stream_mix]                    | `n_residual_streams`    |  ✅   |  ✅  | the MLP's mixing matrix, the other one. Exactly column-stochastic and only roughly row-stochastic, which matters if you check it. Capture-only                                                                                            |

✅ served · ♻️ served by recompute rather than a hook · ❌ not served

## What a ❌ means

A ❌ is one of two things, and which decides whether to file a bug or switch backend: **unimplemented**,
where the module is right there on vLLM's tree and nobody wired the point up, or **unreachable**, where
a fused kernel ate the tensor and no module boundary holds it. Ask the code rather than this table if
you are branching on it — `points.vllm_hookable()` is the served set, `points.reason(name)` is the
sentence for one refusal, and `model.points()` is what a loaded model has.

## Tensor parallelism narrows the vLLM column further

The capture path reads rank 0's payload alone, so a point whose last axis vLLM shards comes back as a
slice: `z`, `value`, `mlp_act` and the four QK-norm points are refused on a multi-GPU pod rather than
returned short, and so is the attention recompute (q/k/v are head-sharded). Everything `d_model` wide
is all-reduced before the hook sees it, and `router_logits` comes off a replicated gate, so those are
unaffected.

## The last seven rows need a hyper-connection trunk

Their residual is a stack of parallel streams rather than one vector — DeepSeek-V4 and Motif 3 today,
and any family whose config reports more than one stream, since the rows are gated on that count
rather than on an architecture name. Two ways to say "one stream" live here and they are not
interchangeable. `resid_streams` is the stack itself: `Address("resid_streams", 5)`, or
`resid_streams.5` as a string, comes back `(tokens, n_streams, d_model)` and stream _k_ is
`stack[:, k]`. The `stream` coordinate is the other way round and qualifies a _residual_ point
instead, so `resid_post.5.stream-2` is eager-only — no residual hook on such a trunk reconstructs a
single stream, and vLLM's refusal points at `resid_streams` and the collapse points. A steer is the
one place the coordinate lands on the stack, where it means one row of a tensor the worker already
holds: `SteeringSpec(point="resid_streams", stream=2)`.

None of the seven is a module hook under vLLM. Two are elements of the decoder layer's return and five
are locals of its forward reached by wrapping the mHC kernel calls, which ties them to vLLM's NVIDIA
tree in a way a module hook would not be — and for three of them the obvious address has the right shape
and the wrong tensor. Those measurements, and how the three that are activations are steered, are in
[ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md#all-seven-mhc-points-are-served-on-vllm-by-two-mechanisms-and-neither-is-a-module-hook).

[← back to the interp-engine README](../README.md)

[embeddings]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=embeddings
[resid_pre]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=resid_pre.2
[attn_in]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=attn_in.2
[q_norm_in]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=q_norm_in.2
[q_norm_out]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=q_norm_out.2
[k_norm_in]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=k_norm_in.2
[k_norm_out]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=k_norm_out.2
[value]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=value.2
[attn_scores]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=attn_scores.2
[attn_probs]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=attn_probs.2
[z]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=z.2
[attn_gate]: https://interp-engine.org/?arch=LagunaForCausalLM&point=attn_gate.2
[attn_out]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=attn_out.2
[attn_out_post]: https://interp-engine.org/?arch=Gemma3ForCausalLM&point=attn_out_post.2
[resid_mid]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=resid_mid.2
[mlp_in]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=mlp_in.2
[mlp_pre]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=mlp_pre.2
[mlp_pre_linear]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=mlp_pre_linear.2
[mlp_act]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=mlp_act.2
[router_logits]: https://interp-engine.org/?arch=Qwen3MoeForCausalLM&point=router_logits.2
[expert_weights]: https://interp-engine.org/?arch=Qwen3MoeForCausalLM&point=expert_weights.2
[expert_indices]: https://interp-engine.org/?arch=Qwen3MoeForCausalLM&point=expert_indices.2
[mlp_out]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=mlp_out.2
[mlp_out_post]: https://interp-engine.org/?arch=Gemma3ForCausalLM&point=mlp_out_post.2
[resid_post]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=resid_post.2
[final_norm]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=final_norm
[lm_head]: https://interp-engine.org/?arch=Qwen3ForCausalLM&point=lm_head
[resid_streams]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=resid_streams.2
[attn_stream_collapse]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=attn_stream_collapse.2
[mlp_stream_collapse]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=mlp_stream_collapse.2
[attn_stream_write]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=attn_stream_write.2
[attn_stream_mix]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=attn_stream_mix.2
[mlp_stream_write]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=mlp_stream_write.2
[mlp_stream_mix]: https://interp-engine.org/?arch=DeepseekV4ForCausalLM&point=mlp_stream_mix.2
