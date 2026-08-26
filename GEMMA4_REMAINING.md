# Gemma 4: what is still unfixed

Three gaps left after the `gemma_4_fixes` branch. Each needs a decision about what a point *means*
on this family, which is why none of them is a patch.

Ground truth for everything below is
`transformers/models/gemma4/modeling_gemma4.py` — `Gemma4TextDecoderLayer.forward`,
`Gemma4TextAttention.__init__`, and `Gemma4TextRouter.forward`.

## Scope: which checkpoint has which problem

| SKU | layers | MoE layers | `full_attention` layers | `attention_k_eq_v` | KV sharing |
| --- | --- | --- | --- | --- | --- |
| `google/gemma-4-E2B` | 35 | 0 | 7 | off | from layer 15 |
| `google/gemma-4-E4B` | 42 | 0 | 7 | off | from layer 24 |
| `google/gemma-4-12B-it` | 48 | 0 | 8 | **on** | none |
| `google/gemma-4-26B-A4B-it` | 30 | **30 (all)** | 5 | **on** | none |
| `google/gemma-4-31B` | 60 | 0 | 10 | **on** | none |

The 26B is the only Gemma 4 with a router. E2B and E4B are unaffected by all three items.

---

## 1. `mlp_out` returns half the feed-forward on the 26B

**Severity: silently wrong numbers. This is the one that matters.**

Gemma 4 does not replace the MLP with an expert bank. It keeps `self.mlp` and hangs the router and
experts beside it on the block, then sums the branches:

```python
hidden_states = self.mlp(self.pre_feedforward_layernorm(residual))
if self.enable_moe_block:
    hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
    _, top_k_weights, top_k_index = self.router(residual_flat)
    hidden_states_2 = self.experts(self.pre_feedforward_layernorm_2(residual_flat), ...)
    hidden_states = hidden_states_1 + self.post_feedforward_layernorm_2(hidden_states_2)
hidden_states = self.post_feedforward_layernorm(hidden_states)
hidden_states = residual + hidden_states
```

`mlp_out` taps `layer.mlp`, so it returns `hidden_states` before the routed branch is added — the
dense half of a two-branch feed-forward. Correctly shaped `[tokens, d_model]`, no warning, no tell.
On the 26B every one of the 30 layers is affected, because it has no dense prefix.

`mlp_in` has the milder version of the same problem: it is the dense branch's input
(`pre_feedforward_layernorm(residual)`), while the expert branch reads a *different* norm of the
same residual (`pre_feedforward_layernorm_2`). Right for the dense branch, not the block.

**There is already a correct point, which is what keeps this from being critical.**
`mlp_out_post` resolves to `post_feedforward_layernorm`, whose input is the summed
`hidden_states_1 + hidden_states_2` and whose output is the true residual contribution. So
`resid_post == resid_mid + mlp_out_post` still holds, and anything built on the residual
decomposition is correct today. Only the raw `mlp_out` is half.

**Decision needed.** Either `mlp_out` means the summed feed-forward here — which has no module
boundary, since the sum happens in the block's `forward`, so it would need the same treatment as
another kernel-local point — or it is refused on a `dense_mlp_beside_experts` layer with a message
pointing at `mlp_out_post`. Refusing costs a point; returning the wrong tensor costs trust.

## 2. The routing points do not resolve at all on the 26B

**Severity: loud error. Nothing wrong is returned.**

`arch.moe_router` looks for the router under `layer.mlp`, and Gemma 4 hangs it on the *block*
(`layer.router`, beside `layer.mlp`). `facts.moe_router_attr` finds nothing on the
`Gemma4TextMLP`, so `router_logits`, `expert_weights` and `expert_indices` raise:

```
AttributeError: No router submodule found on layer N's Gemma4TextMLP
(Gemma4ForConditionalGeneration); tried ('gate', 'router')
```

Annoying, discoverable, and safe. Nobody gets a bad number.

**Decision needed.** `moe_router` has to look at the block as well as the MLP module. Note that
`MOE_ROUTER_ATTRS` already contains `router`, so widening the search to the block would find it —
but that search must not become a general fallback, or it will reach past a nested MLP on some other
family and return a sibling's router.

**Landmine for whoever fixes it.** `Gemma4TextRouter.forward` returns
`(router_probabilities, top_k_weights, top_k_index)`. Element 0 is already softmaxed over all 128
experts, so it matches the default tuple *order* while not being logits, and
`facts.assert_routing_shapes` cannot catch it because the width is right (128 == `n_experts`). Every
other family in `ROUTER_OUTPUTS` puts real logits there. This needs either a family entry recording
that element 0 is post-softmax, or the point has to be served from the router's `proj` output
instead.

## 3. `attention_k_eq_v` layers have no `v_proj`

**Severity: loud error with a misleading message. Already live on the 31B.**

On a `full_attention` layer of the 26B, 31B or 12B, `Gemma4TextAttention` is built with
`self.v_proj = None` and takes the key projection's output as the value:

```python
self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
self.v_proj = nn.Linear(...) if not self.use_alternative_attention else None
...
value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
```

The two are the same projection read through different norms, not the same tensor: the value is
`v_norm(k_proj(h))` (a scale-free RMSNorm, no RoPE) while the key is `rope(k_norm(k_proj(h)))`.

`_first_attr` treats a present-but-`None` attribute as absent, so `ArchSpec.v_proj` returns `None` —
which is its signal for "fused QKV, the caller must split" — on a family whose `fused_qkv` is false.
`model.py` then falls through to the multi-head-latent-attention refusal, so asking for `value` on
layer 5 of the 26B raises a `ValueError` that blames MLA and names DeepSeek. Wrong explanation, and
it sends the reader looking for a kv latent that does not exist.

This is **not** the KV-sharing case the engine already handles. Those three SKUs set
`num_kv_shared_layers: 0`, so no layer reuses another's keys — the value is genuinely computed
*here*, just not by a module of its own.

**Decision needed.** The layer needs a third answer. `value` is capturable as the `v_norm` module's
output; `z` already works either way. At minimum the refusal must name `attention_k_eq_v` rather
than MLA. Affects 5 of 30 layers on the 26B, 10 of 60 on the 31B, 8 of 48 on the 12B.

**Why nothing caught it.** The 31B has been in the sweep since before this branch, but was swept
without the value points.

---

## Suggested order

1. **Item 3 first.** It is already wrong on a checkpoint in the sweep, and the misleading message is
   cheap to fix even if the full capture path takes longer.
2. **Item 1 second.** It is the only silent one. Even the refusal-with-a-pointer version closes it.
3. **Item 2 last.** It is loud and self-describing, so it costs a user time rather than correctness —
   but do not fix the resolution without the tuple-order entry, or item 2 turns into another item 1.
