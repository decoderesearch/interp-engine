# `deepseek-ai/DeepSeek-V2-Lite` — cross-engine results

Every engine's capture of `deepseek-ai/DeepSeek-V2-Lite`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 13, 20, 26.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 34 | 0 | 0 | 7 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 32 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: deepseek-ai/DeepSeek-V2-Lite not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever… | bfloat16 | v3.7.0 | 0 | 0 | 0 | 33 |
| [tlens_v3](tlens_v3.json) | [⚠️](https://github.com/TransformerLensOrg/TransformerLens/issues/1645) | ok | bfloat16 | v3.7.0 | 12 | 12 | 2 | 7 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 23 | 0 | 0 | 6 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `mlp_out`<br>layer 13 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `mlp_out`<br>layer 20 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `mlp_out`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ⚠️ | — |
| `mlp_out_post`<br>layer 13 | ref | ✅ | ✅ | ⚠️ | — |
| `mlp_out_post`<br>layer 20 | ref | ✅ | ✅ | ⚠️ | — |
| `mlp_out_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 13 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `attn_out`<br>layer 20 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `attn_out`<br>layer 26 | ref | ✅ | ✅ | ⚠️ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 13 | ref | ✅ | ✅ | ⚠️ | — |
| `attn_out_post`<br>layer 20 | ref | ✅ | ✅ | ⚠️ | — |
| `attn_out_post`<br>layer 26 | ref | ✅ | ✅ | ⚠️ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 13 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 20 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 26 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | 🐞 | ✅ |
| `mlp_pre`<br>layer 13 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 26 | n/a | — | — | no ref | — |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | n/a | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | 🐞 | ✅ |
| `mlp_act`<br>layer 13 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 26 | n/a | — | — | no ref | — |
| `router_logits`<br>layer 13 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 13 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 20 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 26 | ref | n/a | n/a | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v3 | `attn_out` | 13 | ⚠️ | 0.954385 | 0.3083 | 0.3125 | direction differs (cos 0.954385 below the `tlens` gate of 0.99) |
| tlens_v3 | `attn_out` | 20 | ⚠️ | 0.881086 | 0.5574 | 2.7734 | direction differs (cos 0.881086 below the `tlens` gate of 0.99); same direction, different scale (rel 0.5574 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out` | 26 | ⚠️ | 0.947986 | 0.4265 | 3.5469 | direction differs (cos 0.947986 below the `tlens` gate of 0.99) |
| tlens_v3 | `attn_out_post` | 13 | ⚠️ | 0.954385 | 0.3083 | 0.3125 | direction differs (cos 0.954385 below the `tlens` gate of 0.99) |
| tlens_v3 | `attn_out_post` | 20 | ⚠️ | 0.881086 | 0.5574 | 2.7734 | direction differs (cos 0.881086 below the `tlens` gate of 0.99); same direction, different scale (rel 0.5574 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 26 | ⚠️ | 0.947986 | 0.4265 | 3.5469 | direction differs (cos 0.947986 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_act` | 0 | ❌ | — | — | — | shape `[13, 2048]` against the reference's `[13, 10944]` |
| tlens_v3 | `mlp_out` | 0 | ⚠️ | 0.986824 | 0.1618 | 0.1268 | direction differs (cos 0.986824 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_out` | 13 | ⚠️ | 0.961735 | 0.2743 | 0.3281 | direction differs (cos 0.961735 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_out` | 20 | ⚠️ | 0.974218 | 0.2354 | 1.8594 | direction differs (cos 0.974218 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_out_post` | 0 | ⚠️ | 0.986824 | 0.1618 | 0.1268 | direction differs (cos 0.986824 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_out_post` | 13 | ⚠️ | 0.961735 | 0.2743 | 0.3281 | direction differs (cos 0.961735 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_out_post` | 20 | ⚠️ | 0.974218 | 0.2354 | 1.8594 | direction differs (cos 0.974218 below the `tlens` gate of 0.99) |
| tlens_v3 | `mlp_pre` | 0 | ❌ | — | — | — | shape `[13, 2048]` against the reference's `[13, 10944]` |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `mlp_out` | 13 | 0.996584 | 0.935247 | 10 | 0.3551 |
| interp-engine vllm | `mlp_out_post` | 13 | 0.996584 | 0.935247 | 10 | 0.3551 |
| tlens_v3 | `attn_out` | 0 | 0.990362 | 0.978359 | 4 | 0.2156 |
| tlens_v3 | `attn_out_post` | 0 | 0.990362 | 0.978359 | 4 | 0.2156 |
| tlens_v3 | `mlp_out` | 26 | 0.997684 | 0.976217 | 3 | 0.3093 |
| tlens_v3 | `mlp_out_post` | 26 | 0.997684 | 0.976217 | 3 | 0.3093 |
| tlens_v3 | `resid_mid` | 13 | 0.999979 | 0.979372 | 10 | 0.2087 |
| tlens_v3 | `resid_mid` | 20 | 0.99992 | 0.982561 | 10 | 0.1863 |
| tlens_v3 | `resid_post` | 13 | 0.999979 | 0.981107 | 10 | 0.1947 |
| tlens_v3 | `resid_post` | 20 | 0.999916 | 0.985721 | 10 | 0.1701 |
| tlens_v3 | `resid_post` | 26 | 0.996759 | 0.988258 | 3 | 0.1837 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 13, 20, 26 | nnsight | neither engine captured it |
| `mlp_pre` | 13, 20, 26 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_pre_linear` | 0 | tlens_v3 | this engine declined the point |
| `mlp_act` | 13, 20, 26 | nnsight, interp-engine vllm | neither engine captured it |
| `mlp_act` | 13, 20, 26 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `attn_scores` | 0, 13, 20, 26 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by recomputing from captured q/k, and on MLA there is nothing to recompute from |
