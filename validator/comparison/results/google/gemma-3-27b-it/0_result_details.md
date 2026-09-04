# `google/gemma-3-27b-it` — cross-engine results

Every engine's capture of `google/gemma-3-27b-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 31, 46, 61.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 62 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 60 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.8.1 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v2<br>[v3.8.1](tlens_v2.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 31 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 46 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 61 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 31 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 46 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 61 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 31 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 46 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 61 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 61 | ref | ✅ | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 61 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out` | 46 | 0.995526 | 0.977007 | 11 | 0.2192 |
| tlens_v2 | `attn_out_post` | 31 | 0.996847 | 0.958777 | 9 | 0.2842 |
| tlens_v2 | `attn_out_post` | 46 | 0.98988 | 0.968402 | 11 | 0.2906 |
| tlens_v2 | `attn_out_post` | 61 | 0.996166 | 0.971671 | 2 | 0.257 |
| tlens_v2 | `mlp_act` | 46 | 0.99564 | 0.970219 | 8 | 0.284 |
| tlens_v2 | `mlp_act` | 61 | 0.992644 | 0.978536 | 11 | 0.211 |
| tlens_v2 | `mlp_out_post` | 46 | 0.987707 | 0.969603 | 8 | 0.2837 |
| tlens_v2 | `mlp_out_post` | 61 | 0.985543 | 0.751561 | 11 | 0.6817 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
