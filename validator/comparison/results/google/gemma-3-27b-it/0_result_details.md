# `google/gemma-3-27b-it` — cross-engine results

Every engine's capture of `google/gemma-3-27b-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 31, 46, 61.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 54 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 52 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ⚠️ | ok | bfloat16 | v3.7.0 | 33 | 3 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
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
| `mlp_out_post`<br>layer 46 | ref | ✅ | ✅ | ⚠️ | ✅ | — |
| `mlp_out_post`<br>layer 61 | ref | ✅ | ✅ | ⚠️ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 46 | ref | ✅ | ✅ | ⚠️ | ✅ | — |
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
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 31 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 46 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 61 | ref | ✅ | ✅ | — | — | — |

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out_post` | 46 | ⚠️ | 0.98988 | 0.1559 | 288 | direction differs (cos 0.98988 below the `tlens` gate of 0.99) |
| tlens_v2 | `mlp_out_post` | 46 | ⚠️ | 0.987707 | 0.1695 | 456 | direction differs (cos 0.987707 below the `tlens` gate of 0.99) |
| tlens_v2 | `mlp_out_post` | 61 | ⚠️ | 0.985543 | 0.1748 | 4480 | direction differs (cos 0.985543 below the `tlens` gate of 0.99) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out` | 46 | 0.995526 | 0.977007 | 11 | 0.2192 |
| tlens_v2 | `attn_out` | 61 | 0.993568 | 0.985557 | 4 | 0.1882 |
| tlens_v2 | `attn_out_post` | 31 | 0.996847 | 0.958777 | 9 | 0.2842 |
| tlens_v2 | `attn_out_post` | 61 | 0.996166 | 0.971671 | 2 | 0.257 |
| tlens_v2 | `mlp_act` | 46 | 0.99564 | 0.970219 | 8 | 0.284 |
| tlens_v2 | `mlp_act` | 61 | 0.992644 | 0.978536 | 11 | 0.211 |
| tlens_v2 | `mlp_out_post` | 31 | 0.999872 | 0.987631 | 4 | 0.1569 |
| tlens_v2 | `mlp_pre_linear` | 46 | 0.993582 | 0.984376 | 8 | 0.2028 |
| tlens_v2 | `mlp_pre_linear` | 61 | 0.994886 | 0.989125 | 4 | 0.1521 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
