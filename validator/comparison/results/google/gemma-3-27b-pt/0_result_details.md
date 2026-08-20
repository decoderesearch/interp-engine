# `google/gemma-3-27b-pt` — cross-engine results

Every engine's capture of `google/gemma-3-27b-pt`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 31, 46, 61.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.0.1 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 54 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.0.1](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 46 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 61 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 31 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 46 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 61 | ref | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 31 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 46 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 61 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 31 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 46 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 61 | ref | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 46 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 61 | ref | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 31 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 46 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 61 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 31 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 46 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 61 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 31 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 46 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 61 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 31 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 46 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 61 | ref | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 31 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 46 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 61 | ref | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out` | 61 | 0.992898 | 0.946776 | 6 | 0.3483 |
| tlens_v2 | `attn_out_post` | 31 | 0.996817 | 0.988914 | 10 | 0.1656 |
| tlens_v2 | `attn_out_post` | 61 | 0.996508 | 0.963491 | 11 | 0.2765 |
| tlens_v2 | `mlp_act` | 31 | 0.999005 | 0.989936 | 10 | 0.149 |
| tlens_v2 | `mlp_act` | 46 | 0.994611 | 0.98478 | 6 | 0.2128 |
| tlens_v2 | `mlp_act` | 61 | 0.994353 | 0.950509 | 6 | 0.3112 |
| tlens_v2 | `mlp_out` | 61 | 0.997829 | 0.986967 | 6 | 0.161 |
| tlens_v2 | `mlp_out_post` | 31 | 0.999838 | 0.988116 | 10 | 0.1577 |
| tlens_v2 | `mlp_out_post` | 46 | 0.993689 | 0.976365 | 6 | 0.2434 |
| tlens_v2 | `mlp_out_post` | 61 | 0.998314 | 0.986582 | 6 | 0.1838 |
| tlens_v2 | `mlp_pre_linear` | 46 | 0.995852 | 0.98751 | 6 | 0.1814 |
| tlens_v2 | `mlp_pre_linear` | 61 | 0.995488 | 0.964019 | 6 | 0.266 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
