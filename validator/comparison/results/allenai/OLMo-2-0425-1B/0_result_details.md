# `allenai/OLMo-2-0425-1B` — cross-engine results

Every engine's capture of `allenai/OLMo-2-0425-1B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 8, 12, 15.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.28.0 | 62 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.28.0 | 60 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | float32 | v3.8.1 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | float32 | v3.8.1 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v2<br>[v3.8.1](tlens_v2.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 8 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 15 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 8 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 12 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 15 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 8 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 12 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 15 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 8 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 15 | ref | ✅ | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 8 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 15 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.
