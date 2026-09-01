# `google/gemma-4-E2B` — cross-engine results

Every engine's capture of `google/gemma-4-E2B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 17, 26, 34.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref* | ok | bfloat16 | v1.3.6 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 48 | 0 | 0 | 6 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 46 | 0 | 0 | 6 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-E2B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT', 'a… | bfloat16 | v3.8.0 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.3.6](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 17 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 34 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 17 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 34 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 17 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 34 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 17 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 34 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 17 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 34 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 17 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 34 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 17 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 26 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 34 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 17 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 26 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 34 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 17 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 26 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 34 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 17 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 34 | ref | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 17 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 34 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 17 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 34 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 17 | n/a | — | no ref | — | — |
| `k_norm_in`<br>layer 26 | n/a | — | no ref | — | — |
| `k_norm_in`<br>layer 34 | n/a | — | no ref | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 17 | n/a | — | no ref | — | — |
| `k_norm_out`<br>layer 26 | n/a | — | no ref | — | — |
| `k_norm_out`<br>layer 34 | n/a | — | no ref | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 17 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 34 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `k_norm_in` | 17, 26, 34 | interp-engine vllm | neither engine captured it |
| `k_norm_in` | 17, 26, 34 | interp-engine vllm-static | the `eager` reference declined the point, so there is nothing to score against |
| `k_norm_out` | 17, 26, 34 | interp-engine vllm | neither engine captured it |
| `k_norm_out` | 17, 26, 34 | interp-engine vllm-static | the `eager` reference declined the point, so there is nothing to score against |
