# `openai-community/gpt2` — cross-engine results

Every engine's capture of `openai-community/gpt2`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 6, 11.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.28.0 | 35 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.28.0 | 33 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | float32 | v3.8.1 | 24 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | float32 | v3.8.1 | 24 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 21 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v2<br>[v3.8.1](tlens_v2.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 6 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 11 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 6 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 11 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 6 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 11 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 6 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 11 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 6 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 11 | ref | ✅ | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 6 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 11 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.
