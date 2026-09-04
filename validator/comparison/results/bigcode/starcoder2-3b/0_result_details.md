# `bigcode/starcoder2-3b` — cross-engine results

Every engine's capture of `bigcode/starcoder2-3b`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 15, 22, 29.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.28.0 | 46 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.28.0 | 44 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: bigcode/starcoder2-3b not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | float32 | v3.8.1 | 0 | 0 | 0 | 32 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | float32 | v3.8.1 | 32 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 28 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 22 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 22 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 22 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 22 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 22 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 22 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 15 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 22 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 29 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 15 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 22 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 29 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 22 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 22 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 29 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 22 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 29 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 22 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 29 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.
