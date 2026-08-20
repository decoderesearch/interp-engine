# `bigcode/gpt_bigcode-santacoder` — cross-engine results

Every engine's capture of `bigcode/gpt_bigcode-santacoder`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.26.0 | 38 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.27.1 | 36 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: bigcode/gpt_bigcode-santacoder not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forev… | float32 | v3.7.0 | 0 | 0 | 0 | 32 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | float32 | v3.7.0 | 32 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 28 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 23 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 12 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 18 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 23 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 23 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.
