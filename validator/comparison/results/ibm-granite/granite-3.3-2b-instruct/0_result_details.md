# `ibm-granite/granite-3.3-2b-instruct` — cross-engine results

Every engine's capture of `ibm-granite/granite-3.3-2b-instruct`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 20, 30, 39.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 46 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 44 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | [unsupported](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | skip: ValueError: ibm-granite/granite-3.3-2b-instruct not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | bfloat16 | v3.8.1 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 30 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 30 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 20 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 30 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 39 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 20 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 30 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 39 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 20 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 30 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 39 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 30 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 30 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 30 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 39 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.
