# `google/gemma-2-2b` — cross-engine results

Every engine's capture of `google/gemma-2-2b`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 13, 19, 25.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.0.1 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 38 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | float32 | v3.7.0 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | float32 | v3.7.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.0.1](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 19 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 25 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 19 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 25 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 19 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 25 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 19 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 25 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 19 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 25 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 19 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 25 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 13 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 19 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 25 | ref | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 13 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 19 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 25 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 13 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 19 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 25 | ref | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 19 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 25 | ref | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 13 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 19 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 25 | ref | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `final_norm` | — | 0.998966 | 0.961009 | 10 | 0.2842 |
| interp-engine vllm | `mlp_act` | 25 | 0.99912 | 0.968381 | 10 | 0.2551 |
| interp-engine vllm | `mlp_out` | 25 | 0.999446 | 0.960698 | 10 | 0.2794 |
| interp-engine vllm | `mlp_out_post` | 25 | 0.997493 | 0.989014 | 10 | 0.162 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
