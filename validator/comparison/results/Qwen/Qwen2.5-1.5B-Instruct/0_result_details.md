# `Qwen/Qwen2.5-1.5B-Instruct` — cross-engine results

Every engine's capture of `Qwen/Qwen2.5-1.5B-Instruct`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 14, 21, 27.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.0.1 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 38 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.0.1](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅† | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 14 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 14 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅† | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 14 | ref | ✅† | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅† | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 14 | ref | ✅† | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 14 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 14 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 14 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 21 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 27 | ref | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 14 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 21 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 27 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 14 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 21 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 27 | ref | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅† | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 14 | ref | ✅† | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 14 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 21 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 27 | ref | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 14 | 0.99226 | 0.964687 | 10 | 0.2667 |
| interp-engine vllm | `attn_in` | 21 | 0.995811 | 0.978727 | 10 | 0.2161 |
| interp-engine vllm | `attn_in` | 27 | 0.997684 | 0.980321 | 10 | 0.2048 |
| interp-engine vllm | `attn_out` | 0 | 0.991103 | 0.947026 | 10 | 0.3421 |
| interp-engine vllm | `attn_out` | 14 | 0.992987 | 0.983199 | 10 | 0.1951 |
| interp-engine vllm | `attn_out_post` | 0 | 0.991103 | 0.947026 | 10 | 0.3421 |
| interp-engine vllm | `attn_out_post` | 14 | 0.992987 | 0.983199 | 10 | 0.1951 |
| interp-engine vllm | `mlp_act` | 0 | 0.967734 | 0.843451 | 10 | 0.7146 |
| interp-engine vllm | `mlp_act` | 14 | 0.981263 | 0.942981 | 10 | 0.3416 |
| interp-engine vllm | `mlp_act` | 21 | 0.991793 | 0.96468 | 10 | 0.2905 |
| interp-engine vllm | `mlp_out` | 0 | 0.973697 | 0.859051 | 10 | 0.6909 |
| interp-engine vllm | `mlp_out` | 14 | 0.982179 | 0.934868 | 10 | 0.3587 |
| interp-engine vllm | `mlp_out` | 21 | 0.993831 | 0.971646 | 10 | 0.2622 |
| interp-engine vllm | `mlp_out_post` | 0 | 0.973697 | 0.859051 | 10 | 0.6909 |
| interp-engine vllm | `mlp_out_post` | 14 | 0.982179 | 0.934868 | 10 | 0.3587 |
| interp-engine vllm | `mlp_out_post` | 21 | 0.993831 | 0.971646 | 10 | 0.2622 |
| interp-engine vllm | `resid_mid` | 0 | 0.991113 | 0.947204 | 10 | 0.3417 |
| interp-engine vllm | `resid_mid` | 14 | 0.999997 | 0.971947 | 10 | 0.2495 |
| interp-engine vllm | `resid_mid` | 21 | 0.999996 | 0.980584 | 10 | 0.2027 |
| interp-engine vllm | `resid_post` | 0 | 0.984681 | 0.888635 | 10 | 0.6008 |
| interp-engine vllm | `resid_post` | 14 | 0.999997 | 0.971598 | 10 | 0.2417 |
| interp-engine vllm | `resid_post` | 21 | 0.999996 | 0.984327 | 10 | 0.1835 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Waived passes

| engine | point | layers | worst cos | worst rel diff |
| --- | --- | --- | --- | --- |
| interp-engine vllm | `resid_post` | 0 | 0.984681 | 0.1809 |
| interp-engine vllm | `mlp_out` | 0, 14 | 0.973697 | 0.2382 |
| interp-engine vllm | `mlp_out_post` | 0, 14 | 0.973697 | 0.2382 |
| interp-engine vllm | `mlp_act` | 0, 14 | 0.967734 | 0.265 |

The waiver: Qwen2.5 massive activations in bf16: one residual coordinate dominates the norm, so RMSNorm propagates its rounding everywhere downstream. IE_FORCE_DTYPE=float32 collapses this to cos 0.999999, so it is checkpoint arithmetic, not capture
