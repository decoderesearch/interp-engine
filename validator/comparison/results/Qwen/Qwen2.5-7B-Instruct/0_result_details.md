# `Qwen/Qwen2.5-7B-Instruct` — cross-engine results

Every engine's capture of `Qwen/Qwen2.5-7B-Instruct`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 14, 21, 27.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 38 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 36 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 14 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 14 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 14 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 14 | ref | ✅† | ✅† | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 14 | ref | ✅† | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅† | ✅† | ✅ | ✅ | — |
| `attn_out_post`<br>layer 14 | ref | ✅† | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 14 | ref | ✅† | ✅† | — | — | ✅ |
| `attn_in`<br>layer 21 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 27 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 14 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 21 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 27 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 14 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 21 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 27 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 14 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 21 | ref | ✅† | ✅† | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 27 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 14 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 21 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 27 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 14 | 0.986431 | 0.920334 | 12 | 0.4027 |
| interp-engine vllm | `attn_in` | 21 | 0.99133 | 0.946098 | 12 | 0.3264 |
| interp-engine vllm | `attn_in` | 27 | 0.995851 | 0.951856 | 12 | 0.3091 |
| interp-engine vllm | `attn_out` | 0 | 0.989601 | 0.941683 | 4 | 0.385 |
| interp-engine vllm | `attn_out` | 14 | 0.989911 | 0.890585 | 12 | 0.4734 |
| interp-engine vllm | `attn_out` | 21 | 0.992531 | 0.948508 | 12 | 0.3185 |
| interp-engine vllm | `attn_out` | 27 | 0.999435 | 0.927953 | 6 | 0.7318 |
| interp-engine vllm | `attn_out_post` | 0 | 0.989601 | 0.941683 | 4 | 0.385 |
| interp-engine vllm | `attn_out_post` | 14 | 0.989911 | 0.890585 | 12 | 0.4734 |
| interp-engine vllm | `attn_out_post` | 21 | 0.992531 | 0.948508 | 12 | 0.3185 |
| interp-engine vllm | `attn_out_post` | 27 | 0.999435 | 0.927953 | 6 | 0.7318 |
| interp-engine vllm | `final_norm` | — | 0.995396 | 0.982281 | 12 | 0.19 |
| interp-engine vllm | `mlp_act` | 0 | 0.987626 | 0.655246 | 12 | 0.9327 |
| interp-engine vllm | `mlp_act` | 14 | 0.98238 | 0.85808 | 12 | 0.5482 |
| interp-engine vllm | `mlp_act` | 21 | 0.989748 | 0.949831 | 12 | 0.3226 |
| interp-engine vllm | `mlp_act` | 27 | 0.997275 | 0.977657 | 12 | 0.243 |
| interp-engine vllm | `mlp_out` | 0 | 0.990338 | 0.710177 | 12 | 1.0276 |
| interp-engine vllm | `mlp_out` | 14 | 0.986737 | 0.856019 | 12 | 0.5559 |
| interp-engine vllm | `mlp_out` | 21 | 0.990744 | 0.952014 | 12 | 0.3138 |
| interp-engine vllm | `mlp_out` | 27 | 0.998137 | 0.989195 | 12 | 0.1785 |
| interp-engine vllm | `mlp_out_post` | 0 | 0.990338 | 0.710177 | 12 | 1.0276 |
| interp-engine vllm | `mlp_out_post` | 14 | 0.986737 | 0.856019 | 12 | 0.5559 |
| interp-engine vllm | `mlp_out_post` | 21 | 0.990744 | 0.952014 | 12 | 0.3138 |
| interp-engine vllm | `mlp_out_post` | 27 | 0.998137 | 0.989195 | 12 | 0.1785 |
| interp-engine vllm | `resid_mid` | 0 | 0.989862 | 0.944164 | 4 | 0.3791 |
| interp-engine vllm | `resid_mid` | 14 | 0.999996 | 0.942935 | 12 | 0.3849 |
| interp-engine vllm | `resid_mid` | 21 | 0.999993 | 0.967338 | 12 | 0.2713 |
| interp-engine vllm | `resid_mid` | 27 | 0.99566 | 0.986684 | 6 | 0.23 |
| interp-engine vllm | `resid_post` | 0 | 0.989831 | 0.879206 | 12 | 0.6521 |
| interp-engine vllm | `resid_post` | 14 | 0.999996 | 0.938135 | 12 | 0.3835 |
| interp-engine vllm | `resid_post` | 21 | 0.999992 | 0.970176 | 12 | 0.2615 |
| interp-engine vllm | `resid_post` | 27 | 0.995728 | 0.929876 | 6 | 0.3695 |
| interp-engine vllm-static | `attn_in` | 14 | 0.986354 | 0.919977 | 12 | 0.4031 |
| interp-engine vllm-static | `attn_in` | 21 | 0.991208 | 0.945824 | 12 | 0.3269 |
| interp-engine vllm-static | `attn_in` | 27 | 0.995805 | 0.951631 | 12 | 0.3101 |
| interp-engine vllm-static | `attn_out` | 0 | 0.989602 | 0.941687 | 4 | 0.385 |
| interp-engine vllm-static | `attn_out` | 14 | 0.990004 | 0.893774 | 12 | 0.4644 |
| interp-engine vllm-static | `attn_out` | 21 | 0.992416 | 0.948327 | 12 | 0.3194 |
| interp-engine vllm-static | `attn_out` | 27 | 0.999398 | 0.926469 | 6 | 0.741 |
| interp-engine vllm-static | `attn_out_post` | 0 | 0.989602 | 0.941687 | 4 | 0.385 |
| interp-engine vllm-static | `attn_out_post` | 14 | 0.990004 | 0.893774 | 12 | 0.4644 |
| interp-engine vllm-static | `attn_out_post` | 21 | 0.992416 | 0.948327 | 12 | 0.3194 |
| interp-engine vllm-static | `attn_out_post` | 27 | 0.999398 | 0.926469 | 6 | 0.741 |
| interp-engine vllm-static | `mlp_act` | 0 | 0.987625 | 0.655254 | 12 | 0.9327 |
| interp-engine vllm-static | `mlp_act` | 14 | 0.982224 | 0.857613 | 12 | 0.5483 |
| interp-engine vllm-static | `mlp_act` | 21 | 0.989637 | 0.949778 | 12 | 0.3226 |
| interp-engine vllm-static | `mlp_act` | 27 | 0.99724 | 0.977627 | 12 | 0.243 |
| interp-engine vllm-static | `mlp_out` | 0 | 0.990339 | 0.710179 | 12 | 1.0276 |
| interp-engine vllm-static | `mlp_out` | 14 | 0.986624 | 0.855886 | 12 | 0.555 |
| interp-engine vllm-static | `mlp_out` | 21 | 0.99063 | 0.951909 | 12 | 0.314 |
| interp-engine vllm-static | `mlp_out` | 27 | 0.998135 | 0.98893 | 12 | 0.1779 |
| interp-engine vllm-static | `mlp_out_post` | 0 | 0.990339 | 0.710179 | 12 | 1.0276 |
| interp-engine vllm-static | `mlp_out_post` | 14 | 0.986624 | 0.855886 | 12 | 0.555 |
| interp-engine vllm-static | `mlp_out_post` | 21 | 0.99063 | 0.951909 | 12 | 0.314 |
| interp-engine vllm-static | `mlp_out_post` | 27 | 0.998135 | 0.98893 | 12 | 0.1779 |
| interp-engine vllm-static | `resid_mid` | 0 | 0.989863 | 0.944167 | 4 | 0.3791 |
| interp-engine vllm-static | `resid_mid` | 14 | 0.999992 | 0.942787 | 12 | 0.385 |
| interp-engine vllm-static | `resid_mid` | 21 | 0.999989 | 0.967173 | 12 | 0.2726 |
| interp-engine vllm-static | `resid_mid` | 27 | 0.995113 | 0.986479 | 6 | 0.2336 |
| interp-engine vllm-static | `resid_post` | 0 | 0.989832 | 0.879206 | 12 | 0.6521 |
| interp-engine vllm-static | `resid_post` | 14 | 0.999992 | 0.937919 | 12 | 0.384 |
| interp-engine vllm-static | `resid_post` | 21 | 0.999988 | 0.970065 | 12 | 0.2622 |
| interp-engine vllm-static | `resid_post` | 27 | 0.995543 | 0.928575 | 6 | 0.3724 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Waived passes

| engine | point | layers | worst cos | worst rel diff |
| --- | --- | --- | --- | --- |
| interp-engine vllm | `resid_post` | 0 | 0.989831 | 0.145 |
| interp-engine vllm | `resid_mid` | 0 | 0.989862 | 0.144 |
| interp-engine vllm | `mlp_out` | 14 | 0.986737 | 0.1632 |
| interp-engine vllm | `mlp_out_post` | 14 | 0.986737 | 0.1632 |
| interp-engine vllm | `attn_out` | 0, 14 | 0.989601 | 0.1458 |
| interp-engine vllm | `attn_out_post` | 0, 14 | 0.989601 | 0.1458 |
| interp-engine vllm | `attn_in` | 14 | 0.986431 | 0.1655 |
| interp-engine vllm | `mlp_act` | 0, 14, 21 | 0.98238 | 0.1885 |
| interp-engine vllm-static | `resid_post` | 0 | 0.989832 | 0.1449 |
| interp-engine vllm-static | `resid_mid` | 0 | 0.989863 | 0.144 |
| interp-engine vllm-static | `mlp_out` | 14 | 0.986624 | 0.1638 |
| interp-engine vllm-static | `mlp_out_post` | 14 | 0.986624 | 0.1638 |
| interp-engine vllm-static | `attn_out` | 0 | 0.989602 | 0.1458 |
| interp-engine vllm-static | `attn_out_post` | 0 | 0.989602 | 0.1458 |
| interp-engine vllm-static | `attn_in` | 14 | 0.986354 | 0.166 |
| interp-engine vllm-static | `mlp_act` | 0, 14, 21 | 0.982224 | 0.1893 |

The waiver: Qwen2.5 massive activations in bf16: one residual coordinate dominates the norm, so RMSNorm propagates its rounding everywhere downstream. IE_FORCE_DTYPE=float32 collapses this to cos 0.999999, so it is checkpoint arithmetic, not capture
