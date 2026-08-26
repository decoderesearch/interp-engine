# `google/gemma-4-26B-A4B-it` — cross-engine results

Every engine's capture of `google/gemma-4-26B-A4B-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 15, 22, 29.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.3.5 | — | — | — | — |
| [interp-engine vllm](vllm.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | ok | bfloat16 | v0.27.1 | 29 | 25 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | ok | bfloat16 | v0.27.1 | 28 | 24 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-26B-A4B-it not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mG… | bfloat16 | v3.8.0 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1647) | ok | bfloat16 | v3.8.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.3.5](eager.json) | interp-engine vllm<br>[v0.27.1](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.8.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `resid_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `resid_mid`<br>layer 29 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_out`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 15 | ref | 🐞 | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 15 | ref | ✅ | 🐞 | ✅ | ✅ |
| `attn_out`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `attn_out`<br>layer 29 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 15 | ref | 🐞 | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 29 | ref | 🐞 | 🐞 | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 15 | ref | 🐞 | 🐞 | — | ✅ |
| `attn_in`<br>layer 22 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 29 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 15 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 22 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 29 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 15 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 22 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 29 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 22 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_act`<br>layer 29 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 22 | ref | 🐞 | 🐞 | — | — |
| `q_norm_in`<br>layer 29 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 22 | ref | 🐞 | 🐞 | — | — |
| `q_norm_out`<br>layer 29 | ref | 🐞 | 🐞 | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 15 | ref | 🐞 | ✅ | — | — |
| `k_norm_in`<br>layer 22 | ref | 🐞 | 🐞 | — | — |
| `k_norm_in`<br>layer 29 | ref | 🐞 | 🐞 | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 22 | ref | 🐞 | 🐞 | — | — |
| `k_norm_out`<br>layer 29 | ref | 🐞 | 🐞 | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | 🐞 | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 22 | ref | 🐞 | 🐞 | — | — |
| `attn_scores`<br>layer 29 | ref | 🐞 | 🐞 | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 15 | ⚠️ | 0.989601 | 0.1441 | 20.5 | direction differs (cos 0.989601 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 22 | ⚠️ | 0.982139 | 0.1924 | 3.125 | direction differs (cos 0.982139 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 29 | ⚠️ | 0.973237 | 0.2306 | 5.3438 | direction differs (cos 0.973237 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 15 | ⚠️ | 0.982798 | 0.1861 | 1.5781 | direction differs (cos 0.982798 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 22 | ⚠️ | 0.965681 | 0.2616 | 0.9609 | direction differs (cos 0.965681 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 29 | ⚠️ | 0.977097 | 0.2143 | 6.0938 | direction differs (cos 0.977097 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_scores` | 22 | ⚠️ | 0.985915 | 0.174 | 4.2726 | direction differs (cos 0.985915 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_scores` | 29 | ⚠️ | 0.986112 | 0.1701 | 3.3269 | direction differs (cos 0.986112 below the `fused` gate of 0.99) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.976667 | 0.2197 | 16.5 | direction differs (cos 0.976667 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 15 | ⚠️ | 0.989936 | 0.1415 | 17.25 | direction differs (cos 0.989936 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 22 | ⚠️ | 0.976926 | 0.2749 | 30.5 | direction differs (cos 0.976926 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 29 | ⚠️ | 0.98462 | 0.1747 | 1.6133 | direction differs (cos 0.98462 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 22 | ⚠️ | 0.986161 | 0.1664 | 0.4253 | direction differs (cos 0.986161 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 29 | ⚠️ | 0.983349 | 0.1825 | 0.1229 | direction differs (cos 0.983349 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 22 | ⚠️ | 0.977543 | 0.2108 | 6.25 | direction differs (cos 0.977543 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 29 | ⚠️ | 0.986289 | 0.1665 | 6.5312 | direction differs (cos 0.986289 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out` | 22 | ⚠️ | 0.989127 | 0.1471 | 1.3281 | direction differs (cos 0.989127 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 15 | ⚠️ | 0.984192 | 0.1772 | 23.25 | direction differs (cos 0.984192 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 22 | ⚠️ | 0.926104 | 0.4102 | 24.6875 | direction differs (cos 0.926104 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_in` | 22 | ⚠️ | 0.975639 | 0.2194 | 11.0625 | direction differs (cos 0.975639 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 22 | ⚠️ | 0.980112 | 0.1994 | 4.5859 | direction differs (cos 0.980112 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 29 | ⚠️ | 0.98743 | 0.1586 | 2.6094 | direction differs (cos 0.98743 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_mid` | 22 | ⚠️ | 0.965081 | 0.2763 | 35 | direction differs (cos 0.965081 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_mid` | 29 | ⚠️ | 0.97389 | 0.2305 | 6.5625 | direction differs (cos 0.97389 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_post` | 22 | ⚠️ | 0.963104 | 0.2798 | 17.25 | direction differs (cos 0.963104 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 15 | ⚠️ | 0.98942 | 0.1455 | 20 | direction differs (cos 0.98942 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 15 | ⚠️ | 0.989804 | 0.1443 | 4.875 | direction differs (cos 0.989804 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 22 | ⚠️ | 0.980275 | 0.2026 | 3.25 | direction differs (cos 0.980275 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 29 | ⚠️ | 0.975452 | 0.2217 | 3.8438 | direction differs (cos 0.975452 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 15 | ⚠️ | 0.97955 | 0.2042 | 1.5469 | direction differs (cos 0.97955 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 22 | ⚠️ | 0.961904 | 0.2763 | 1.0664 | direction differs (cos 0.961904 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 29 | ⚠️ | 0.978531 | 0.2071 | 4.5312 | direction differs (cos 0.978531 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_scores` | 22 | ⚠️ | 0.983065 | 0.1839 | 4.8101 | direction differs (cos 0.983065 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_scores` | 29 | ⚠️ | 0.957185 | 0.2916 | 5.0702 | direction differs (cos 0.957185 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_in` | 22 | ⚠️ | 0.983869 | 0.2324 | 25 | direction differs (cos 0.983869 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_in` | 29 | ⚠️ | 0.985558 | 0.1695 | 1.2972 | direction differs (cos 0.985558 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 22 | ⚠️ | 0.98416 | 0.178 | 0.4541 | direction differs (cos 0.98416 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 29 | ⚠️ | 0.984555 | 0.1758 | 0.0988 | direction differs (cos 0.984555 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 22 | ⚠️ | 0.979034 | 0.204 | 5.6562 | direction differs (cos 0.979034 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 29 | ⚠️ | 0.987614 | 0.1579 | 3.8438 | direction differs (cos 0.987614 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out` | 22 | ⚠️ | 0.98954 | 0.1447 | 1.3125 | direction differs (cos 0.98954 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 15 | ⚠️ | 0.986204 | 0.1656 | 24.5 | direction differs (cos 0.986204 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 22 | ⚠️ | 0.939036 | 0.3688 | 21.6875 | direction differs (cos 0.939036 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_in` | 22 | ⚠️ | 0.978008 | 0.2086 | 9.5625 | direction differs (cos 0.978008 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 22 | ⚠️ | 0.981949 | 0.19 | 4.3672 | direction differs (cos 0.981949 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 29 | ⚠️ | 0.988694 | 0.1504 | 2.8828 | direction differs (cos 0.988694 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_mid` | 22 | ⚠️ | 0.968172 | 0.2647 | 34 | direction differs (cos 0.968172 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_mid` | 29 | ⚠️ | 0.975344 | 0.2235 | 5 | direction differs (cos 0.975344 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_post` | 22 | ⚠️ | 0.966466 | 0.2669 | 20.25 | direction differs (cos 0.966466 below the `fused` gate of 0.99) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 22 | 0.991733 | 0.826619 | 2 | 0.5816 |
| interp-engine vllm | `attn_in` | 29 | 0.991045 | 0.968591 | 10 | 0.2519 |
| interp-engine vllm | `attn_out` | 15 | 0.9913 | 0.975132 | 10 | 0.2698 |
| interp-engine vllm | `k_norm_out` | 15 | 0.991988 | 0.98222 | 5 | 0.1886 |
| interp-engine vllm | `mlp_out` | 29 | 0.992943 | 0.954896 | 11 | 0.3234 |
| interp-engine vllm | `mlp_out_post` | 29 | 0.995053 | 0.969689 | 11 | 0.3155 |
| interp-engine vllm | `q_norm_in` | 15 | 0.992809 | 0.988478 | 5 | 0.1516 |
| interp-engine vllm | `q_norm_in` | 29 | 0.990196 | 0.94499 | 10 | 0.3345 |
| interp-engine vllm | `q_norm_out` | 15 | 0.99388 | 0.987507 | 5 | 0.1581 |
| interp-engine vllm | `resid_mid` | 15 | 0.99467 | 0.986848 | 10 | 0.162 |
| interp-engine vllm | `resid_post` | 15 | 0.992724 | 0.95382 | 2 | 0.3004 |
| interp-engine vllm | `resid_post` | 29 | 0.994263 | 0.965403 | 11 | 0.3363 |
| interp-engine vllm-static | `attn_in` | 22 | 0.994075 | 0.792487 | 2 | 0.6382 |
| interp-engine vllm-static | `attn_in` | 29 | 0.991919 | 0.980298 | 10 | 0.2004 |
| interp-engine vllm-static | `k_norm_in` | 15 | 0.991365 | 0.982001 | 5 | 0.1926 |
| interp-engine vllm-static | `k_norm_out` | 15 | 0.990894 | 0.979938 | 5 | 0.2003 |
| interp-engine vllm-static | `mlp_out` | 29 | 0.992675 | 0.967544 | 11 | 0.2755 |
| interp-engine vllm-static | `mlp_out_post` | 29 | 0.995359 | 0.983557 | 11 | 0.2596 |
| interp-engine vllm-static | `q_norm_in` | 15 | 0.992633 | 0.98733 | 5 | 0.1587 |
| interp-engine vllm-static | `q_norm_in` | 29 | 0.99129 | 0.955044 | 10 | 0.3021 |
| interp-engine vllm-static | `q_norm_out` | 15 | 0.992802 | 0.986169 | 5 | 0.1663 |
| interp-engine vllm-static | `resid_mid` | 15 | 0.993513 | 0.985518 | 2 | 0.1719 |
| interp-engine vllm-static | `resid_post` | 15 | 0.993427 | 0.946989 | 2 | 0.3216 |
| interp-engine vllm-static | `resid_post` | 29 | 0.994492 | 0.980406 | 11 | 0.2766 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
