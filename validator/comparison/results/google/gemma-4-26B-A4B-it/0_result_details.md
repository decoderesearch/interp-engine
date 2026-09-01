# `google/gemma-4-26B-A4B-it` — cross-engine results

Every engine's capture of `google/gemma-4-26B-A4B-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 15, 22, 29.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.3.6 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ⚠️ | ok | bfloat16 | v0.28.0 | 29 | 25 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ⚠️ | ok | bfloat16 | v0.28.0 | 28 | 24 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-26B-A4B-it not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mG… | bfloat16 | v3.8.0 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.3.6](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `resid_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `resid_mid`<br>layer 29 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `mlp_out`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 15 | ref | ⚠️ | ⚠️ | ✅ | — |
| `mlp_out_post`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | — |
| `mlp_out_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 15 | ref | ✅ | ⚠️ | ✅ | ✅ |
| `attn_out`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `attn_out`<br>layer 29 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 15 | ref | ⚠️ | ⚠️ | ✅ | — |
| `attn_out_post`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | — |
| `attn_out_post`<br>layer 29 | ref | ⚠️ | ⚠️ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 15 | ref | ⚠️ | ⚠️ | — | ✅ |
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
| `mlp_act`<br>layer 22 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `mlp_act`<br>layer 29 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 22 | ref | ⚠️ | ⚠️ | — | — |
| `q_norm_in`<br>layer 29 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 22 | ref | ⚠️ | ⚠️ | — | — |
| `q_norm_out`<br>layer 29 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 15 | ref | ⚠️ | ✅ | — | — |
| `k_norm_in`<br>layer 22 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_in`<br>layer 29 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 22 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_out`<br>layer 29 | ref | ⚠️ | ⚠️ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ⚠️ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 22 | ref | ⚠️ | ⚠️ | — | — |
| `attn_scores`<br>layer 29 | ref | ⚠️ | ⚠️ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 15 | ⚠️ | 0.987834 | 0.1562 | 27.5 | direction differs (cos 0.987834 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 22 | ⚠️ | 0.981336 | 0.1953 | 2.75 | direction differs (cos 0.981336 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 29 | ⚠️ | 0.974931 | 0.2233 | 5.1875 | direction differs (cos 0.974931 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 15 | ⚠️ | 0.982827 | 0.1876 | 1.0469 | direction differs (cos 0.982827 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 22 | ⚠️ | 0.96469 | 0.2669 | 0.959 | direction differs (cos 0.96469 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 29 | ⚠️ | 0.978844 | 0.2059 | 5.7812 | direction differs (cos 0.978844 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_scores` | 22 | ⚠️ | 0.986837 | 0.1637 | 3.9311 | direction differs (cos 0.986837 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_scores` | 29 | ⚠️ | 0.987481 | 0.1611 | 3.2109 | direction differs (cos 0.987481 below the `fused` gate of 0.99) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.980459 | 0.2006 | 13.5 | direction differs (cos 0.980459 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 15 | ⚠️ | 0.987619 | 0.1572 | 21 | direction differs (cos 0.987619 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 22 | ⚠️ | 0.977421 | 0.2746 | 30.5 | direction differs (cos 0.977421 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 29 | ⚠️ | 0.985146 | 0.1718 | 1.6953 | direction differs (cos 0.985146 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 22 | ⚠️ | 0.985951 | 0.1676 | 0.4248 | direction differs (cos 0.985951 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 29 | ⚠️ | 0.983972 | 0.179 | 0.129 | direction differs (cos 0.983972 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 22 | ⚠️ | 0.979288 | 0.2032 | 5.8125 | direction differs (cos 0.979288 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 29 | ⚠️ | 0.987309 | 0.1591 | 6.0938 | direction differs (cos 0.987309 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out` | 22 | ⚠️ | 0.989324 | 0.1465 | 1.2188 | direction differs (cos 0.989324 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 15 | ⚠️ | 0.97963 | 0.2011 | 24 | direction differs (cos 0.97963 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 22 | ⚠️ | 0.933253 | 0.3874 | 23.1875 | direction differs (cos 0.933253 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_in` | 22 | ⚠️ | 0.977518 | 0.2109 | 10.5 | direction differs (cos 0.977518 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 22 | ⚠️ | 0.981383 | 0.193 | 4.3359 | direction differs (cos 0.981383 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 29 | ⚠️ | 0.987862 | 0.1558 | 2.6562 | direction differs (cos 0.987862 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_mid` | 22 | ⚠️ | 0.968509 | 0.2636 | 28.5 | direction differs (cos 0.968509 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_mid` | 29 | ⚠️ | 0.975581 | 0.2225 | 6.375 | direction differs (cos 0.975581 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_post` | 22 | ⚠️ | 0.967008 | 0.2648 | 14 | direction differs (cos 0.967008 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 15 | ⚠️ | 0.989317 | 0.1458 | 18 | direction differs (cos 0.989317 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 15 | ⚠️ | 0.989336 | 0.1493 | 5 | direction differs (cos 0.989336 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 22 | ⚠️ | 0.982186 | 0.1923 | 3 | direction differs (cos 0.982186 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 29 | ⚠️ | 0.975022 | 0.2234 | 4.0312 | direction differs (cos 0.975022 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 15 | ⚠️ | 0.979012 | 0.2069 | 1.2188 | direction differs (cos 0.979012 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 22 | ⚠️ | 0.965579 | 0.2625 | 0.9336 | direction differs (cos 0.965579 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 29 | ⚠️ | 0.978496 | 0.2073 | 4.9062 | direction differs (cos 0.978496 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_scores` | 22 | ⚠️ | 0.980647 | 0.1973 | 5.9315 | direction differs (cos 0.980647 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_scores` | 29 | ⚠️ | 0.989421 | 0.1479 | 1.8156 | direction differs (cos 0.989421 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_in` | 22 | ⚠️ | 0.98048 | 0.2594 | 29 | direction differs (cos 0.98048 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_in` | 29 | ⚠️ | 0.985064 | 0.1722 | 1.6094 | direction differs (cos 0.985064 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 22 | ⚠️ | 0.986525 | 0.1642 | 0.3975 | direction differs (cos 0.986525 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 29 | ⚠️ | 0.983923 | 0.1793 | 0.1224 | direction differs (cos 0.983923 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 22 | ⚠️ | 0.978023 | 0.2093 | 6.375 | direction differs (cos 0.978023 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 29 | ⚠️ | 0.987407 | 0.1588 | 5.0312 | direction differs (cos 0.987407 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out` | 22 | ⚠️ | 0.989308 | 0.147 | 1.3086 | direction differs (cos 0.989308 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 15 | ⚠️ | 0.987005 | 0.1607 | 21 | direction differs (cos 0.987005 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 22 | ⚠️ | 0.929612 | 0.3993 | 24.6875 | direction differs (cos 0.929612 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_in` | 22 | ⚠️ | 0.976455 | 0.2159 | 10.5938 | direction differs (cos 0.976455 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 22 | ⚠️ | 0.981052 | 0.1947 | 4.9336 | direction differs (cos 0.981052 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 29 | ⚠️ | 0.987846 | 0.1559 | 2.7266 | direction differs (cos 0.987846 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_mid` | 22 | ⚠️ | 0.965579 | 0.2771 | 37.5 | direction differs (cos 0.965579 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_mid` | 29 | ⚠️ | 0.975014 | 0.2252 | 5.4375 | direction differs (cos 0.975014 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_post` | 22 | ⚠️ | 0.963714 | 0.2786 | 20.125 | direction differs (cos 0.963714 below the `fused` gate of 0.99) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 22 | 0.992157 | 0.824115 | 2 | 0.5856 |
| interp-engine vllm | `attn_in` | 29 | 0.991089 | 0.972933 | 10 | 0.2373 |
| interp-engine vllm | `attn_out` | 15 | 0.991228 | 0.976828 | 10 | 0.2488 |
| interp-engine vllm | `k_norm_out` | 15 | 0.991691 | 0.981389 | 5 | 0.1929 |
| interp-engine vllm | `mlp_out` | 29 | 0.993172 | 0.960196 | 11 | 0.2997 |
| interp-engine vllm | `mlp_out_post` | 29 | 0.995596 | 0.976955 | 11 | 0.2732 |
| interp-engine vllm | `q_norm_in` | 15 | 0.991393 | 0.989002 | 5 | 0.1874 |
| interp-engine vllm | `q_norm_in` | 29 | 0.990428 | 0.952519 | 10 | 0.3118 |
| interp-engine vllm | `q_norm_out` | 15 | 0.993828 | 0.988158 | 5 | 0.1539 |
| interp-engine vllm | `resid_mid` | 15 | 0.994469 | 0.984161 | 10 | 0.1781 |
| interp-engine vllm | `resid_post` | 15 | 0.990908 | 0.945044 | 10 | 0.3387 |
| interp-engine vllm | `resid_post` | 29 | 0.994753 | 0.972513 | 11 | 0.2942 |
| interp-engine vllm-static | `attn_in` | 22 | 0.993144 | 0.839692 | 2 | 0.5608 |
| interp-engine vllm-static | `attn_in` | 29 | 0.991528 | 0.975736 | 10 | 0.2239 |
| interp-engine vllm-static | `k_norm_in` | 15 | 0.990419 | 0.980012 | 5 | 0.2048 |
| interp-engine vllm-static | `k_norm_out` | 15 | 0.99032 | 0.977849 | 5 | 0.2105 |
| interp-engine vllm-static | `mlp_out` | 29 | 0.992876 | 0.966401 | 11 | 0.2882 |
| interp-engine vllm-static | `mlp_out_post` | 29 | 0.995546 | 0.981577 | 11 | 0.261 |
| interp-engine vllm-static | `q_norm_in` | 15 | 0.992713 | 0.986963 | 5 | 0.1612 |
| interp-engine vllm-static | `q_norm_in` | 29 | 0.990699 | 0.950037 | 10 | 0.3176 |
| interp-engine vllm-static | `q_norm_out` | 15 | 0.992763 | 0.985978 | 5 | 0.1675 |
| interp-engine vllm-static | `resid_mid` | 15 | 0.993639 | 0.986954 | 2 | 0.163 |
| interp-engine vllm-static | `resid_post` | 15 | 0.993614 | 0.957216 | 2 | 0.2895 |
| interp-engine vllm-static | `resid_post` | 29 | 0.994672 | 0.977464 | 11 | 0.2798 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
