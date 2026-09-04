# `google/gemma-4-26B-A4B-it` — cross-engine results

Every engine's capture of `google/gemma-4-26B-A4B-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 15, 22, 29.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ⚠️ | ok | bfloat16 | v0.28.0 | 46 | 16 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | [🐞](https://github.com/vllm-project/vllm/issues/55238) | ok | bfloat16 | v0.28.0 | 27 | 32 | 1 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-26B-A4B-it not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mG… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 32 | 0 | 0 | 4 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 28 | 0 | 0 | 4 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 15 | ref | ✅ | 🐞 | ✅ | ✅ |
| `resid_post`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `resid_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `resid_mid`<br>layer 29 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `mlp_out`<br>layer 0 | n/a | — | — | no ref | no ref |
| `mlp_out`<br>layer 15 | n/a | — | — | no ref | no ref |
| `mlp_out`<br>layer 22 | n/a | — | — | no ref | no ref |
| `mlp_out`<br>layer 29 | n/a | — | — | no ref | no ref |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 15 | ref | ⚠️ | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 29 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 15 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `attn_out`<br>layer 29 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 15 | ref | ⚠️ | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 29 | ref | ⚠️ | 🐞 | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 15 | ref | ✅ | 🐞 | — | ✅ |
| `attn_in`<br>layer 22 | ref | ✅ | 🐞 | — | ✅ |
| `attn_in`<br>layer 29 | ref | ✅ | 🐞 | — | ✅ |
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
| `mlp_act`<br>layer 22 | ref | ⚠️ | 🐞 | ✅ | ✅ |
| `mlp_act`<br>layer 29 | ref | ✅ | 🐞 | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 22 | ref | ⚠️ | 🐞 | — | — |
| `q_norm_in`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 22 | ref | ✅ | 🐞 | — | — |
| `q_norm_out`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 15 | ref | ✅ | 🐞 | — | — |
| `k_norm_in`<br>layer 22 | ref | ⚠️ | 🐞 | — | — |
| `k_norm_in`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 22 | ref | ✅ | 🐞 | — | — |
| `k_norm_out`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 15 | ref | ✅ | 🐞 | — | — |
| `value`<br>layer 22 | ref | ⚠️ | 🐞 | — | — |
| `value`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 22 | ref | ⚠️ | 🐞 | — | — |
| `z`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 22 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 29 | ref | ✅ | 🐞 | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ⚠️ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 15 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 22 | ref | ✅ | 🐞 | — | — |
| `attn_scores`<br>layer 29 | ref | ✅ | 🐞 | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_out` | 22 | ⚠️ | 0.979237 | 0.2063 | 3 | direction differs (cos 0.979237 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out` | 29 | ⚠️ | 0.969315 | 0.2474 | 4.6875 | direction differs (cos 0.969315 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out_post` | 15 | ⚠️ | 0.978144 | 0.2121 | 1.4531 | direction differs (cos 0.978144 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out_post` | 22 | ⚠️ | 0.960195 | 0.2827 | 1.0977 | direction differs (cos 0.960195 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out_post` | 29 | ⚠️ | 0.973692 | 0.2294 | 4.4062 | direction differs (cos 0.973692 below the `fused` gate of 0.98) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.977109 | 0.2154 | 10.5938 | direction differs (cos 0.977109 below the `fused` gate of 0.98) |
| interp-engine vllm | `k_norm_in` | 22 | ⚠️ | 0.979072 | 0.2651 | 29 | direction differs (cos 0.979072 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_act` | 22 | ⚠️ | 0.978272 | 0.2081 | 5.2812 | direction differs (cos 0.978272 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_out_post` | 15 | ⚠️ | 0.970312 | 0.2427 | 34.875 | direction differs (cos 0.970312 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_out_post` | 22 | ⚠️ | 0.935956 | 0.3792 | 21.6875 | direction differs (cos 0.935956 below the `fused` gate of 0.98) |
| interp-engine vllm | `q_norm_in` | 22 | ⚠️ | 0.976747 | 0.2144 | 10.3438 | direction differs (cos 0.976747 below the `fused` gate of 0.98) |
| interp-engine vllm | `resid_mid` | 22 | ⚠️ | 0.968983 | 0.2629 | 26 | direction differs (cos 0.968983 below the `fused` gate of 0.98) |
| interp-engine vllm | `resid_mid` | 29 | ⚠️ | 0.971139 | 0.2416 | 4.9375 | direction differs (cos 0.971139 below the `fused` gate of 0.98) |
| interp-engine vllm | `resid_post` | 22 | ⚠️ | 0.966802 | 0.2666 | 14.75 | direction differs (cos 0.966802 below the `fused` gate of 0.98) |
| interp-engine vllm | `value` | 22 | ⚠️ | 0.979154 | 0.2042 | 3.8125 | direction differs (cos 0.979154 below the `fused` gate of 0.98) |
| interp-engine vllm | `z` | 22 | ⚠️ | 0.976979 | 0.214 | 7.3594 | direction differs (cos 0.976979 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_in` | 15 | ⚠️ | 0.978003 | 0.2109 | 41 | direction differs (cos 0.978003 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_in` | 22 | ❌ | 0.001192 | 1.0619 | 493.375 | unrelated direction (cos 0.001192, below 0.5) |
| interp-engine vllm-static | `attn_in` | 29 | ⚠️ | 0.974722 | 0.2252 | 9.4688 | direction differs (cos 0.974722 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out` | 22 | ⚠️ | 0.954579 | 0.2996 | 9.625 | direction differs (cos 0.954579 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out` | 29 | ⚠️ | 0.934264 | 0.3664 | 5.4297 | direction differs (cos 0.934264 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out_post` | 15 | ⚠️ | 0.976775 | 0.2179 | 1.3281 | direction differs (cos 0.976775 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out_post` | 22 | ⚠️ | 0.920358 | 0.4004 | 1.7891 | direction differs (cos 0.920358 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out_post` | 29 | ⚠️ | 0.941125 | 0.3419 | 3.6211 | direction differs (cos 0.941125 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_scores` | 22 | ⚠️ | 0.96127 | 0.2758 | 7.5026 | direction differs (cos 0.96127 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_scores` | 29 | ⚠️ | 0.968796 | 0.2529 | 3.9122 | direction differs (cos 0.968796 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_in` | 15 | ⚠️ | 0.974869 | 0.2245 | 32.25 | direction differs (cos 0.974869 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_in` | 22 | ⚠️ | 0.7178 | 0.7019 | 106.5625 | direction differs (cos 0.7178 below the `fused` gate of 0.98); same direction, different scale (rel 0.7019 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `k_norm_in` | 29 | ⚠️ | 0.956149 | 0.2944 | 2.6953 | direction differs (cos 0.956149 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_out` | 22 | ⚠️ | 0.945832 | 0.3291 | 2.1914 | direction differs (cos 0.945832 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_out` | 29 | ⚠️ | 0.95156 | 0.3112 | 0.2939 | direction differs (cos 0.95156 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_act` | 22 | ⚠️ | 0.91774 | 0.4005 | 16.1914 | direction differs (cos 0.91774 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_act` | 29 | ⚠️ | 0.968461 | 0.2553 | 12.2188 | direction differs (cos 0.968461 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_out_post` | 15 | ⚠️ | 0.871039 | 0.5035 | 75.125 | direction differs (cos 0.871039 below the `fused` gate of 0.98); same direction, different scale (rel 0.5035 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `mlp_out_post` | 22 | ⚠️ | 0.877775 | 0.4922 | 28.6875 | direction differs (cos 0.877775 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `q_norm_in` | 22 | ⚠️ | 0.927233 | 0.3822 | 15.5625 | direction differs (cos 0.927233 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `q_norm_in` | 29 | ⚠️ | 0.974055 | 0.2264 | 4.9375 | direction differs (cos 0.974055 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `q_norm_out` | 22 | ⚠️ | 0.941172 | 0.343 | 11.375 | direction differs (cos 0.941172 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `q_norm_out` | 29 | ⚠️ | 0.967229 | 0.256 | 4.5117 | direction differs (cos 0.967229 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `resid_mid` | 22 | ⚠️ | 0.899081 | 0.4555 | 74.25 | direction differs (cos 0.899081 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `resid_mid` | 29 | ⚠️ | 0.941173 | 0.344 | 4.6055 | direction differs (cos 0.941173 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `resid_post` | 15 | ⚠️ | 0.94971 | 0.3169 | 38.8125 | direction differs (cos 0.94971 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `resid_post` | 22 | ⚠️ | 0.91402 | 0.4223 | 46.625 | direction differs (cos 0.91402 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `router_logits` | 29 | ⚠️ | 0.976406 | 0.2163 | 1.1421 | direction differs (cos 0.976406 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `value` | 15 | ⚠️ | 0.979453 | 0.2027 | 2.3125 | direction differs (cos 0.979453 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `value` | 22 | ⚠️ | 0.833517 | 0.577 | 31.75 | direction differs (cos 0.833517 below the `fused` gate of 0.98); same direction, different scale (rel 0.577 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `value` | 29 | ⚠️ | 0.951574 | 0.3112 | 4.7812 | direction differs (cos 0.951574 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `z` | 22 | ⚠️ | 0.902599 | 0.4565 | 26.5 | direction differs (cos 0.902599 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `z` | 29 | ⚠️ | 0.959156 | 0.286 | 4.3555 | direction differs (cos 0.959156 below the `fused` gate of 0.98) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 22 | 0.992327 | 0.764182 | 2 | 0.6823 |
| interp-engine vllm | `attn_in` | 29 | 0.989669 | 0.97406 | 10 | 0.2341 |
| interp-engine vllm | `attn_out` | 15 | 0.988587 | 0.972433 | 10 | 0.2817 |
| interp-engine vllm | `k_norm_in` | 29 | 0.982257 | 0.929819 | 10 | 0.3708 |
| interp-engine vllm | `k_norm_out` | 22 | 0.982474 | 0.888817 | 2 | 0.4715 |
| interp-engine vllm | `k_norm_out` | 29 | 0.981154 | 0.926571 | 10 | 0.3832 |
| interp-engine vllm | `mlp_act` | 29 | 0.984418 | 0.919606 | 11 | 0.42 |
| interp-engine vllm | `mlp_out_post` | 29 | 0.992571 | 0.966138 | 11 | 0.4052 |
| interp-engine vllm | `q_norm_in` | 29 | 0.989109 | 0.954077 | 10 | 0.3077 |
| interp-engine vllm | `q_norm_out` | 22 | 0.980103 | 0.894142 | 10 | 0.4601 |
| interp-engine vllm | `q_norm_out` | 29 | 0.9866 | 0.947001 | 10 | 0.3256 |
| interp-engine vllm | `resid_post` | 15 | 0.987392 | 0.906857 | 2 | 0.4235 |
| interp-engine vllm | `resid_post` | 29 | 0.99156 | 0.961791 | 11 | 0.42 |
| interp-engine vllm | `router_logits` | 22 | 0.996245 | 0.662511 | 10 | 0.7677 |
| interp-engine vllm | `router_logits` | 29 | 0.987792 | 0.960887 | 11 | 0.2871 |
| interp-engine vllm | `value` | 15 | 0.982238 | 0.95555 | 5 | 0.2981 |
| interp-engine vllm | `value` | 29 | 0.98116 | 0.926572 | 10 | 0.3832 |
| interp-engine vllm | `z` | 15 | 0.985 | 0.969329 | 5 | 0.25 |
| interp-engine vllm | `z` | 29 | 0.982272 | 0.956775 | 10 | 0.2919 |
| interp-engine vllm-static | `attn_out` | 15 | 0.987435 | 0.962885 | 10 | 0.3152 |
| interp-engine vllm-static | `k_norm_out` | 15 | 0.989889 | 0.977952 | 10 | 0.21 |
| interp-engine vllm-static | `mlp_out_post` | 29 | 0.984499 | 0.933509 | 11 | 0.7683 |
| interp-engine vllm-static | `resid_mid` | 15 | 0.992086 | 0.972084 | 10 | 0.2357 |
| interp-engine vllm-static | `resid_post` | 29 | 0.982364 | 0.919141 | 11 | 0.796 |
| interp-engine vllm-static | `router_logits` | 22 | 0.985766 | 0.896938 | 10 | 1.6374 |
| interp-engine vllm-static | `z` | 15 | 0.984463 | 0.975139 | 5 | 0.225 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_out` | 0, 15, 22, 29 | interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_out` | 0, 15, 22, 29 | nnsight, tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — experts beside the dense MLP: every layer here keeps `layer.mlp` and hangs the router and experts on the *block*, then sums the two branches in its own forward -- so the module this point taps is the dense half of a two-branch feed-forward. The unusual thing about this gap is that it is not a disagreement: both engines built the same tree and returned the same half, and the cell was green at cos 0.9999 for four layers before anyone read what the tensor was. So it is refused on both (`EagerModel._require_whole_feed_forward` and `vllm_capture._tree._split_feed_forward_reason`) rather than compared. `mlp_out_post` is the post-feedforward norm's output, downstream of the sum, and is scored on both engines -- it is also the row that keeps the residual decomposition checkable here, since resid_post == resid_mid + mlp_out_post holds for it and not for the raw point |
