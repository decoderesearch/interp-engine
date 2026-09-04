# `google/gemma-4-12B-it` — cross-engine results

Every engine's capture of `google/gemma-4-12B-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 24, 36, 47.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ⚠️ | ok | bfloat16 | v0.28.0 | 43 | 19 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ⚠️ | ok | bfloat16 | v0.28.0 | 42 | 18 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-12B-it not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 24 | ref | ⚠️ | ⚠️ | ✅ | — |
| `mlp_out_post`<br>layer 36 | ref | ⚠️ | ⚠️ | ✅ | — |
| `mlp_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 24 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `attn_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 24 | ref | ⚠️ | ⚠️ | ✅ | — |
| `attn_out_post`<br>layer 36 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 24 | ref | ⚠️ | ⚠️ | — | ✅ |
| `attn_in`<br>layer 36 | ref | ⚠️ | ⚠️ | — | ✅ |
| `attn_in`<br>layer 47 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 24 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 36 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 47 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 24 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 36 | ref | — | — | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 47 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 24 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `mlp_act`<br>layer 36 | ref | ⚠️ | ⚠️ | ✅ | ✅ |
| `mlp_act`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 36 | ref | ⚠️ | ⚠️ | — | — |
| `q_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 24 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_in`<br>layer 36 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 36 | ref | ⚠️ | ⚠️ | — | — |
| `k_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 24 | ref | ⚠️ | ⚠️ | — | — |
| `value`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 24 | ref | ⚠️ | ⚠️ | — | — |
| `z`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ⚠️ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ⚠️ | ⚠️ | — | — |
| `attn_scores`<br>layer 24 | ref | ⚠️ | ⚠️ | — | — |
| `attn_scores`<br>layer 36 | ref | ⚠️ | ⚠️ | — | — |
| `attn_scores`<br>layer 47 | ref | ⚠️ | ⚠️ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 24 | ⚠️ | 0.978888 | 0.2092 | 14.5625 | direction differs (cos 0.978888 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_in` | 36 | ⚠️ | 0.970533 | 0.2424 | 5.5938 | direction differs (cos 0.970533 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out` | 24 | ⚠️ | 0.967107 | 0.2546 | 5.6875 | direction differs (cos 0.967107 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_out_post` | 24 | ⚠️ | 0.966221 | 0.259 | 2.375 | direction differs (cos 0.966221 below the `fused` gate of 0.98) |
| interp-engine vllm | `attn_scores` | 0 | ⚠️ | 0.999998 | 0.9375 | 18.6351 | same direction, different scale (rel 0.9375 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 24 | ⚠️ | 0.983359 | 0.9383 | 10.2843 | same direction, different scale (rel 0.9383 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 36 | ⚠️ | 0.983012 | 0.9403 | 9.9979 | same direction, different scale (rel 0.9403 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 47 | ⚠️ | 0.996147 | 0.9565 | 6.4327 | same direction, different scale (rel 0.9565 above the `fused` gate of 0.5) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.978952 | 0.2071 | 20.125 | direction differs (cos 0.978952 below the `fused` gate of 0.98) |
| interp-engine vllm | `k_norm_in` | 24 | ⚠️ | 0.97874 | 0.2188 | 12.5 | direction differs (cos 0.97874 below the `fused` gate of 0.98) |
| interp-engine vllm | `k_norm_in` | 36 | ⚠️ | 0.977722 | 0.2104 | 2.6562 | direction differs (cos 0.977722 below the `fused` gate of 0.98) |
| interp-engine vllm | `k_norm_out` | 36 | ⚠️ | 0.975766 | 0.2202 | 0.323 | direction differs (cos 0.975766 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_act` | 24 | ⚠️ | 0.954381 | 0.304 | 2.7656 | direction differs (cos 0.954381 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_act` | 36 | ⚠️ | 0.967111 | 0.2604 | 3.4219 | direction differs (cos 0.967111 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_out_post` | 24 | ⚠️ | 0.968105 | 0.2513 | 2.875 | direction differs (cos 0.968105 below the `fused` gate of 0.98) |
| interp-engine vllm | `mlp_out_post` | 36 | ⚠️ | 0.956087 | 0.2973 | 0.9453 | direction differs (cos 0.956087 below the `fused` gate of 0.98) |
| interp-engine vllm | `q_norm_out` | 36 | ⚠️ | 0.979826 | 0.2009 | 3.0156 | direction differs (cos 0.979826 below the `fused` gate of 0.98) |
| interp-engine vllm | `value` | 24 | ⚠️ | 0.975843 | 0.2198 | 8.0625 | direction differs (cos 0.975843 below the `fused` gate of 0.98) |
| interp-engine vllm | `z` | 24 | ⚠️ | 0.951431 | 0.3104 | 9.3203 | direction differs (cos 0.951431 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_in` | 24 | ⚠️ | 0.978888 | 0.2092 | 14.5625 | direction differs (cos 0.978888 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_in` | 36 | ⚠️ | 0.970533 | 0.2424 | 5.5938 | direction differs (cos 0.970533 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out` | 24 | ⚠️ | 0.967107 | 0.2546 | 5.6875 | direction differs (cos 0.967107 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_out_post` | 24 | ⚠️ | 0.966221 | 0.259 | 2.375 | direction differs (cos 0.966221 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `attn_scores` | 0 | ⚠️ | 0.999998 | 0.9375 | 18.6351 | same direction, different scale (rel 0.9375 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 24 | ⚠️ | 0.983359 | 0.9383 | 10.2843 | same direction, different scale (rel 0.9383 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 36 | ⚠️ | 0.983012 | 0.9403 | 9.9979 | same direction, different scale (rel 0.9403 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 47 | ⚠️ | 0.996147 | 0.9565 | 6.4327 | same direction, different scale (rel 0.9565 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `k_norm_in` | 24 | ⚠️ | 0.97874 | 0.2188 | 12.5 | direction differs (cos 0.97874 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_in` | 36 | ⚠️ | 0.977722 | 0.2104 | 2.6562 | direction differs (cos 0.977722 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `k_norm_out` | 36 | ⚠️ | 0.975766 | 0.2202 | 0.323 | direction differs (cos 0.975766 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_act` | 24 | ⚠️ | 0.954381 | 0.304 | 2.7656 | direction differs (cos 0.954381 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_act` | 36 | ⚠️ | 0.967111 | 0.2604 | 3.4219 | direction differs (cos 0.967111 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_out_post` | 24 | ⚠️ | 0.968105 | 0.2513 | 2.875 | direction differs (cos 0.968105 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `mlp_out_post` | 36 | ⚠️ | 0.956087 | 0.2973 | 0.9453 | direction differs (cos 0.956087 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `q_norm_out` | 36 | ⚠️ | 0.979826 | 0.2009 | 3.0156 | direction differs (cos 0.979826 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `value` | 24 | ⚠️ | 0.975843 | 0.2198 | 8.0625 | direction differs (cos 0.975843 below the `fused` gate of 0.98) |
| interp-engine vllm-static | `z` | 24 | ⚠️ | 0.951431 | 0.3104 | 9.3203 | direction differs (cos 0.951431 below the `fused` gate of 0.98) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_out_post` | 36 | 0.980418 | 0.92596 | 8 | 0.3898 |
| interp-engine vllm | `k_norm_out` | 24 | 0.987095 | 0.947793 | 8 | 0.3231 |
| interp-engine vllm | `mlp_act` | 47 | 0.992144 | 0.979378 | 10 | 0.2065 |
| interp-engine vllm | `mlp_out` | 24 | 0.986244 | 0.88196 | 8 | 0.503 |
| interp-engine vllm | `mlp_out` | 36 | 0.993985 | 0.974669 | 8 | 0.2254 |
| interp-engine vllm | `q_norm_in` | 24 | 0.980909 | 0.877344 | 8 | 0.5281 |
| interp-engine vllm | `q_norm_in` | 36 | 0.982538 | 0.898498 | 8 | 0.4446 |
| interp-engine vllm | `q_norm_out` | 24 | 0.980596 | 0.873017 | 8 | 0.5041 |
| interp-engine vllm | `resid_mid` | 36 | 0.987684 | 0.934927 | 8 | 0.3553 |
| interp-engine vllm | `resid_mid` | 47 | 0.99223 | 0.979307 | 10 | 0.2024 |
| interp-engine vllm | `resid_post` | 36 | 0.987854 | 0.936095 | 8 | 0.3526 |
| interp-engine vllm | `value` | 36 | 0.989256 | 0.94999 | 8 | 0.3163 |
| interp-engine vllm-static | `attn_out_post` | 36 | 0.980418 | 0.92596 | 8 | 0.3898 |
| interp-engine vllm-static | `k_norm_out` | 24 | 0.987095 | 0.947793 | 8 | 0.3231 |
| interp-engine vllm-static | `mlp_act` | 47 | 0.992144 | 0.979378 | 10 | 0.2065 |
| interp-engine vllm-static | `mlp_out` | 24 | 0.986244 | 0.88196 | 8 | 0.503 |
| interp-engine vllm-static | `mlp_out` | 36 | 0.993985 | 0.974669 | 8 | 0.2254 |
| interp-engine vllm-static | `q_norm_in` | 24 | 0.980909 | 0.877344 | 8 | 0.5281 |
| interp-engine vllm-static | `q_norm_in` | 36 | 0.982538 | 0.898498 | 8 | 0.4446 |
| interp-engine vllm-static | `q_norm_out` | 24 | 0.980596 | 0.873017 | 8 | 0.5041 |
| interp-engine vllm-static | `resid_mid` | 36 | 0.987684 | 0.934927 | 8 | 0.3553 |
| interp-engine vllm-static | `resid_mid` | 47 | 0.99223 | 0.979307 | 10 | 0.2024 |
| interp-engine vllm-static | `resid_post` | 36 | 0.987854 | 0.936095 | 8 | 0.3526 |
| interp-engine vllm-static | `value` | 36 | 0.989256 | 0.94999 | 8 | 0.3163 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
