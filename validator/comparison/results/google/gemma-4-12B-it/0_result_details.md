# `google/gemma-4-12B-it` — cross-engine results

Every engine's capture of `google/gemma-4-12B-it`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 24, 36, 47.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.3.5 | — | — | — | — |
| [interp-engine vllm](vllm.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | ok | bfloat16 | v0.27.1 | 29 | 25 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | ok | bfloat16 | v0.27.1 | 28 | 24 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-12B-it not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | bfloat16 | v3.8.0 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1647) | ok | bfloat16 | v3.8.0 | 36 | 0 | 0 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.3.5](eager.json) | interp-engine vllm<br>[v0.27.1](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.8.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 36 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `resid_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 36 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `resid_mid`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 24 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 24 | ref | 🐞 | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 36 | ref | 🐞 | 🐞 | ✅ | — |
| `mlp_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 24 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `attn_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 24 | ref | 🐞 | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 36 | ref | 🐞 | 🐞 | ✅ | — |
| `attn_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 24 | ref | 🐞 | 🐞 | — | ✅ |
| `attn_in`<br>layer 36 | ref | 🐞 | 🐞 | — | ✅ |
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
| `mlp_act`<br>layer 24 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_act`<br>layer 36 | ref | 🐞 | 🐞 | ✅ | ✅ |
| `mlp_act`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 24 | ref | 🐞 | 🐞 | — | — |
| `q_norm_in`<br>layer 36 | ref | 🐞 | 🐞 | — | — |
| `q_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 24 | ref | 🐞 | 🐞 | — | — |
| `q_norm_out`<br>layer 36 | ref | 🐞 | 🐞 | — | — |
| `q_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 24 | ref | 🐞 | 🐞 | — | — |
| `k_norm_in`<br>layer 36 | ref | 🐞 | 🐞 | — | — |
| `k_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 24 | ref | 🐞 | 🐞 | — | — |
| `k_norm_out`<br>layer 36 | ref | 🐞 | 🐞 | — | — |
| `k_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | 🐞 | — | — | — |
| `attn_scores`<br>layer 0 | ref | 🐞 | 🐞 | — | — |
| `attn_scores`<br>layer 24 | ref | 🐞 | 🐞 | — | — |
| `attn_scores`<br>layer 36 | ref | 🐞 | 🐞 | — | — |
| `attn_scores`<br>layer 47 | ref | 🐞 | 🐞 | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 24 | ⚠️ | 0.976871 | 0.2197 | 16.4375 | direction differs (cos 0.976871 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_in` | 36 | ⚠️ | 0.968681 | 0.2501 | 4.9375 | direction differs (cos 0.968681 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 24 | ⚠️ | 0.964675 | 0.2636 | 6.6875 | direction differs (cos 0.964675 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 24 | ⚠️ | 0.962425 | 0.2724 | 3 | direction differs (cos 0.962425 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 36 | ⚠️ | 0.980255 | 0.1995 | 0.6875 | direction differs (cos 0.980255 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_scores` | 0 | ⚠️ | 0.999998 | 0.9375 | 18.6351 | same direction, different scale (rel 0.9375 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 24 | ⚠️ | 0.981911 | 0.9381 | 10.2899 | direction differs (cos 0.981911 below the `fused` gate of 0.99); same direction, different scale (rel 0.9381 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 36 | ⚠️ | 0.981605 | 0.9403 | 10.002 | direction differs (cos 0.981605 below the `fused` gate of 0.99); same direction, different scale (rel 0.9403 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_scores` | 47 | ⚠️ | 0.995646 | 0.9567 | 6.4326 | same direction, different scale (rel 0.9567 above the `fused` gate of 0.5) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.978046 | 0.2113 | 21.0938 | direction differs (cos 0.978046 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 24 | ⚠️ | 0.973742 | 0.2468 | 16 | direction differs (cos 0.973742 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_in` | 36 | ⚠️ | 0.976006 | 0.2184 | 3.0742 | direction differs (cos 0.976006 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 24 | ⚠️ | 0.986329 | 0.1653 | 0.375 | direction differs (cos 0.986329 below the `fused` gate of 0.99) |
| interp-engine vllm | `k_norm_out` | 36 | ⚠️ | 0.974224 | 0.2271 | 0.3467 | direction differs (cos 0.974224 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 24 | ⚠️ | 0.948884 | 0.3224 | 3.1406 | direction differs (cos 0.948884 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_act` | 36 | ⚠️ | 0.965544 | 0.2667 | 3.625 | direction differs (cos 0.965544 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out` | 24 | ⚠️ | 0.984755 | 0.1741 | 0.7969 | direction differs (cos 0.984755 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 24 | ⚠️ | 0.964758 | 0.2645 | 3.625 | direction differs (cos 0.964758 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out_post` | 36 | ⚠️ | 0.955134 | 0.2994 | 0.9355 | direction differs (cos 0.955134 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_in` | 24 | ⚠️ | 0.979526 | 0.2045 | 5.4062 | direction differs (cos 0.979526 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_in` | 36 | ⚠️ | 0.980766 | 0.1958 | 4.1406 | direction differs (cos 0.980766 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 24 | ⚠️ | 0.979633 | 0.2018 | 3.7695 | direction differs (cos 0.979633 below the `fused` gate of 0.99) |
| interp-engine vllm | `q_norm_out` | 36 | ⚠️ | 0.978031 | 0.2096 | 3.0938 | direction differs (cos 0.978031 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_mid` | 36 | ⚠️ | 0.986767 | 0.1623 | 10 | direction differs (cos 0.986767 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_post` | 36 | ⚠️ | 0.986988 | 0.161 | 8 | direction differs (cos 0.986988 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 24 | ⚠️ | 0.976871 | 0.2197 | 16.4375 | direction differs (cos 0.976871 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 36 | ⚠️ | 0.968681 | 0.2501 | 4.9375 | direction differs (cos 0.968681 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 24 | ⚠️ | 0.964675 | 0.2636 | 6.6875 | direction differs (cos 0.964675 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 24 | ⚠️ | 0.962425 | 0.2724 | 3 | direction differs (cos 0.962425 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 36 | ⚠️ | 0.980255 | 0.1995 | 0.6875 | direction differs (cos 0.980255 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_scores` | 0 | ⚠️ | 0.999998 | 0.9375 | 18.6351 | same direction, different scale (rel 0.9375 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 24 | ⚠️ | 0.981911 | 0.9381 | 10.2899 | direction differs (cos 0.981911 below the `fused` gate of 0.99); same direction, different scale (rel 0.9381 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 36 | ⚠️ | 0.981605 | 0.9403 | 10.002 | direction differs (cos 0.981605 below the `fused` gate of 0.99); same direction, different scale (rel 0.9403 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_scores` | 47 | ⚠️ | 0.995646 | 0.9567 | 6.4326 | same direction, different scale (rel 0.9567 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `k_norm_in` | 24 | ⚠️ | 0.973742 | 0.2468 | 16 | direction differs (cos 0.973742 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_in` | 36 | ⚠️ | 0.976006 | 0.2184 | 3.0742 | direction differs (cos 0.976006 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 24 | ⚠️ | 0.986329 | 0.1653 | 0.375 | direction differs (cos 0.986329 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `k_norm_out` | 36 | ⚠️ | 0.974224 | 0.2271 | 0.3467 | direction differs (cos 0.974224 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 24 | ⚠️ | 0.948884 | 0.3224 | 3.1406 | direction differs (cos 0.948884 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_act` | 36 | ⚠️ | 0.965544 | 0.2667 | 3.625 | direction differs (cos 0.965544 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out` | 24 | ⚠️ | 0.984755 | 0.1741 | 0.7969 | direction differs (cos 0.984755 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 24 | ⚠️ | 0.964758 | 0.2645 | 3.625 | direction differs (cos 0.964758 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out_post` | 36 | ⚠️ | 0.955134 | 0.2994 | 0.9355 | direction differs (cos 0.955134 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_in` | 24 | ⚠️ | 0.979526 | 0.2045 | 5.4062 | direction differs (cos 0.979526 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_in` | 36 | ⚠️ | 0.980766 | 0.1958 | 4.1406 | direction differs (cos 0.980766 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 24 | ⚠️ | 0.979633 | 0.2018 | 3.7695 | direction differs (cos 0.979633 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `q_norm_out` | 36 | ⚠️ | 0.978031 | 0.2096 | 3.0938 | direction differs (cos 0.978031 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_mid` | 36 | ⚠️ | 0.986767 | 0.1623 | 10 | direction differs (cos 0.986767 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_post` | 36 | ⚠️ | 0.986988 | 0.161 | 8 | direction differs (cos 0.986988 below the `fused` gate of 0.99) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 47 | 0.994284 | 0.983959 | 8 | 0.1797 |
| interp-engine vllm | `attn_out` | 36 | 0.994006 | 0.980901 | 8 | 0.1948 |
| interp-engine vllm | `attn_out` | 47 | 0.994879 | 0.985841 | 8 | 0.1679 |
| interp-engine vllm | `attn_out_post` | 47 | 0.995033 | 0.98536 | 8 | 0.1714 |
| interp-engine vllm | `k_norm_in` | 47 | 0.994083 | 0.979339 | 8 | 0.2139 |
| interp-engine vllm | `k_norm_out` | 47 | 0.99402 | 0.979334 | 8 | 0.2032 |
| interp-engine vllm | `mlp_act` | 47 | 0.991387 | 0.97611 | 8 | 0.2305 |
| interp-engine vllm | `mlp_out` | 36 | 0.994024 | 0.973263 | 8 | 0.2332 |
| interp-engine vllm | `mlp_out` | 47 | 0.995097 | 0.988054 | 8 | 0.1682 |
| interp-engine vllm | `q_norm_in` | 47 | 0.996367 | 0.987995 | 8 | 0.1633 |
| interp-engine vllm | `q_norm_out` | 47 | 0.995264 | 0.983931 | 8 | 0.1793 |
| interp-engine vllm | `resid_mid` | 47 | 0.991885 | 0.978285 | 8 | 0.2086 |
| interp-engine vllm | `resid_post` | 24 | 0.99802 | 0.989468 | 8 | 0.158 |
| interp-engine vllm | `resid_post` | 47 | 0.994801 | 0.988777 | 10 | 0.1685 |
| interp-engine vllm-static | `attn_in` | 47 | 0.994284 | 0.983959 | 8 | 0.1797 |
| interp-engine vllm-static | `attn_out` | 36 | 0.994006 | 0.980901 | 8 | 0.1948 |
| interp-engine vllm-static | `attn_out` | 47 | 0.994879 | 0.985841 | 8 | 0.1679 |
| interp-engine vllm-static | `attn_out_post` | 47 | 0.995033 | 0.98536 | 8 | 0.1714 |
| interp-engine vllm-static | `k_norm_in` | 47 | 0.994083 | 0.979339 | 8 | 0.2139 |
| interp-engine vllm-static | `k_norm_out` | 47 | 0.99402 | 0.979334 | 8 | 0.2032 |
| interp-engine vllm-static | `mlp_act` | 47 | 0.991387 | 0.97611 | 8 | 0.2305 |
| interp-engine vllm-static | `mlp_out` | 36 | 0.994024 | 0.973263 | 8 | 0.2332 |
| interp-engine vllm-static | `mlp_out` | 47 | 0.995097 | 0.988054 | 8 | 0.1682 |
| interp-engine vllm-static | `q_norm_in` | 47 | 0.996367 | 0.987995 | 8 | 0.1633 |
| interp-engine vllm-static | `q_norm_out` | 47 | 0.995264 | 0.983931 | 8 | 0.1793 |
| interp-engine vllm-static | `resid_mid` | 47 | 0.991885 | 0.978285 | 8 | 0.2086 |
| interp-engine vllm-static | `resid_post` | 24 | 0.99802 | 0.989468 | 8 | 0.158 |
| interp-engine vllm-static | `resid_post` | 47 | 0.994801 | 0.988777 | 10 | 0.1685 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.
