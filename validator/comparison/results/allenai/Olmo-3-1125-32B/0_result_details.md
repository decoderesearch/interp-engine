# `allenai/Olmo-3-1125-32B` — cross-engine results

Every engine's capture of `allenai/Olmo-3-1125-32B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 32, 48, 63.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 54 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 52 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | bfloat16 | v3.7.0 | 28 | 6 | 2 | 0 |
| [tlens_v3](tlens_v3.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | bfloat16 | v3.7.0 | 28 | 6 | 2 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 32 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 48 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 63 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 32 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 48 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 63 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 32 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 48 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 63 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 32 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 48 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 63 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 32 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 48 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 63 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 32 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 48 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 63 | ref | ✅ | ✅ | 🐞 | 🐞 | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 32 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 48 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 63 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 32 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 48 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 63 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 32 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 48 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 63 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 32 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 48 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 63 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 32 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 48 | ref | ✅ | ✅ | — | — | — |
| `q_norm_in`<br>layer 63 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 32 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 48 | ref | ✅ | ✅ | — | — | — |
| `q_norm_out`<br>layer 63 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 32 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 48 | ref | ✅ | ✅ | — | — | — |
| `k_norm_in`<br>layer 63 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 32 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 48 | ref | ✅ | ✅ | — | — | — |
| `k_norm_out`<br>layer 63 | ref | ✅ | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 32 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 48 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 63 | ref | ✅ | ✅ | — | — | — |

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out_post` | 0 | ⚠️ | 0.576129 | 11.0158 | 9.229 | direction differs (cos 0.576129 below the `tlens` gate of 0.99); same direction, different scale (rel 11.0158 above the `tlens` gate of 0.5) |
| tlens_v2 | `attn_out_post` | 32 | ❌ | 0.426634 | 38.0554 | 61.991 | unrelated direction (cos 0.426634, below 0.5) |
| tlens_v2 | `attn_out_post` | 48 | ❌ | 0.26381 | 79.4132 | 105.5317 | unrelated direction (cos 0.26381, below 0.5) |
| tlens_v2 | `attn_out_post` | 63 | ⚠️ | 0.531401 | 23.9493 | 114.7676 | direction differs (cos 0.531401 below the `tlens` gate of 0.99); same direction, different scale (rel 23.9493 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 0 | ⚠️ | 0.67133 | 9.1777 | 7.0938 | direction differs (cos 0.67133 below the `tlens` gate of 0.99); same direction, different scale (rel 9.1777 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 32 | ⚠️ | 0.788452 | 18.725 | 106.4111 | direction differs (cos 0.788452 below the `tlens` gate of 0.99); same direction, different scale (rel 18.725 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 48 | ⚠️ | 0.747309 | 28.9652 | 167.0225 | direction differs (cos 0.747309 below the `tlens` gate of 0.99); same direction, different scale (rel 28.9652 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 63 | ⚠️ | 0.82729 | 54.9399 | 554.75 | direction differs (cos 0.82729 below the `tlens` gate of 0.99); same direction, different scale (rel 54.9399 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 0 | ⚠️ | 0.576059 | 11.0193 | 9.229 | direction differs (cos 0.576059 below the `tlens` gate of 0.99); same direction, different scale (rel 11.0193 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 32 | ❌ | 0.427097 | 38.0605 | 61.991 | unrelated direction (cos 0.427097, below 0.5) |
| tlens_v3 | `attn_out_post` | 48 | ❌ | 0.263563 | 79.5931 | 105.0317 | unrelated direction (cos 0.263563, below 0.5) |
| tlens_v3 | `attn_out_post` | 63 | ⚠️ | 0.53125 | 23.9419 | 114.2676 | direction differs (cos 0.53125 below the `tlens` gate of 0.99); same direction, different scale (rel 23.9419 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 0 | ⚠️ | 0.671545 | 9.1706 | 7.0626 | direction differs (cos 0.671545 below the `tlens` gate of 0.99); same direction, different scale (rel 9.1706 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 32 | ⚠️ | 0.788602 | 18.7281 | 106.9111 | direction differs (cos 0.788602 below the `tlens` gate of 0.99); same direction, different scale (rel 18.7281 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 48 | ⚠️ | 0.747818 | 28.9342 | 167.0225 | direction differs (cos 0.747818 below the `tlens` gate of 0.99); same direction, different scale (rel 28.9342 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 63 | ⚠️ | 0.827385 | 54.9972 | 554.75 | direction differs (cos 0.827385 below the `tlens` gate of 0.99); same direction, different scale (rel 54.9972 above the `tlens` gate of 0.5) |
