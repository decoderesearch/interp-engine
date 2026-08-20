# `allenai/Olmo-3-1025-7B` — cross-engine results

Every engine's capture of `allenai/Olmo-3-1025-7B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 16, 24, 31.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.0.1 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 54 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1620) | ok | bfloat16 | v3.7.0 | 28 | 3 | 5 | 0 |
| [tlens_v3](tlens_v3.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | bfloat16 | v3.7.0 | 28 | 3 | 5 | 0 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.0.1](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 16 | ref | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 24 | ref | ✅ | 🐞 | 🐞 | — |
| `mlp_out_post`<br>layer 31 | ref | ✅ | 🐞 | 🐞 | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 16 | ref | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 24 | ref | ✅ | 🐞 | 🐞 | — |
| `attn_out_post`<br>layer 31 | ref | ✅ | 🐞 | 🐞 | — |
| `attn_in`<br>layer 0 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 16 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 24 | ref | ✅ | — | — | ✅ |
| `attn_in`<br>layer 31 | ref | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 16 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 24 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 31 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 16 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 24 | ref | — | ✅ | ✅ | ✅ |
| `mlp_pre_linear`<br>layer 31 | ref | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `q_norm_in`<br>layer 0 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 16 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 24 | ref | ✅ | — | — | — |
| `q_norm_in`<br>layer 31 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 16 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 24 | ref | ✅ | — | — | — |
| `q_norm_out`<br>layer 31 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 16 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 24 | ref | ✅ | — | — | — |
| `k_norm_in`<br>layer 31 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 16 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 24 | ref | ✅ | — | — | — |
| `k_norm_out`<br>layer 31 | ref | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 16 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 24 | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 31 | ref | ✅ | — | — | — |

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v2 | `attn_out_post` | 0 | ❌ | 0.300545 | 13.5654 | 12.9979 | unrelated direction (cos 0.300545, below 0.5) |
| tlens_v2 | `attn_out_post` | 16 | ❌ | 0.376807 | 11.6636 | 32.9859 | unrelated direction (cos 0.376807, below 0.5) |
| tlens_v2 | `attn_out_post` | 24 | ❌ | 0.423474 | 20.2825 | 34.3135 | unrelated direction (cos 0.423474, below 0.5) |
| tlens_v2 | `attn_out_post` | 31 | ❌ | 0.425765 | 9.0857 | 49.4997 | unrelated direction (cos 0.425765, below 0.5) |
| tlens_v2 | `mlp_out_post` | 0 | ❌ | 0.455113 | 3.5266 | 1.9979 | unrelated direction (cos 0.455113, below 0.5) |
| tlens_v2 | `mlp_out_post` | 16 | ⚠️ | 0.830165 | 2.4939 | 8.3777 | direction differs (cos 0.830165 below the `tlens` gate of 0.99); same direction, different scale (rel 2.4939 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 24 | ⚠️ | 0.773502 | 7.8536 | 50.9624 | direction differs (cos 0.773502 below the `tlens` gate of 0.99); same direction, different scale (rel 7.8536 above the `tlens` gate of 0.5) |
| tlens_v2 | `mlp_out_post` | 31 | ⚠️ | 0.862571 | 20.144 | 171.125 | direction differs (cos 0.862571 below the `tlens` gate of 0.99); same direction, different scale (rel 20.144 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 0 | ❌ | 0.300649 | 13.5644 | 12.9979 | unrelated direction (cos 0.300649, below 0.5) |
| tlens_v3 | `attn_out_post` | 16 | ❌ | 0.376925 | 11.6761 | 32.9859 | unrelated direction (cos 0.376925, below 0.5) |
| tlens_v3 | `attn_out_post` | 24 | ❌ | 0.423353 | 20.2474 | 34.3135 | unrelated direction (cos 0.423353, below 0.5) |
| tlens_v3 | `attn_out_post` | 31 | ❌ | 0.425269 | 9.1044 | 49.4997 | unrelated direction (cos 0.425269, below 0.5) |
| tlens_v3 | `mlp_out_post` | 0 | ❌ | 0.455586 | 3.5237 | 1.9979 | unrelated direction (cos 0.455586, below 0.5) |
| tlens_v3 | `mlp_out_post` | 16 | ⚠️ | 0.83063 | 2.4919 | 8.3152 | direction differs (cos 0.83063 below the `tlens` gate of 0.99); same direction, different scale (rel 2.4919 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 24 | ⚠️ | 0.77398 | 7.855 | 50.9624 | direction differs (cos 0.77398 below the `tlens` gate of 0.99); same direction, different scale (rel 7.855 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 31 | ⚠️ | 0.862476 | 20.1535 | 170.125 | direction differs (cos 0.862476 below the `tlens` gate of 0.99); same direction, different scale (rel 20.1535 above the `tlens` gate of 0.5) |
