# `bigscience/bloom-560m` — cross-engine results

Every engine's capture of `bigscience/bloom-560m`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.26.0 | 34 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.27.1 | 32 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | float32 | v3.7.0 | 32 | 0 | 0 | 0 |
| [tlens_v3](tlens_v3.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1639) | ok | float32 | v3.7.0 | 16 | 2 | 14 | 0 |
| [nnsight](nnsight.json) | [🐞](https://github.com/ndif-team/nnterp/issues/51) | ok | float32 | v0.7.0 | 20 | 0 | 8 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `mlp_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `attn_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | 🐞 | 🐞 |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | 🐞 | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 23 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 12 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 18 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 23 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | n/a | no ref | no ref | — | — | — |
| `attn_scores`<br>layer 12 | n/a | no ref | no ref | — | — | — |
| `attn_scores`<br>layer 18 | n/a | no ref | no ref | — | — | — |
| `attn_scores`<br>layer 23 | n/a | no ref | no ref | — | — | — |

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v3 | `attn_out` | 0 | ❌ | 0.400386 | 2.3257 | 8.1797 | unrelated direction (cos 0.400386, below 0.5) |
| tlens_v3 | `attn_out` | 12 | ❌ | 0.053246 | 56.2909 | 516.9155 | unrelated direction (cos 0.053246, below 0.5) |
| tlens_v3 | `attn_out` | 18 | ❌ | 0.061402 | 47.6459 | 521.5597 | unrelated direction (cos 0.061402, below 0.5) |
| tlens_v3 | `attn_out` | 23 | ❌ | 0.369461 | 13.4075 | 50.2084 | unrelated direction (cos 0.369461, below 0.5) |
| tlens_v3 | `attn_out_post` | 0 | ❌ | 0.400386 | 2.3257 | 8.1797 | unrelated direction (cos 0.400386, below 0.5) |
| tlens_v3 | `attn_out_post` | 12 | ❌ | 0.053246 | 56.2909 | 516.9155 | unrelated direction (cos 0.053246, below 0.5) |
| tlens_v3 | `attn_out_post` | 18 | ❌ | 0.061402 | 47.6459 | 521.5597 | unrelated direction (cos 0.061402, below 0.5) |
| tlens_v3 | `attn_out_post` | 23 | ❌ | 0.369461 | 13.4075 | 50.2084 | unrelated direction (cos 0.369461, below 0.5) |
| tlens_v3 | `mlp_out` | 0 | ⚠️ | 0.652846 | 1.4701 | 8.1542 | direction differs (cos 0.652846 below the `tlens` gate of 0.99); same direction, different scale (rel 1.4701 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out` | 12 | ❌ | 0.083558 | 34.2752 | 517.0175 | unrelated direction (cos 0.083558, below 0.5) |
| tlens_v3 | `mlp_out` | 18 | ❌ | 0.069371 | 25.5609 | 521.4425 | unrelated direction (cos 0.069371, below 0.5) |
| tlens_v3 | `mlp_out` | 23 | ❌ | 0.036179 | 15.0037 | 52.7476 | unrelated direction (cos 0.036179, below 0.5) |
| tlens_v3 | `mlp_out_post` | 0 | ⚠️ | 0.652846 | 1.4701 | 8.1542 | direction differs (cos 0.652846 below the `tlens` gate of 0.99); same direction, different scale (rel 1.4701 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 12 | ❌ | 0.083558 | 34.2752 | 517.0175 | unrelated direction (cos 0.083558, below 0.5) |
| tlens_v3 | `mlp_out_post` | 18 | ❌ | 0.069371 | 25.5609 | 521.4425 | unrelated direction (cos 0.069371, below 0.5) |
| tlens_v3 | `mlp_out_post` | 23 | ❌ | 0.036179 | 15.0037 | 52.7476 | unrelated direction (cos 0.036179, below 0.5) |
| nnsight | `attn_out` | 0 | ❌ | 0.400386 | 2.3257 | 8.1797 | unrelated direction (cos 0.400386, below 0.5) |
| nnsight | `attn_out` | 12 | ❌ | 0.053246 | 56.2909 | 516.9155 | unrelated direction (cos 0.053246, below 0.5) |
| nnsight | `attn_out` | 18 | ❌ | 0.061402 | 47.6459 | 521.5597 | unrelated direction (cos 0.061402, below 0.5) |
| nnsight | `attn_out` | 23 | ❌ | 0.369461 | 13.4075 | 50.2084 | unrelated direction (cos 0.369461, below 0.5) |
| nnsight | `mlp_out` | 0 | ❌ | 0.652846 | 1.4701 | 8.1542 | direction differs (cos 0.652846 below the `raw_hf` gate of 0.9999); absolute diff 8.154169 above the `raw_hf` gate of 0.002 |
| nnsight | `mlp_out` | 12 | ❌ | 0.083558 | 34.2752 | 517.0175 | unrelated direction (cos 0.083558, below 0.5) |
| nnsight | `mlp_out` | 18 | ❌ | 0.069371 | 25.5609 | 521.4425 | unrelated direction (cos 0.069371, below 0.5) |
| nnsight | `mlp_out` | 23 | ❌ | 0.036179 | 15.0037 | 52.7475 | unrelated direction (cos 0.036179, below 0.5) |

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `attn_scores` | 0, 12, 18, 23 | interp-engine vllm, interp-engine vllm-static | the `eager` reference declined the point, so there is nothing to score against — BLOOM computes attention with its own fused `baddbmm` path rather than delegating to `eager_attention_forward`, so the recompute interp-engine reaches the scores through has nothing to intercept on this family |
