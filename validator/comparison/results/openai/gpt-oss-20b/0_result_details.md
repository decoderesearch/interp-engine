# `openai/gpt-oss-20b` — cross-engine results

Every engine's capture of `openai/gpt-oss-20b`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 38 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 36 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.7.0 | 24 | 0 | 0 | 8 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 24 | 0 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 20 | 0 | 0 | 8 |

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
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 23 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | n/a | — | — | — | no ref | — |
| `mlp_pre`<br>layer 12 | n/a | — | — | — | no ref | — |
| `mlp_pre`<br>layer 18 | n/a | — | — | — | no ref | — |
| `mlp_pre`<br>layer 23 | n/a | — | — | — | no ref | — |
| `mlp_act`<br>layer 0 | n/a | — | — | — | no ref | — |
| `mlp_act`<br>layer 12 | n/a | — | — | — | no ref | — |
| `mlp_act`<br>layer 18 | n/a | — | — | — | no ref | — |
| `mlp_act`<br>layer 23 | n/a | — | — | — | no ref | — |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `router_logits`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `router_logits`<br>layer 18 | ref | ✅ | ✅ | — | — | — |
| `router_logits`<br>layer 23 | ref | ✅ | ✅ | — | — | — |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 18 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 23 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_out` | 12 | 0.996748 | 0.989827 | 3 | 0.1424 |
| interp-engine vllm | `attn_out_post` | 12 | 0.996748 | 0.989827 | 3 | 0.1424 |
| interp-engine vllm | `final_norm` | — | 0.996255 | 0.966391 | 3 | 0.2579 |
| interp-engine vllm | `mlp_out` | 12 | 0.997847 | 0.986677 | 5 | 0.1627 |
| interp-engine vllm | `mlp_out` | 23 | 0.99716 | 0.953151 | 3 | 0.3838 |
| interp-engine vllm | `mlp_out_post` | 12 | 0.997847 | 0.986677 | 5 | 0.1627 |
| interp-engine vllm | `mlp_out_post` | 23 | 0.99716 | 0.953151 | 3 | 0.3838 |
| interp-engine vllm | `router_logits` | 23 | 0.997716 | 0.97971 | 3 | 0.2683 |
| interp-engine vllm-static | `attn_out` | 12 | 0.995214 | 0.989368 | 3 | 0.1457 |
| interp-engine vllm-static | `attn_out_post` | 12 | 0.995214 | 0.989368 | 3 | 0.1457 |
| interp-engine vllm-static | `mlp_out` | 12 | 0.99689 | 0.986828 | 5 | 0.1619 |
| interp-engine vllm-static | `mlp_out` | 23 | 0.998107 | 0.954037 | 3 | 0.3734 |
| interp-engine vllm-static | `mlp_out_post` | 12 | 0.99689 | 0.986828 | 5 | 0.1619 |
| interp-engine vllm-static | `mlp_out_post` | 23 | 0.998107 | 0.954037 | 3 | 0.3734 |
| interp-engine vllm-static | `router_logits` | 23 | 0.997498 | 0.978059 | 3 | 0.276 |
| tlens_v2 | `mlp_out` | 23 | 0.99876 | 0.977798 | 9 | 0.2126 |
| tlens_v2 | `mlp_out_post` | 23 | 0.99876 | 0.977798 | 9 | 0.2126 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 0, 12, 18, 23 | nnsight, tlens_v2 | neither engine captured it |
| `mlp_pre` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 12, 18, 23 | nnsight, tlens_v2, interp-engine vllm | neither engine captured it |
| `mlp_act` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
