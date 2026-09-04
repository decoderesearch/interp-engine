# `openai/gpt-oss-20b` — cross-engine results

Every engine's capture of `openai/gpt-oss-20b`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 46 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 44 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | bfloat16 | v3.8.1 | 24 | 0 | 0 | 8 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 24 | 0 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 20 | 0 | 0 | 8 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v2<br>[v3.8.1](tlens_v2.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
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
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 18 | ref | ✅ | ✅ | — | — | — |
| `value`<br>layer 23 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 12 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 18 | ref | ✅ | ✅ | — | — | — |
| `z`<br>layer 23 | ref | ✅ | ✅ | — | — | — |
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
| interp-engine vllm | `final_norm` | — | 0.996389 | 0.964705 | 3 | 0.2635 |
| interp-engine vllm | `mlp_out` | 23 | 0.997968 | 0.951947 | 3 | 0.4005 |
| interp-engine vllm | `mlp_out_post` | 23 | 0.997968 | 0.951947 | 3 | 0.4005 |
| interp-engine vllm | `router_logits` | 23 | 0.997447 | 0.978743 | 3 | 0.2862 |
| interp-engine vllm-static | `mlp_out` | 18 | 0.996552 | 0.963165 | 5 | 0.2698 |
| interp-engine vllm-static | `mlp_out` | 23 | 0.996886 | 0.952187 | 3 | 0.3956 |
| interp-engine vllm-static | `mlp_out_post` | 18 | 0.996552 | 0.963165 | 5 | 0.2698 |
| interp-engine vllm-static | `mlp_out_post` | 23 | 0.996886 | 0.952187 | 3 | 0.3956 |
| interp-engine vllm-static | `router_logits` | 23 | 0.997449 | 0.979033 | 3 | 0.2828 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 0, 12, 18, 23 | nnsight, tlens_v2 | neither engine captured it |
| `mlp_pre` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 12, 18, 23 | nnsight, tlens_v2, interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_act` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
