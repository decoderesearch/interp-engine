# `moonshotai/Kimi-K2.6` — cross-engine results

Every engine's capture of `moonshotai/Kimi-K2.6`, point by point, against the `eager` reference on 8x NVIDIA H200. Layers requested: 0, 30, 45, 60.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref* | ok | bfloat16 | v1.8.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.29.0 | 35 | 0 | 0 | 7 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.29.0 | 33 | 0 | 0 | 7 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.8.0](eager.json) | interp-engine vllm<br>[v0.29.0](vllm.json) | interp-engine vllm-static<br>[v0.29.0](vllm-static.json) |
| --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ |
| `resid_post`<br>layer 30 | ref | ✅ | ✅ |
| `resid_post`<br>layer 45 | ref | ✅ | ✅ |
| `resid_post`<br>layer 60 | ref | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ |
| `resid_mid`<br>layer 30 | ref | ✅ | ✅ |
| `resid_mid`<br>layer 45 | ref | ✅ | ✅ |
| `resid_mid`<br>layer 60 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 30 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 45 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 60 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 30 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 45 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 60 | ref | ✅ | ✅ |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ |
| `attn_out`<br>layer 30 | ref | ✅ | ✅ |
| `attn_out`<br>layer 45 | ref | ✅ | ✅ |
| `attn_out`<br>layer 60 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 30 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 45 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 60 | ref | ✅ | ✅ |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ |
| `attn_in`<br>layer 30 | ref | ✅ | ✅ |
| `attn_in`<br>layer 45 | ref | ✅ | ✅ |
| `attn_in`<br>layer 60 | ref | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ |
| `z`<br>layer 0 | ref | ✅ | ✅ |
| `z`<br>layer 30 | ref | ✅ | ✅ |
| `z`<br>layer 45 | ref | ✅ | ✅ |
| `z`<br>layer 60 | ref | ✅ | ✅ |
| `router_logits`<br>layer 30 | n/a | no ref | no ref |
| `router_logits`<br>layer 45 | n/a | no ref | no ref |
| `router_logits`<br>layer 60 | n/a | no ref | no ref |
| `embeddings` | ref | ✅ | — |
| `final_norm` | ref | ✅ | — |
| `attn_scores`<br>layer 0 | ref | n/a | n/a |
| `attn_scores`<br>layer 30 | ref | n/a | n/a |
| `attn_scores`<br>layer 45 | ref | n/a | n/a |
| `attn_scores`<br>layer 60 | ref | n/a | n/a |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 30 | 0.996038 | 0.974864 | 12 | 0.2242 |
| interp-engine vllm | `mlp_out` | 30 | 0.994272 | 0.960792 | 8 | 0.2799 |
| interp-engine vllm | `mlp_out_post` | 30 | 0.994272 | 0.960792 | 8 | 0.2799 |
| interp-engine vllm | `resid_mid` | 30 | 0.999918 | 0.977353 | 12 | 0.2122 |
| interp-engine vllm | `resid_post` | 30 | 0.999917 | 0.978707 | 12 | 0.206 |
| interp-engine vllm-static | `mlp_out` | 45 | 0.992047 | 0.894091 | 5 | 0.4679 |
| interp-engine vllm-static | `mlp_out_post` | 45 | 0.992047 | 0.894091 | 5 | 0.4679 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `router_logits` | 30, 45, 60 | interp-engine vllm, interp-engine vllm-static | the `eager` reference declined the point, so there is nothing to score against |
| `attn_scores` | 0, 30, 45, 60 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by recomputing from captured q/k, and on MLA there is nothing to recompute from |
