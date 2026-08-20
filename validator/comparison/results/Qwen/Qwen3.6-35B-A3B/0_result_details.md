# `Qwen/Qwen3.6-35B-A3B` — cross-engine results

Every engine's capture of `Qwen/Qwen3.6-35B-A3B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 20, 30, 39.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 30 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ⚠️ | ok | bfloat16 | v0.27.1 | 24 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: Qwen/Qwen3.6-35B-A3B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | bfloat16 | v3.7.0 | 0 | 0 | 0 | 26 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 18 | 0 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 14 | 0 | 0 | 8 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 30 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 30 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 39 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 39 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 39 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 30 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 39 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 30 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 39 | n/a | — | — | no ref | — |
| `q_norm_in`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 39 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 0 | ref | ✅ | n/a | — | — |
| `router_logits`<br>layer 20 | ref | ✅ | n/a | — | — |
| `router_logits`<br>layer 30 | ref | ✅ | n/a | — | — |
| `router_logits`<br>layer 39 | ref | ✅ | n/a | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 39 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `mlp_out` | 20 | 0.992263 | 0.940336 | 1 | 0.3626 |
| interp-engine vllm | `mlp_out` | 30 | 0.99617 | 0.980566 | 9 | 0.1962 |
| interp-engine vllm | `mlp_out_post` | 20 | 0.992263 | 0.940336 | 1 | 0.3626 |
| interp-engine vllm | `mlp_out_post` | 30 | 0.99617 | 0.980566 | 9 | 0.1962 |
| interp-engine vllm-static | `mlp_out` | 20 | 0.995474 | 0.947781 | 1 | 0.3267 |
| interp-engine vllm-static | `mlp_out` | 30 | 0.997464 | 0.975221 | 12 | 0.2283 |
| interp-engine vllm-static | `mlp_out_post` | 20 | 0.995474 | 0.947781 | 1 | 0.3267 |
| interp-engine vllm-static | `mlp_out_post` | 30 | 0.997464 | 0.975221 | 12 | 0.2283 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 0, 20, 30, 39 | nnsight | neither engine captured it |
| `mlp_pre` | 0, 20, 30, 39 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 20, 30, 39 | nnsight, interp-engine vllm | neither engine captured it |
| `mlp_act` | 0, 20, 30, 39 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `router_logits` | 0, 20, 30, 39 | interp-engine vllm-static | this engine declined the point |
