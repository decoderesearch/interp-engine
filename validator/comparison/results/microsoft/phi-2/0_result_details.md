# `microsoft/phi-2` — cross-engine results

Every engine's capture of `microsoft/phi-2`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 16, 24, 31.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.26.0 | 34 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.27.1 | 32 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | ✅ | ok | float32 | v3.7.0 | 28 | 0 | 0 | 4 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | float32 | v3.7.0 | 28 | 0 | 0 | 4 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 24 | 0 | 0 | 4 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v2<br>[v3.7.0](tlens_v2.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | n/a | — | — | — | — | no ref |
| `resid_mid`<br>layer 16 | n/a | — | — | — | — | no ref |
| `resid_mid`<br>layer 24 | n/a | — | — | — | — | no ref |
| `resid_mid`<br>layer 31 | n/a | — | — | — | — | no ref |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 16 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 24 | ref | ✅ | ✅ | — | — | ✅ |
| `attn_in`<br>layer 31 | ref | ✅ | ✅ | — | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 16 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 24 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 31 | ref | — | — | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 16 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ | ✅ |
| `embeddings` | ref | ✅ | — | — | — | — |
| `final_norm` | ref | ✅ | — | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 16 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 24 | ref | ✅ | ✅ | — | — | — |
| `attn_scores`<br>layer 31 | ref | ✅ | ✅ | — | — | — |

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_out` | 31 | 0.999049 | 0.962386 | 12 | 0.2717 |
| interp-engine vllm | `attn_out_post` | 31 | 0.999049 | 0.962386 | 12 | 0.2717 |
| interp-engine vllm-static | `attn_out` | 31 | 0.999031 | 0.960951 | 12 | 0.2768 |
| interp-engine vllm-static | `attn_out_post` | 31 | 0.999031 | 0.960951 | 12 | 0.2768 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `resid_mid` | 0, 16, 24, 31 | tlens_v2, tlens_v3, interp-engine vllm | neither engine captured it |
| `resid_mid` | 0, 16, 24, 31 | nnsight | the `eager` reference declined the point, so there is nothing to score against — parallel block: attention and the MLP both read the layer input, so no residual exists *between* them. The module a resid_mid would be read from is still there -- GPT-NeoX and phi-2 keep `post_attention_layernorm` -- but it is applied to resid_pre, so an engine that hooks it returns resid_pre under this name: vLLM's came back bit-identical to the embeddings before interp-engine refused the point on that backend too. Read resid_pre or resid_post; nnterp still hands one back |
