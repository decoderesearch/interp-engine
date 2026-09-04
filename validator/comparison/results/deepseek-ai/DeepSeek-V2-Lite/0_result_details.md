# `deepseek-ai/DeepSeek-V2-Lite` — cross-engine results

Every engine's capture of `deepseek-ai/DeepSeek-V2-Lite`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 13, 20, 26.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 38 | 0 | 0 | 7 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 36 | 0 | 0 | 7 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: deepseek-ai/DeepSeek-V2-Lite not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 33 |
| [tlens_v3](tlens_v3.json) | [✅](https://github.com/TransformerLensOrg/TransformerLens/issues/1645) | ok | bfloat16 | v3.8.1 | 27 | 0 | 0 | 6 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 23 | 0 | 0 | 6 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 13 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 20 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 26 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 13 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 20 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 26 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 13 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 20 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 26 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 13 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 26 | n/a | — | — | no ref | — |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 13 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 20 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 26 | n/a | — | — | no ref | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 13 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 13 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 20 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 26 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 13 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 20 | ref | n/a | n/a | — | — |
| `attn_scores`<br>layer 26 | ref | n/a | n/a | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 13, 20, 26 | nnsight | neither engine captured it |
| `mlp_pre` | 13, 20, 26 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 13, 20, 26 | nnsight, interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_act` | 13, 20, 26 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `attn_scores` | 0, 13, 20, 26 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by recomputing from captured q/k, and on MLA there is nothing to recompute from |
