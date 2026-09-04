# `LiquidAI/LFM2-8B-A1B` — cross-engine results

Every engine's capture of `LiquidAI/LFM2-8B-A1B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 2, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 39 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 37 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: LiquidAI/LFM2-8B-A1B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 35 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 27 | 0 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 22 | 0 | 0 | 8 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 2 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 23 | ref | ✅† | ✅† | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 2 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 23 | ref | ✅† | ✅† | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 2 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 23 | ref | ✅† | ✅† | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 2 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 23 | ref | ✅† | ✅† | ✅ | — |
| `attn_out`<br>layer 2 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 2 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 2 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_pre`<br>layer 2 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 12 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 18 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 23 | n/a | — | — | no ref | — |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | ✅ | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 2 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 12 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 18 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 23 | n/a | — | — | no ref | — |
| `value`<br>layer 2 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 2 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 2 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 23 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅† | — | — | — |
| `attn_scores`<br>layer 2 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 18 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `final_norm` | — | 0.942326 | 0.216824 | 6 | 1.2525 |
| interp-engine vllm | `mlp_out` | 12 | 0.98915 | 0.925352 | 13 | 0.3795 |
| interp-engine vllm | `mlp_out` | 23 | 0.926416 | -0.077129 | 6 | 1.537 |
| interp-engine vllm | `mlp_out_post` | 12 | 0.98915 | 0.925352 | 13 | 0.3795 |
| interp-engine vllm | `mlp_out_post` | 23 | 0.926416 | -0.077129 | 6 | 1.537 |
| interp-engine vllm | `resid_mid` | 23 | 0.959727 | 0.75734 | 6 | 0.6948 |
| interp-engine vllm | `resid_post` | 23 | 0.953693 | 0.114693 | 6 | 1.4903 |
| interp-engine vllm-static | `mlp_out` | 12 | 0.98915 | 0.925352 | 13 | 0.3795 |
| interp-engine vllm-static | `mlp_out` | 23 | 0.926416 | -0.077129 | 6 | 1.537 |
| interp-engine vllm-static | `mlp_out_post` | 12 | 0.98915 | 0.925352 | 13 | 0.3795 |
| interp-engine vllm-static | `mlp_out_post` | 23 | 0.926416 | -0.077129 | 6 | 1.537 |
| interp-engine vllm-static | `resid_mid` | 23 | 0.959727 | 0.75734 | 6 | 0.6948 |
| interp-engine vllm-static | `resid_post` | 23 | 0.953693 | 0.114693 | 6 | 1.4903 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Waived passes

| engine | point | layers | worst cos | worst rel diff |
| --- | --- | --- | --- | --- |
| interp-engine vllm | `resid_post` | 23 | 0.953693 | 0.305 |
| interp-engine vllm | `resid_mid` | 23 | 0.959727 | 0.2837 |
| interp-engine vllm | `mlp_out` | 23 | 0.926416 | 0.3835 |
| interp-engine vllm | `mlp_out_post` | 23 | 0.926416 | 0.3835 |
| interp-engine vllm | `final_norm` | — | 0.942326 | 0.3398 |
| interp-engine vllm-static | `resid_post` | 23 | 0.953693 | 0.305 |
| interp-engine vllm-static | `resid_mid` | 23 | 0.959727 | 0.2837 |
| interp-engine vllm-static | `mlp_out` | 23 | 0.926416 | 0.3835 |
| interp-engine vllm-static | `mlp_out_post` | 23 | 0.926416 | 0.3835 |

The waiver: the reference's own bf16 rounding, not vLLM's: against a float32 eager run of this checkpoint, vLLM's bf16 capture of mlp_out.23 scores cos 0.9989 while eager's own bf16 capture of it scores 0.9277 -- so the ~0.94 between the two engines is almost entirely the reference moving. 24 layers of hybrid trunk compound to ~10% by layer 22, and layer 23's conv contributes a token whose output is as large as the residual it writes into

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 2, 12, 18, 23 | nnsight | neither engine captured it |
| `mlp_pre` | 2, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 2, 12, 18, 23 | nnsight, interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_act` | 2, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
