# `Qwen/Qwen3-30B-A3B` — cross-engine results

Every engine's capture of `Qwen/Qwen3-30B-A3B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 24, 36, 47.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 62 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 60 | 0 | 0 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: Qwen/Qwen3-30B-A3B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT', 'a… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 32 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 24 | 0 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 20 | 0 | 0 | 8 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 24 | ref | ✅ | ✅† | ✅ | ✅ |
| `mlp_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 24 | ref | ✅ | ✅† | ✅ | — |
| `mlp_out_post`<br>layer 36 | ref | ✅ | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 24 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 36 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 47 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 24 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 36 | ref | ✅ | ✅ | ✅ | — |
| `attn_out_post`<br>layer 47 | ref | ✅ | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 24 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 36 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 47 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 24 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 36 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 47 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 24 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 36 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 47 | n/a | — | — | no ref | — |
| `q_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `q_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `q_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `k_norm_in`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `k_norm_out`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `value`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 47 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 24 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 36 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 47 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_out` | 36 | 0.992526 | 0.972096 | 8 | 0.2364 |
| interp-engine vllm | `attn_out_post` | 36 | 0.992526 | 0.972096 | 8 | 0.2364 |
| interp-engine vllm | `mlp_out` | 24 | 0.982781 | 0.8896 | 8 | 0.4626 |
| interp-engine vllm | `mlp_out` | 36 | 0.981782 | 0.93903 | 8 | 0.345 |
| interp-engine vllm | `mlp_out_post` | 24 | 0.982781 | 0.8896 | 8 | 0.4626 |
| interp-engine vllm | `mlp_out_post` | 36 | 0.981782 | 0.93903 | 8 | 0.345 |
| interp-engine vllm | `value` | 36 | 0.994991 | 0.97001 | 8 | 0.2433 |
| interp-engine vllm-static | `mlp_out` | 24 | 0.978216 | 0.895397 | 8 | 0.4594 |
| interp-engine vllm-static | `mlp_out` | 36 | 0.983187 | 0.947881 | 8 | 0.3223 |
| interp-engine vllm-static | `mlp_out_post` | 24 | 0.978216 | 0.895397 | 8 | 0.4594 |
| interp-engine vllm-static | `mlp_out_post` | 36 | 0.983187 | 0.947881 | 8 | 0.3223 |
| interp-engine vllm-static | `value` | 36 | 0.995293 | 0.97559 | 8 | 0.2219 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Waived passes

| engine | point | layers | worst cos | worst rel diff |
| --- | --- | --- | --- | --- |
| interp-engine vllm-static | `mlp_out` | 24 | 0.978216 | 0.2098 |
| interp-engine vllm-static | `mlp_out_post` | 24 | 0.978216 | 0.2098 |

The waiver: a top-k boundary that bf16 cannot resolve on either side: against a float32 eager run of this checkpoint, eager's own bf16 capture sends 4 of 13 tokens to different experts at layer 24 (vLLM 2) and 6 of 13 at layer 36 (vLLM 6), with mlp_out at cos 0.982/0.975 for eager against 0.987/0.979 for vLLM. The k-th and (k+1)-th router logits at the flipped tokens are 0.016-0.063 apart, one to four bf16 ulps, so which expert wins is decided below the precision either engine is running in

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 0, 24, 36, 47 | nnsight | neither engine captured it |
| `mlp_pre` | 0, 24, 36, 47 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 24, 36, 47 | nnsight, interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_act` | 0, 24, 36, 47 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
