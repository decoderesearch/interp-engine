# `LiquidAI/LFM2-8B-A1B` — cross-engine results

Every engine's capture of `LiquidAI/LFM2-8B-A1B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 2, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 35 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 33 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: LiquidAI/LFM2-8B-A1B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT',… | bfloat16 | v3.7.0 | 0 | 0 | 0 | 27 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.7.0 | 5 | 0 | 0 | 22 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 22 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 2 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 23 | ref | ✅† | ✅† | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | n/a | ✅ |
| `resid_mid`<br>layer 2 | ref | ✅ | ✅ | n/a | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | n/a | ✅ |
| `resid_mid`<br>layer 18 | ref | ✅ | ✅ | n/a | ✅ |
| `resid_mid`<br>layer 23 | ref | ✅† | ✅† | n/a | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | n/a | ✅ |
| `mlp_out`<br>layer 2 | ref | ✅ | ✅ | n/a | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅† | ✅† | n/a | ✅ |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | n/a | ✅ |
| `mlp_out`<br>layer 23 | ref | ✅† | ✅† | n/a | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | n/a | — |
| `mlp_out_post`<br>layer 2 | ref | ✅ | ✅ | n/a | — |
| `mlp_out_post`<br>layer 12 | ref | ✅† | ✅† | n/a | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | n/a | — |
| `mlp_out_post`<br>layer 23 | ref | ✅† | ✅† | n/a | — |
| `attn_out`<br>layer 2 | ref | ✅ | ✅ | n/a | ✅ |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | n/a | ✅ |
| `attn_out_post`<br>layer 2 | ref | ✅ | ✅ | n/a | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | n/a | — |
| `attn_in`<br>layer 2 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | n/a | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | — | — | n/a | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | n/a | ✅ |
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
| interp-engine vllm | `final_norm` | — | 0.950597 | 0.33652 | 6 | 1.1538 |
| interp-engine vllm | `mlp_out` | 12 | 0.989461 | 0.925262 | 13 | 0.38 |
| interp-engine vllm | `mlp_out` | 18 | 0.993934 | 0.985344 | 8 | 0.1708 |
| interp-engine vllm | `mlp_out` | 23 | 0.940806 | 0.09182 | 6 | 1.3816 |
| interp-engine vllm | `mlp_out_post` | 12 | 0.989461 | 0.925262 | 13 | 0.38 |
| interp-engine vllm | `mlp_out_post` | 18 | 0.993934 | 0.985344 | 8 | 0.1708 |
| interp-engine vllm | `mlp_out_post` | 23 | 0.940806 | 0.09182 | 6 | 1.3816 |
| interp-engine vllm | `resid_mid` | 18 | 0.999886 | 0.989999 | 8 | 0.1417 |
| interp-engine vllm | `resid_mid` | 23 | 0.965323 | 0.789398 | 6 | 0.6448 |
| interp-engine vllm | `resid_post` | 12 | 0.999978 | 0.986913 | 13 | 0.1617 |
| interp-engine vllm | `resid_post` | 18 | 0.999851 | 0.989831 | 8 | 0.1428 |
| interp-engine vllm | `resid_post` | 23 | 0.962593 | 0.249955 | 6 | 1.3418 |
| interp-engine vllm-static | `mlp_out` | 12 | 0.989461 | 0.925262 | 13 | 0.38 |
| interp-engine vllm-static | `mlp_out` | 18 | 0.993934 | 0.985344 | 8 | 0.1708 |
| interp-engine vllm-static | `mlp_out` | 23 | 0.940806 | 0.09182 | 6 | 1.3816 |
| interp-engine vllm-static | `mlp_out_post` | 12 | 0.989461 | 0.925262 | 13 | 0.38 |
| interp-engine vllm-static | `mlp_out_post` | 18 | 0.993934 | 0.985344 | 8 | 0.1708 |
| interp-engine vllm-static | `mlp_out_post` | 23 | 0.940806 | 0.09182 | 6 | 1.3816 |
| interp-engine vllm-static | `resid_mid` | 18 | 0.999886 | 0.989999 | 8 | 0.1417 |
| interp-engine vllm-static | `resid_mid` | 23 | 0.965323 | 0.789398 | 6 | 0.6448 |
| interp-engine vllm-static | `resid_post` | 12 | 0.999978 | 0.986913 | 13 | 0.1617 |
| interp-engine vllm-static | `resid_post` | 18 | 0.999851 | 0.989831 | 8 | 0.1428 |
| interp-engine vllm-static | `resid_post` | 23 | 0.962593 | 0.249955 | 6 | 1.3418 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Waived passes

| engine | point | layers | worst cos | worst rel diff |
| --- | --- | --- | --- | --- |
| interp-engine vllm | `resid_post` | 23 | 0.962593 | 0.2749 |
| interp-engine vllm | `resid_mid` | 23 | 0.965323 | 0.263 |
| interp-engine vllm | `mlp_out` | 12, 23 | 0.940806 | 0.3449 |
| interp-engine vllm | `mlp_out_post` | 12, 23 | 0.940806 | 0.3449 |
| interp-engine vllm | `final_norm` | — | 0.950597 | 0.3143 |
| interp-engine vllm-static | `resid_post` | 23 | 0.962593 | 0.2749 |
| interp-engine vllm-static | `resid_mid` | 23 | 0.965323 | 0.263 |
| interp-engine vllm-static | `mlp_out` | 12, 23 | 0.940806 | 0.3449 |
| interp-engine vllm-static | `mlp_out_post` | 12, 23 | 0.940806 | 0.3449 |

The waiver: the reference's own bf16 rounding, not vLLM's: against a float32 eager run of this checkpoint, vLLM's bf16 capture of mlp_out.23 scores cos 0.9989 while eager's own bf16 capture of it scores 0.9277 -- so the ~0.94 between the two engines is almost entirely the reference moving. 24 layers of hybrid trunk compound to ~10% by layer 22, and layer 23's conv contributes a token whose output is as large as the residual it writes into

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `resid_mid` | 0, 2, 12, 18, 23 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_out` | 0, 2, 12, 18, 23 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_out_post` | 0, 2, 12, 18, 23 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `attn_out` | 2, 18 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `attn_out_post` | 2, 18 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_pre` | 0 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_pre_linear` | 0 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_act` | 0 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
