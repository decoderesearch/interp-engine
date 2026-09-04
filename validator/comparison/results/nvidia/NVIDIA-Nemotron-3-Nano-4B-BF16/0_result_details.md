# `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` — cross-engine results

Every engine's capture of `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 21, 31, 41.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 16 | 0 | 0 | 0 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 14 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'a… | bfloat16 | v3.8.1 | 0 | 0 | 0 | 11 |
| [tlens_v3](tlens_v3.json) | ✅ | ok | bfloat16 | v3.8.1 | 5 | 0 | 0 | 6 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 10 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 21 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 31 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 41 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 41 | ref | ✅ | ✅ | n/a | ✅ |
| `mlp_out_post`<br>layer 41 | ref | ✅ | ✅ | n/a | — |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | n/a | ✅ |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | n/a | — |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 41 | ref | — | — | n/a | ✅ |
| `mlp_act`<br>layer 41 | ref | ✅ | ✅ | n/a | ✅ |
| `value`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `z`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 12 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_out` | 41 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_out_post` | 41 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `attn_out` | 12 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `attn_out_post` | 12 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_pre` | 41 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
| `mlp_act` | 41 | tlens_v3 | this engine declined the point — the bridge has no component map for these architectures, so a block bridges to `hook_in`/`hook_out` and nothing else -- checked on the model: `blocks.2` of the LFM2 MoE has no `attn`, `mlp` or `ln` child, and Nemotron-H's has only `norm`/`mixer`. `resid_post` is the one point a block-level hook can serve, and it is the one that scores. Architecture-shaped rather than checkpoint-shaped: `LFM2.5-230M`, the dense model of the same family, is mapped and delivers every point |
