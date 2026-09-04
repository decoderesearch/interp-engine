# `deepseek-ai/DeepSeek-V4-Flash-0731` — cross-engine results

Every engine's capture of `deepseek-ai/DeepSeek-V4-Flash-0731`, point by point, against the `eager` reference on NVIDIA GeForce RTX 5090. Layers requested: 0, 21, 32, 42.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.28.0 | 54 | 0 | 0 | 12 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.28.0 | 52 | 0 | 0 | 12 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: MemoryError: deepseek-ai/DeepSeek-V4-Flash-0731 needs ~1084 GiB to load+convert into HookedTransformer (2x bfloat16 weights) but only 351 GiB of host memory is available | bfloat16 | v3.8.1 | 0 | 0 | 0 | 16 |
| [tlens_v3](tlens_v3.json) | ⚠️ | ok | bfloat16 | v3.8.1 | 36 | 0 | 0 | 16 |
| [nnsight](nnsight.json) | unsupported | skip: RenamingError: Could not check the IO of deepseek-ai/DeepSeek-V4-Flash-0731 | bfloat16 | v0.7.0 | 0 | 0 | 0 | 20 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | tlens_v3<br>[v3.8.1](tlens_v3.json) |
| --- | --- | --- | --- | --- |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | n/a |
| `mlp_out_post`<br>layer 21 | ref | ✅ | ✅ | n/a |
| `mlp_out_post`<br>layer 32 | ref | ✅ | ✅ | n/a |
| `mlp_out_post`<br>layer 42 | ref | ✅ | ✅ | n/a |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | n/a |
| `attn_out_post`<br>layer 21 | ref | ✅ | ✅ | n/a |
| `attn_out_post`<br>layer 32 | ref | ✅ | ✅ | n/a |
| `attn_out_post`<br>layer 42 | ref | ✅ | ✅ | n/a |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — |
| `attn_in`<br>layer 21 | ref | ✅ | ✅ | — |
| `attn_in`<br>layer 32 | ref | ✅ | ✅ | — |
| `attn_in`<br>layer 42 | ref | ✅ | ✅ | — |
| `mlp_pre`<br>layer 0 | n/a | — | — | no ref |
| `mlp_pre`<br>layer 21 | n/a | — | — | no ref |
| `mlp_pre`<br>layer 32 | n/a | — | — | no ref |
| `mlp_pre`<br>layer 42 | n/a | — | — | no ref |
| `mlp_act`<br>layer 0 | n/a | — | — | no ref |
| `mlp_act`<br>layer 21 | n/a | — | — | no ref |
| `mlp_act`<br>layer 32 | n/a | — | — | no ref |
| `mlp_act`<br>layer 42 | n/a | — | — | no ref |
| `z`<br>layer 0 | ref | n/a | n/a | — |
| `z`<br>layer 21 | ref | n/a | n/a | — |
| `z`<br>layer 32 | ref | n/a | n/a | — |
| `z`<br>layer 42 | ref | n/a | n/a | — |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ | — |
| `router_logits`<br>layer 21 | ref | ✅ | ✅ | — |
| `router_logits`<br>layer 32 | ref | ✅ | ✅ | — |
| `router_logits`<br>layer 42 | ref | ✅ | ✅ | — |
| `embeddings` | ref | ✅ | — | — |
| `final_norm` | ref | ✅ | — | — |
| `attn_scores`<br>layer 0 | ref | n/a | n/a | — |
| `attn_scores`<br>layer 21 | ref | n/a | n/a | — |
| `attn_scores`<br>layer 32 | ref | n/a | n/a | — |
| `attn_scores`<br>layer 42 | ref | n/a | n/a | — |
| `resid_streams`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `resid_streams`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `resid_streams`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `resid_streams`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `attn_stream_write`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_stream_write`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `attn_stream_write`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `attn_stream_write`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `attn_stream_mix`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_stream_mix`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `attn_stream_mix`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `attn_stream_mix`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_write`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_write`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_write`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_write`<br>layer 42 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 21 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 32 | ref | ✅ | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 42 | ref | ✅ | ✅ | ✅ |

Not in this table: tlens_v2 (unsupported), nnsight (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 21 | 0.982153 | 0.95328 | 8 | 0.3056 |
| interp-engine vllm | `attn_in` | 32 | 0.992504 | 0.967275 | 12 | 0.2799 |
| interp-engine vllm | `attn_out` | 21 | 0.981444 | 0.961795 | 8 | 0.2764 |
| interp-engine vllm | `attn_out` | 32 | 0.984936 | 0.977401 | 6 | 0.2115 |
| interp-engine vllm | `attn_out_post` | 21 | 0.981444 | 0.961795 | 8 | 0.2764 |
| interp-engine vllm | `attn_out_post` | 32 | 0.984936 | 0.977401 | 6 | 0.2115 |
| interp-engine vllm | `attn_stream_collapse` | 21 | 0.999454 | 0.95354 | 8 | 0.3051 |
| interp-engine vllm | `attn_stream_collapse` | 32 | 0.999902 | 0.976456 | 8 | 0.223 |
| interp-engine vllm | `mlp_out` | 21 | 0.998625 | 0.914863 | 10 | 0.4143 |
| interp-engine vllm | `mlp_out` | 32 | 0.999971 | 0.952761 | 8 | 0.3057 |
| interp-engine vllm | `mlp_out_post` | 21 | 0.998625 | 0.914863 | 10 | 0.4143 |
| interp-engine vllm | `mlp_out_post` | 32 | 0.999971 | 0.952761 | 8 | 0.3057 |
| interp-engine vllm | `mlp_stream_collapse` | 21 | 0.996458 | 0.966989 | 8 | 0.2596 |
| interp-engine vllm | `mlp_stream_collapse` | 32 | 0.999893 | 0.973511 | 8 | 0.229 |
| interp-engine vllm | `mlp_stream_collapse` | 42 | 0.998648 | 0.977368 | 11 | 0.2118 |
| interp-engine vllm | `resid_streams` | 21 | 0.999641 | 0.952559 | 10 | 0.3101 |
| interp-engine vllm | `resid_streams` | 32 | 0.999675 | 0.977832 | 8 | 0.2105 |
| interp-engine vllm-static | `attn_in` | 21 | 0.983188 | 0.961573 | 10 | 0.2772 |
| interp-engine vllm-static | `attn_out` | 21 | 0.981354 | 0.965716 | 10 | 0.2596 |
| interp-engine vllm-static | `attn_out` | 32 | 0.985323 | 0.978024 | 6 | 0.2089 |
| interp-engine vllm-static | `attn_out_post` | 21 | 0.981354 | 0.965716 | 10 | 0.2596 |
| interp-engine vllm-static | `attn_out_post` | 32 | 0.985323 | 0.978024 | 6 | 0.2089 |
| interp-engine vllm-static | `attn_stream_collapse` | 21 | 0.999442 | 0.961538 | 10 | 0.2774 |
| interp-engine vllm-static | `attn_stream_collapse` | 32 | 0.999903 | 0.978651 | 8 | 0.2124 |
| interp-engine vllm-static | `mlp_out` | 21 | 0.998618 | 0.917802 | 10 | 0.4081 |
| interp-engine vllm-static | `mlp_out` | 32 | 0.999971 | 0.958274 | 7 | 0.2859 |
| interp-engine vllm-static | `mlp_out_post` | 21 | 0.998618 | 0.917802 | 10 | 0.4081 |
| interp-engine vllm-static | `mlp_out_post` | 32 | 0.999971 | 0.958274 | 7 | 0.2859 |
| interp-engine vllm-static | `mlp_stream_collapse` | 21 | 0.996249 | 0.970956 | 10 | 0.2425 |
| interp-engine vllm-static | `mlp_stream_collapse` | 32 | 0.999898 | 0.977785 | 8 | 0.2096 |
| interp-engine vllm-static | `mlp_stream_collapse` | 42 | 0.99865 | 0.97805 | 11 | 0.2094 |
| interp-engine vllm-static | `resid_streams` | 21 | 0.999652 | 0.955635 | 10 | 0.2991 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_out_post` | 0, 21, 32, 42 | tlens_v3 | this engine declined the point |
| `attn_out_post` | 0, 21, 32, 42 | tlens_v3 | this engine declined the point |
| `mlp_pre` | 0, 21, 32, 42 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 21, 32, 42 | interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `mlp_act` | 0, 21, 32, 42 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `z` | 0, 21, 32, 42 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — the output projection is factored into `wo_a`/`wo_b` and computed inside a fused `_o_proj` kernel that is handed `wo_a` rather than calling it, on all four of vLLM's platform implementations (FlashMLA, FlashInfer, ROCm, XPU). So the module whose input `z` names is never called and a hook on it never fires. interp-engine refuses the point (`vllm_capture._tree.absent_point_reason`) rather than resolving it: when it did resolve, `vllm-static` returned a full-looking (13, 4096) of exact zeros -- its buffer, never written -- against a real (13, 8, 4096) eager reference. Eager serves `z` on the same checkpoint, where the same pair is called as ordinary modules |
| `attn_scores` | 0, 21, 32, 42 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by recomputing from captured q/k, and on MLA there is nothing to recompute from |
