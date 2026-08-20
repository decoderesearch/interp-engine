# `deepseek-ai/DeepSeek-V4-Flash-0731` — cross-engine results

Every engine's capture of `deepseek-ai/DeepSeek-V4-Flash-0731`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 21, 32, 42.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ❌ | ok | bfloat16 | v0.27.1 | 38 | 12 | 4 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ❌ | ok | bfloat16 | v0.27.1 | 37 | 11 | 4 | 4 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: MemoryError: deepseek-ai/DeepSeek-V4-Flash-0731 needs ~1084 GiB to load+convert into HookedTransformer (2x bfloat16 weights) but only 234 GiB of host memory is available | bfloat16 | v3.7.2 | 0 | 0 | 0 | 16 |
| [tlens_v3](tlens_v3.json) | unsupported | skip: RuntimeError: Promotion for Float8 Types is not supported, attempted to promote BFloat16 and Float8_e4m3fn | bfloat16 | v3.7.2 | 0 | 0 | 0 | 44 |
| [nnsight](nnsight.json) | unsupported | skip: RenamingError: Could not check the IO of deepseek-ai/DeepSeek-V4-Flash-0731 | bfloat16 | v0.7.0 | 0 | 0 | 0 | 12 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.27.1](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) |
| --- | --- | --- | --- |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 21 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 32 | ref | ✅ | ✅ |
| `mlp_out`<br>layer 42 | ref | ❌ | ❌ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 21 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 32 | ref | ✅ | ✅ |
| `mlp_out_post`<br>layer 42 | ref | ❌ | ❌ |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ |
| `attn_out`<br>layer 21 | ref | ⚠️ | ⚠️ |
| `attn_out`<br>layer 32 | ref | ⚠️ | ⚠️ |
| `attn_out`<br>layer 42 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ |
| `attn_out_post`<br>layer 21 | ref | ⚠️ | ⚠️ |
| `attn_out_post`<br>layer 32 | ref | ⚠️ | ⚠️ |
| `attn_out_post`<br>layer 42 | ref | ✅ | ✅ |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ |
| `attn_in`<br>layer 21 | ref | ⚠️ | ⚠️ |
| `attn_in`<br>layer 32 | ref | ✅ | ✅ |
| `attn_in`<br>layer 42 | ref | ⚠️ | ⚠️ |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ |
| `router_logits`<br>layer 21 | ref | ✅ | ✅ |
| `router_logits`<br>layer 32 | ref | ✅ | ✅ |
| `router_logits`<br>layer 42 | ref | ⚠️ | ⚠️ |
| `embeddings` | ref | ✅ | — |
| `final_norm` | ref | ⚠️ | — |
| `attn_scores`<br>layer 0 | ref | n/a | n/a |
| `attn_scores`<br>layer 21 | ref | n/a | n/a |
| `attn_scores`<br>layer 32 | ref | n/a | n/a |
| `attn_scores`<br>layer 42 | ref | n/a | n/a |
| `resid_streams`<br>layer 0 | ref | ✅ | ✅ |
| `resid_streams`<br>layer 21 | ref | ✅ | ✅ |
| `resid_streams`<br>layer 32 | ref | ✅ | ✅ |
| `resid_streams`<br>layer 42 | ref | ❌ | ❌ |
| `attn_stream_collapse`<br>layer 0 | ref | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 21 | ref | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 32 | ref | ✅ | ✅ |
| `attn_stream_collapse`<br>layer 42 | ref | ⚠️ | ⚠️ |
| `attn_stream_write`<br>layer 0 | ref | ✅ | ✅ |
| `attn_stream_write`<br>layer 21 | ref | ✅ | ✅ |
| `attn_stream_write`<br>layer 32 | ref | ✅ | ✅ |
| `attn_stream_write`<br>layer 42 | ref | ⚠️ | ⚠️ |
| `attn_stream_mix`<br>layer 0 | ref | ✅ | ✅ |
| `attn_stream_mix`<br>layer 21 | ref | ✅ | ✅ |
| `attn_stream_mix`<br>layer 32 | ref | ✅ | ✅ |
| `attn_stream_mix`<br>layer 42 | ref | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 21 | ref | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 32 | ref | ✅ | ✅ |
| `mlp_stream_collapse`<br>layer 42 | ref | ❌ | ❌ |
| `mlp_stream_write`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_stream_write`<br>layer 21 | ref | ✅ | ✅ |
| `mlp_stream_write`<br>layer 32 | ref | ✅ | ✅ |
| `mlp_stream_write`<br>layer 42 | ref | ⚠️ | ⚠️ |
| `mlp_stream_mix`<br>layer 0 | ref | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 21 | ref | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 32 | ref | ✅ | ✅ |
| `mlp_stream_mix`<br>layer 42 | ref | ⚠️ | ⚠️ |

Not in this table: tlens_v2 (unsupported), tlens_v3 (unsupported), nnsight (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 21 | ⚠️ | 0.985353 | 0.1712 | 0.0449 | direction differs (cos 0.985353 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_in` | 42 | ⚠️ | 0.980537 | 0.1964 | 1.2812 | direction differs (cos 0.980537 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 21 | ⚠️ | 0.982529 | 0.188 | 2.3125 | direction differs (cos 0.982529 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out` | 32 | ⚠️ | 0.983397 | 0.1816 | 1.6875 | direction differs (cos 0.983397 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 21 | ⚠️ | 0.982529 | 0.188 | 2.3125 | direction differs (cos 0.982529 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_out_post` | 32 | ⚠️ | 0.983397 | 0.1816 | 1.6875 | direction differs (cos 0.983397 below the `fused` gate of 0.99) |
| interp-engine vllm | `attn_stream_collapse` | 42 | ⚠️ | 0.867778 | 1.0634 | 424 | direction differs (cos 0.867778 below the `fused` gate of 0.99); same direction, different scale (rel 1.0634 above the `fused` gate of 0.5) |
| interp-engine vllm | `attn_stream_write` | 42 | ⚠️ | 0.975604 | 0.2345 | 1.2689 | direction differs (cos 0.975604 below the `fused` gate of 0.99) |
| interp-engine vllm | `final_norm` | — | ⚠️ | 0.954278 | 0.3023 | 8.3281 | direction differs (cos 0.954278 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_out` | 42 | ❌ | 0.235296 | 4.6323 | 1730.6875 | unrelated direction (cos 0.235296, below 0.5) |
| interp-engine vllm | `mlp_out_post` | 42 | ❌ | 0.235296 | 4.6323 | 1730.6875 | unrelated direction (cos 0.235296, below 0.5) |
| interp-engine vllm | `mlp_stream_collapse` | 42 | ❌ | 0.048101 | 30.3183 | 3616.0297 | unrelated direction (cos 0.048101, below 0.5) |
| interp-engine vllm | `mlp_stream_mix` | 42 | ⚠️ | 0.982191 | 0.1887 | 0.6951 | direction differs (cos 0.982191 below the `fused` gate of 0.99) |
| interp-engine vllm | `mlp_stream_write` | 42 | ⚠️ | 0.976308 | 0.2342 | 0.4253 | direction differs (cos 0.976308 below the `fused` gate of 0.99) |
| interp-engine vllm | `resid_streams` | 42 | ❌ | 0.30954 | 10.4249 | 8526.5 | unrelated direction (cos 0.30954, below 0.5) |
| interp-engine vllm | `router_logits` | 42 | ⚠️ | 0.96885 | 0.2559 | 12.8652 | direction differs (cos 0.96885 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 21 | ⚠️ | 0.985406 | 0.1708 | 0.0446 | direction differs (cos 0.985406 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_in` | 42 | ⚠️ | 0.980391 | 0.1971 | 1.2812 | direction differs (cos 0.980391 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 21 | ⚠️ | 0.982651 | 0.1879 | 1.6875 | direction differs (cos 0.982651 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out` | 32 | ⚠️ | 0.9832 | 0.1825 | 1.375 | direction differs (cos 0.9832 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 21 | ⚠️ | 0.982651 | 0.1879 | 1.6875 | direction differs (cos 0.982651 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_out_post` | 32 | ⚠️ | 0.9832 | 0.1825 | 1.375 | direction differs (cos 0.9832 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `attn_stream_collapse` | 42 | ⚠️ | 0.867842 | 1.0634 | 424 | direction differs (cos 0.867842 below the `fused` gate of 0.99); same direction, different scale (rel 1.0634 above the `fused` gate of 0.5) |
| interp-engine vllm-static | `attn_stream_write` | 42 | ⚠️ | 0.975546 | 0.235 | 1.2689 | direction differs (cos 0.975546 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_out` | 42 | ❌ | 0.235934 | 4.6324 | 1730.6875 | unrelated direction (cos 0.235934, below 0.5) |
| interp-engine vllm-static | `mlp_out_post` | 42 | ❌ | 0.235934 | 4.6324 | 1730.6875 | unrelated direction (cos 0.235934, below 0.5) |
| interp-engine vllm-static | `mlp_stream_collapse` | 42 | ❌ | 0.048117 | 30.3184 | 3616.0297 | unrelated direction (cos 0.048117, below 0.5) |
| interp-engine vllm-static | `mlp_stream_mix` | 42 | ⚠️ | 0.982192 | 0.1887 | 0.6951 | direction differs (cos 0.982192 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `mlp_stream_write` | 42 | ⚠️ | 0.976157 | 0.236 | 0.4253 | direction differs (cos 0.976157 below the `fused` gate of 0.99) |
| interp-engine vllm-static | `resid_streams` | 42 | ❌ | 0.309549 | 10.4249 | 8526.5 | unrelated direction (cos 0.309549, below 0.5) |
| interp-engine vllm-static | `router_logits` | 42 | ⚠️ | 0.968915 | 0.2559 | 12.8652 | direction differs (cos 0.968915 below the `fused` gate of 0.99) |

### Agrees on the tensor, not on every token

| engine | point | layer | cos | worst token's cos | which token | its rel diff |
| --- | --- | --- | --- | --- | --- | --- |
| interp-engine vllm | `attn_in` | 32 | 0.994308 | 0.983001 | 5 | 0.1853 |
| interp-engine vllm | `attn_out` | 42 | 0.995285 | 0.950184 | 0 | 0.3209 |
| interp-engine vllm | `attn_out_post` | 42 | 0.995285 | 0.950184 | 0 | 0.3209 |
| interp-engine vllm | `attn_stream_collapse` | 21 | 0.999505 | 0.97164 | 10 | 0.2375 |
| interp-engine vllm | `attn_stream_collapse` | 32 | 0.999862 | 0.978568 | 4 | 0.2328 |
| interp-engine vllm | `mlp_out` | 21 | 0.998499 | 0.912989 | 7 | 0.4163 |
| interp-engine vllm | `mlp_out` | 32 | 0.999954 | 0.938387 | 7 | 0.3515 |
| interp-engine vllm | `mlp_out_post` | 21 | 0.998499 | 0.912989 | 7 | 0.4163 |
| interp-engine vllm | `mlp_out_post` | 32 | 0.999954 | 0.938387 | 7 | 0.3515 |
| interp-engine vllm | `mlp_stream_collapse` | 21 | 0.996687 | 0.976653 | 5 | 0.221 |
| interp-engine vllm | `mlp_stream_collapse` | 32 | 0.999847 | 0.980456 | 1 | 0.1989 |
| interp-engine vllm | `resid_streams` | 21 | 0.999662 | 0.971337 | 10 | 0.2407 |
| interp-engine vllm | `resid_streams` | 32 | 0.995204 | 0.983451 | 6 | 0.1825 |
| interp-engine vllm-static | `attn_in` | 32 | 0.994109 | 0.979303 | 11 | 0.2025 |
| interp-engine vllm-static | `attn_out` | 42 | 0.995344 | 0.950184 | 0 | 0.3209 |
| interp-engine vllm-static | `attn_out_post` | 42 | 0.995344 | 0.950184 | 0 | 0.3209 |
| interp-engine vllm-static | `attn_stream_collapse` | 21 | 0.999517 | 0.972986 | 5 | 0.2357 |
| interp-engine vllm-static | `attn_stream_collapse` | 32 | 0.999861 | 0.977294 | 4 | 0.2326 |
| interp-engine vllm-static | `mlp_out` | 21 | 0.998532 | 0.909757 | 7 | 0.4213 |
| interp-engine vllm-static | `mlp_out` | 32 | 0.999954 | 0.937696 | 7 | 0.3558 |
| interp-engine vllm-static | `mlp_out_post` | 21 | 0.998532 | 0.909757 | 7 | 0.4213 |
| interp-engine vllm-static | `mlp_out_post` | 32 | 0.999954 | 0.937696 | 7 | 0.3558 |
| interp-engine vllm-static | `mlp_stream_collapse` | 21 | 0.996812 | 0.976544 | 5 | 0.223 |
| interp-engine vllm-static | `mlp_stream_collapse` | 32 | 0.999845 | 0.98012 | 4 | 0.2006 |
| interp-engine vllm-static | `resid_streams` | 21 | 0.999668 | 0.971799 | 6 | 0.2393 |
| interp-engine vllm-static | `resid_streams` | 32 | 0.995195 | 0.982996 | 6 | 0.1872 |

These cells pass: the scored cosine is over the whole tensor, and it clears the tier. The column beside it is the same measurement on the single worst token, and it does not -- so a reader who takes one token's activations out of this capture is not getting the agreement the verdict promises. Nothing here is re-scored on that number (the tiers were calibrated against whole-tensor metrics), but a sublayer point that warns while the residual around it passes is usually this, arriving where the massive coordinates are no longer there to average it away.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `attn_scores` | 0, 21, 32, 42 | interp-engine vllm, interp-engine vllm-static | this engine declined the point — multi-head latent attention: the block has no `self_attn.attn` to read q/k off, because the kernel attends over a compressed KV it decompresses internally. vLLM serves `attn_scores` by recomputing from captured q/k, and on MLA there is nothing to recompute from |
