# `ibm-granite/granite-3.0-1b-a400m-base` — cross-engine results

Every engine's capture of `ibm-granite/granite-3.0-1b-a400m-base`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 12, 18, 23.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | bfloat16 | v0.26.0 | 38 | 0 | 0 | 4 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | bfloat16 | v0.27.1 | 36 | 0 | 0 | 0 |
| [tlens_v2](tlens_v2.json) | [unsupported](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | skip: ValueError: ibm-granite/granite-3.0-1b-a400m-base not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'a… | float32 | v3.7.0 | 0 | 0 | 0 | 32 |
| [tlens_v3](tlens_v3.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1648) | ok | float32 | v3.7.0 | 16 | 8 | 0 | 8 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 20 | 0 | 0 | 8 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | interp-engine vllm<br>[v0.26.0](vllm.json) | interp-engine vllm-static<br>[v0.27.1](vllm-static.json) | tlens_v3<br>[v3.7.0](tlens_v3.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 12 | ref | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 18 | ref | ✅ | ✅ | 🐞 | — |
| `mlp_out_post`<br>layer 23 | ref | ✅ | ✅ | 🐞 | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 12 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 18 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 23 | ref | ✅ | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 12 | ref | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 18 | ref | ✅ | ✅ | 🐞 | — |
| `attn_out_post`<br>layer 23 | ref | ✅ | ✅ | 🐞 | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 12 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 18 | ref | ✅ | ✅ | — | ✅ |
| `attn_in`<br>layer 23 | ref | ✅ | ✅ | — | ✅ |
| `mlp_pre`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 12 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 18 | n/a | — | — | no ref | — |
| `mlp_pre`<br>layer 23 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 0 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 12 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 18 | n/a | — | — | no ref | — |
| `mlp_act`<br>layer 23 | n/a | — | — | no ref | — |
| `router_logits`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `router_logits`<br>layer 23 | ref | ✅ | ✅ | — | — |
| `embeddings` | ref | ✅ | — | — | — |
| `final_norm` | ref | ✅ | — | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 12 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 18 | ref | ✅ | ✅ | — | — |
| `attn_scores`<br>layer 23 | ref | ✅ | ✅ | — | — |

Not in this table: tlens_v2 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

| engine | point | layer | verdict | cos | rel diff | max abs diff | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tlens_v3 | `attn_out_post` | 0 | ⚠️ | 1 | 3.5455 | 4.5389 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 12 | ⚠️ | 1 | 3.5455 | 3.0265 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 18 | ⚠️ | 1 | 3.5455 | 3.1208 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `attn_out_post` | 23 | ⚠️ | 1 | 3.5455 | 53.3526 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 0 | ⚠️ | 1 | 3.5455 | 5.4639 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 12 | ⚠️ | 1 | 3.5455 | 19.3012 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 18 | ⚠️ | 1 | 3.5455 | 10.9102 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |
| tlens_v3 | `mlp_out_post` | 23 | ⚠️ | 1 | 3.5455 | 5844.7393 | same direction, different scale (rel 3.5455 above the `tlens` gate of 0.5) |

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `mlp_pre` | 0, 12, 18, 23 | nnsight | neither engine captured it |
| `mlp_pre` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
| `mlp_act` | 0, 12, 18, 23 | nnsight, interp-engine vllm | neither engine captured it |
| `mlp_act` | 0, 12, 18, 23 | tlens_v3 | the `eager` reference declined the point, so there is nothing to score against — sparse block: a routed MLP has no single pre-activation or gate: each expert has its own, and the families that fuse their expert banks have no module boundary at either. eager refuses the point rather than returning one expert's tensor under a whole-layer name; TransformerLens 3's bridge returns the fused bank's, which is a different quantity |
