# `google/gemma-4-31B` — cross-engine results

Every engine's capture of `google/gemma-4-31B`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 30, 45, 59.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | bfloat16 | v1.1.0+dirty | — | — | — | — |
| [interp-engine vllm](vllm.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | error: AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute and may vary across layers. Access it via the individual layer configs instead (e.g. config.per_layer_co… | bfloat16 | v0.26.0 | 0 | 0 | 0 | 54 |
| [interp-engine vllm-static](vllm-static.json) | [🐞](https://github.com/vllm-project/vllm/issues/51744) | error: AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute and may vary across layers. Access it via the individual layer configs instead (e.g. config.per_layer_co… | bfloat16 | v0.27.1 | 0 | 0 | 0 | 52 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: ValueError: google/gemma-4-31B not found. Valid official model names (excl aliases): ['01-ai/Yi-34B', '01-ai/Yi-34B-Chat', '01-ai/Yi-6B', '01-ai/Yi-6B-Chat', 'ai-forever/mGPT', 'a… | bfloat16 | v3.7.0 | 0 | 0 | 0 | 36 |
| [tlens_v3](tlens_v3.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1647) | error: AmbiguousGlobalPerLayerAttributeError: 'num_key_value_heads' is a per-layer attribute and may vary across layers. Access it via the individual layer configs instead (e.g. config.p… | bfloat16 | v3.7.0 | 0 | 0 | 0 | 36 |
| [nnsight](nnsight.json) | ✅ | ok | bfloat16 | v0.7.0 | 32 | 0 | 0 | 0 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.1.0+dirty](eager.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ |
| `resid_post`<br>layer 30 | ref | ✅ |
| `resid_post`<br>layer 45 | ref | ✅ |
| `resid_post`<br>layer 59 | ref | ✅ |
| `resid_mid`<br>layer 0 | ref | ✅ |
| `resid_mid`<br>layer 30 | ref | ✅ |
| `resid_mid`<br>layer 45 | ref | ✅ |
| `resid_mid`<br>layer 59 | ref | ✅ |
| `mlp_out`<br>layer 0 | ref | ✅ |
| `mlp_out`<br>layer 30 | ref | ✅ |
| `mlp_out`<br>layer 45 | ref | ✅ |
| `mlp_out`<br>layer 59 | ref | ✅ |
| `attn_out`<br>layer 0 | ref | ✅ |
| `attn_out`<br>layer 30 | ref | ✅ |
| `attn_out`<br>layer 45 | ref | ✅ |
| `attn_out`<br>layer 59 | ref | ✅ |
| `attn_in`<br>layer 0 | ref | ✅ |
| `attn_in`<br>layer 30 | ref | ✅ |
| `attn_in`<br>layer 45 | ref | ✅ |
| `attn_in`<br>layer 59 | ref | ✅ |
| `mlp_pre`<br>layer 0 | ref | ✅ |
| `mlp_pre`<br>layer 30 | ref | ✅ |
| `mlp_pre`<br>layer 45 | ref | ✅ |
| `mlp_pre`<br>layer 59 | ref | ✅ |
| `mlp_pre_linear`<br>layer 0 | ref | ✅ |
| `mlp_pre_linear`<br>layer 30 | ref | ✅ |
| `mlp_pre_linear`<br>layer 45 | ref | ✅ |
| `mlp_pre_linear`<br>layer 59 | ref | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ |
| `mlp_act`<br>layer 30 | ref | ✅ |
| `mlp_act`<br>layer 45 | ref | ✅ |
| `mlp_act`<br>layer 59 | ref | ✅ |

Not in this table: interp-engine vllm (🐞), interp-engine vllm-static (🐞), tlens_v2 (unsupported), tlens_v3 (🐞) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.
