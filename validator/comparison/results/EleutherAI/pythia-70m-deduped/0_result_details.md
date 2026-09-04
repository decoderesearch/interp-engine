# `EleutherAI/pythia-70m-deduped` — cross-engine results

Every engine's capture of `EleutherAI/pythia-70m-deduped`, point by point, against the `eager` reference on NVIDIA B200. Layers requested: 0, 3, 5.

Generated from the `<engine>.json` files beside this one, which hold the same numbers with nothing rolled up; the summary table is in [the README](../../../../README.md).

### Engines

| engine | verdict | capture | dtype | version | agreed | differs | failed | not compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [interp-engine eager](eager.json) *(reference)* | ref | ok | float32 | v1.6.0 | — | — | — | — |
| [interp-engine vllm](vllm.json) | ✅ | ok | float32 | v0.28.0 | 32 | 0 | 0 | 3 |
| [interp-engine vllm-static](vllm-static.json) | ✅ | ok | float32 | v0.28.0 | 30 | 0 | 0 | 3 |
| [tlens_v2](tlens_v2.json) | unsupported | skip: AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out' | float32 | v3.8.1 | 0 | 0 | 0 | 24 |
| [tlens_v3](tlens_v3.json) | unsupported | skip: AttributeError: 'GPTNeoXForCausalLM' object has no attribute 'embed_out' | float32 | v3.8.1 | 0 | 0 | 0 | 24 |
| [nnsight](nnsight.json) | ✅ | ok | float32 | v0.7.0 | 18 | 0 | 0 | 3 |

### Point by point

✅ agrees · ⚠️ differs in value · ❌ structurally wrong, or the engine did not deliver it · 🐞 differs because the reference is wrong here, with an issue filed against it (`ref🐞` marks the reference's own column) · `ref` the reference produced this point (nothing scores it — it *is* the baseline) · `n/a` this engine declines the point · `no ref` the reference declined it · `—` no comparison here — the point is not asked of this engine, or it is listed under *Not compared* · † a waiver carried the pass (listed below)

| point<br>layer | interp-engine eager<br>[v1.6.0](eager.json) | interp-engine vllm<br>[v0.28.0](vllm.json) | interp-engine vllm-static<br>[v0.28.0](vllm-static.json) | nnsight<br>[v0.7.0](nnsight.json) |
| --- | --- | --- | --- | --- |
| `resid_post`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 3 | ref | ✅ | ✅ | ✅ |
| `resid_post`<br>layer 5 | ref | ✅ | ✅ | ✅ |
| `resid_mid`<br>layer 0 | n/a | — | — | no ref |
| `resid_mid`<br>layer 3 | n/a | — | — | no ref |
| `resid_mid`<br>layer 5 | n/a | — | — | no ref |
| `mlp_out`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 3 | ref | ✅ | ✅ | ✅ |
| `mlp_out`<br>layer 5 | ref | ✅ | ✅ | ✅ |
| `mlp_out_post`<br>layer 0 | ref | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 3 | ref | ✅ | ✅ | — |
| `mlp_out_post`<br>layer 5 | ref | ✅ | ✅ | — |
| `attn_out`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 3 | ref | ✅ | ✅ | ✅ |
| `attn_out`<br>layer 5 | ref | ✅ | ✅ | ✅ |
| `attn_out_post`<br>layer 0 | ref | ✅ | ✅ | — |
| `attn_out_post`<br>layer 3 | ref | ✅ | ✅ | — |
| `attn_out_post`<br>layer 5 | ref | ✅ | ✅ | — |
| `attn_in`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `attn_in`<br>layer 3 | ref | ✅ | ✅ | ✅ |
| `attn_in`<br>layer 5 | ref | ✅ | ✅ | ✅ |
| `mlp_pre`<br>layer 0 | ref | — | — | ✅ |
| `mlp_pre`<br>layer 3 | ref | — | — | ✅ |
| `mlp_pre`<br>layer 5 | ref | — | — | ✅ |
| `mlp_act`<br>layer 0 | ref | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 3 | ref | ✅ | ✅ | ✅ |
| `mlp_act`<br>layer 5 | ref | ✅ | ✅ | ✅ |
| `value`<br>layer 0 | ref | ✅ | ✅ | — |
| `value`<br>layer 3 | ref | ✅ | ✅ | — |
| `value`<br>layer 5 | ref | ✅ | ✅ | — |
| `z`<br>layer 0 | ref | ✅ | ✅ | — |
| `z`<br>layer 3 | ref | ✅ | ✅ | — |
| `z`<br>layer 5 | ref | ✅ | ✅ | — |
| `embeddings` | ref | ✅ | — | — |
| `final_norm` | ref | ✅ | — | — |
| `attn_scores`<br>layer 0 | ref | ✅ | ✅ | — |
| `attn_scores`<br>layer 3 | ref | ✅ | ✅ | — |
| `attn_scores`<br>layer 5 | ref | ✅ | ✅ | — |

Not in this table: tlens_v2 (unsupported), tlens_v3 (unsupported) — captured nothing for this checkpoint, for the reason in the table above.

### What differs

Nothing: every point every engine captured agreed with the reference.

### Not compared

| point | layers | engines | why |
| --- | --- | --- | --- |
| `resid_mid` | 0, 3, 5 | interp-engine vllm, interp-engine vllm-static | neither engine captured it |
| `resid_mid` | 0, 3, 5 | nnsight | the `eager` reference declined the point, so there is nothing to score against — parallel block: attention and the MLP both read the layer input, so no residual exists *between* them. The module a resid_mid would be read from is still there -- GPT-NeoX and phi-2 keep `post_attention_layernorm` -- but it is applied to resid_pre, so an engine that hooks it returns resid_pre under this name: vLLM's came back bit-identical to the embeddings before interp-engine refused the point on that backend too. Read resid_pre or resid_post; nnterp still hands one back |
