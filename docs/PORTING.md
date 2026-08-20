# Porting from TransformerLens, nnsight or nnterp

`mappers.py` translates hook names both directions, so porting code off `HookedTransformer` or
`StandardizedTransformer` does not require learning our vocabulary:

```python
from interp_engine import tlens_hook_to_point, point_to_tlens_hook, nnsight_accessor_to_point

tlens_hook_to_point("blocks.7.attn.hook_z")  # ("z", 7)
tlens_hook_to_point("blocks.7.hook_mlp_out", model)  # ("mlp_out_post", 7) on gemma-3
point_to_tlens_hook("attn_probs", 7)  # "blocks.7.attn.hook_pattern"
nnsight_accessor_to_point("mlps_output[7]")  # ("mlp_out", 7)
```

Unmappable names raise `UnmappedHook` (a `ValueError` subclass) listing what is mappable, rather
than guessing. Points that are ours alone raise on the way out: `attn_gate` has no equivalent in
either framework, and `z`/`value`/`attn_probs` have no nnterp accessor because they live inside the
attention module, which nnterp does not standardize.

**`tlens_hook_to_point` takes an optional model, and that argument is the whole point of the
module.** TransformerLens has *two* names for the MLP output and they are different tensors:

| TransformerLens | canonical | what it is |
| --- | --- | --- |
| `blocks.{i}.mlp.hook_out` | `mlp_out` | raw module output, every architecture |
| `blocks.{i}.hook_mlp_out` | `mlp_out` *or* `mlp_out_post` | the **residual contribution** — so post-norm on a sandwich-norm model |

Pass the model (an `EagerModel`, its `.arch`, or `.arch.quirks`) and the block-level hook resolves
the way TransformerLens actually fires it. Omit it and you get pure string translation, which is
correct everywhere except Gemma-2/3/4 and OLMo-2/3 — where it silently hands back a tensor with a
cosine of ~0.2–0.4 against what TransformerLens would have given you. See [Post-sublayer (sandwich)
norms](ARCHITECTURE_QUIRKS.md#post-sublayer-sandwich-norms-the-post_attention_layernorm-trap).

**Gemma Scope's MLP SAEs read the contribution, and this is now measured rather than assumed.** On
`gemma-2-2b` layer 4, `gemmascope-mlp-16k` reconstructs `mlp_out_post` at FVU 0.26 with an L0 of 81
against the SAE's declared 85, and raw `mlp_out` at FVU 9.8 — worse than predicting the mean — with an
L0 of 8. So a source read off `mlp_out` is not merely noisier, it is dead: the layer-4 feature whose
dashboard tops out at 23.5 on `mass-production` fires at no position in that text at all, which is
how the mistake was found rather than by anything raising.

`apps/inference` therefore resolves the block-level hooks to `mlp_out_post` / `attn_out_post` on
every architecture, and does it *without* passing a model — see `engine_adapter.tlens_hook_to_point`.
The model-aware branch is unavailable to it for a structural reason worth knowing if you are in the
same position: `has_sandwich_norms` reads a flag detected on a real module tree, and the vLLM client
holds no modules, since they live in the worker processes. Asking for the `*_post` point needs no
branch and is correct either way, because both resolvers alias it to the raw output where the
architecture has no post-sublayer norm.

If you are adding an SAE source on a sandwich-norm model, still check it against both candidates —
FVU separates them by an order of magnitude in one run.
`apps/inference/tests/integration/test_gemma_mlp_sae_hook.py` is the worked example, negative control
included.

## Three names that read like each other and are not

Same-looking name, different tensor, is the failure mode this module exists to prevent. The mapper
handles all three; these are here because a human or a model writing code by analogy will not.

| you might read | it is | not |
| --- | --- | --- |
| TL `mlp.hook_post` → `mlp_act` | the post-activation **neurons**, `d_mlp` wide, the down projection's input | our `mlp_out_post`, which is the MLP's `d_model` residual contribution after a sandwich norm. The words "post" collide; the tensors differ in width and in position |
| TL `mlp.hook_gate` → our `mlp_pre` (dense) | one expert's **SwiGLU gate branch** | the MoE router. Despite the name, TL's router hooks are `hook_expert_weights` / `hook_expert_indices` |
| TL `mlp.hook_expert_weights` → `softmax(router_logits)` | the softmax over **all** experts, before the top-k | our `expert_weights`, which is the `k` selected weights, renormalized as the checkpoint does it. Both sum to 1, in different lengths |

Coming the other way, our `mlp_pre` / `mlp_pre_linear` map to TL's `hook_pre` / `hook_pre_linear`
exactly — but do not translate them via **weight** names, which cross over: TL's `W_gate` is HF's
`gate_proj`, while TL's `W_in` is HF's `up_proj`, the *other* branch.


[← back to the interp-engine README](../README.md)
