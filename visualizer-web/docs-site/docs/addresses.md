---
id: addresses
title: Points and addresses
sidebar_position: 2
---

# Points and addresses

A **point** is a named tensor in the forward pass. An **address** is a point plus the coordinates
that pick out one of them, and it is what every capture and steering call takes.

## Three coordinates

```python
from interp_engine import Address

Address("resid_post", 10)                # name and layer
Address("embeddings")                    # trunk-level, so no layer
Address("resid_streams", 5, 2)           # name, layer, stream
Address("model.layers.5.mlp.down_proj")  # a module path, for what the table below does not name
```

**`name`** is a point from the table below, or a dotted module path. The point set is open: an
unrecognized name with no layer is looked up as a module, which is how you tap something the engine
does not enumerate.

**`layer`** is position in flattened forward order — one total order over sublayer executions in a
single forward pass, rather than a block index plus a sublayer index. So no tensor has two
spellings.

**`stream`** is which parallel residual stream, on a hyper-connection trunk. It is the only
coordinate that is a real tensor axis: all of them exist at once in a `[batch, pos, streams, d_model]`
activation, so nothing can flatten them the way execution order flattens.

The field set is closed, so no family can invent a coordinate of its own.

## The string form

Every address round-trips through one string, and a `Cache` accepts either form on lookup.

```python
from interp_engine import Address, format_address, parse_address, to_address

format_address(Address("resid_post", 5, 2))  # "resid_post.5.stream-2"
parse_address("mlp_act.10")
to_address(("z", 7))                         # the older tuple form still reads
```

Parsing is strict, and that is the useful part. A coordinate selects a tensor rather than annotating
one, so a parser that skipped `stream-2` in `resid_post.5.stream-2` would hand back `resid_post.5` —
a real tensor of the right shape from the wrong place. An unrecognized coordinate raises
`UnknownCoordinate` instead, which a caller can catch to report version skew specifically. The
legacy `"resid_post:5"` spelling raises too.

Only unreserved URL characters are emitted, so an address needs no escaping in a query string, no
quoting in a shell, and is a legal filename everywhere.

## The 34 points

27 on every model, and 7 more that need a hyper-connection trunk. Width is what the last axis
counts.

| point                       | width                   | what it is                                                        |
| --------------------------- | ----------------------- | ----------------------------------------------------------------- |
| `embeddings`                | `d_model`               | the embedding module's output. Trunk-level, so no layer           |
| `resid_pre`                 | `d_model`               | the residual entering the block                                   |
| `attn_in`                   | `d_model`               | what the attention module was handed                              |
| `q_norm_in` / `q_norm_out`  | `n_heads * head_dim`    | the query on either side of QK-norm                               |
| `k_norm_in` / `k_norm_out`  | `n_kv_heads * head_dim` | the key on either side of QK-norm                                 |
| `value`                     | `n_heads * head_dim`    | the value projection                                              |
| `attn_scores`               | `n_heads * query * key` | the QK matrix, before the softmax                                 |
| `attn_probs`                | `n_heads * query * key` | the attention pattern, after it                                   |
| `z`                         | `n_heads * head_dim`    | per-head attention output, before `W_O`                           |
| `attn_gate`                 | `n_heads * head_dim`    | the raw double-width projection, on a gated-attention model       |
| `attn_out`                  | `d_model`               | attention's raw module output                                     |
| `attn_out_post`             | `d_model`               | attention's residual contribution                                 |
| `resid_mid`                 | `d_model`               | the input to the pre-MLP norm                                     |
| `mlp_in`                    | `d_model`               | what the MLP was handed                                           |
| `mlp_pre`                   | `d_mlp`                 | pre-activation neurons                                            |
| `mlp_pre_linear`            | `d_mlp`                 | the ungated branch, on gated MLPs                                 |
| `mlp_act`                   | `d_mlp`                 | post-activation neurons                                           |
| `router_logits`             | `n_experts`             | the MoE router's output                                           |
| `expert_weights`            | `n_experts`             | the top-k gate weights, in the router's ranking order             |
| `expert_indices`            | `n_experts`             | which experts were picked; the one integer-valued point           |
| `mlp_out`                   | `d_model`               | the MLP's raw module output                                       |
| `mlp_out_post`              | `d_model`               | the MLP's residual contribution                                   |
| `resid_post`                | `d_model`               | the residual leaving the block                                    |
| `final_norm`                | `d_model`               | the trunk's last norm. Trunk-level                                |
| `lm_head`                   | `vocab_size`            | the bare unembed. Trunk-level                                     |

**Reach for the `_post` point when you want a sublayer's contribution to the residual.** It is the
same tensor as the raw output on most families and differs on a sandwich-norm one (Gemma-2/3,
OLMo-2/3), where the raw output has not been through the post-sublayer norm yet. `attn_out_post`
and `mlp_out_post` alias the raw points wherever that norm does not exist, so asking for the
contribution is safe everywhere — and this is the mapping mistake that produces a plausible-looking
wrong number rather than an error.

### Only on a hyper-connection trunk

DeepSeek-V4 and Motif 3 today, and any family whose config reports more than one residual stream.
These are gated on that count rather than on an architecture name.

| point                  | width                  | what it is                                                     |
| ---------------------- | ---------------------- | -------------------------------------------------------------- |
| `resid_streams`        | `n_residual_streams`   | the block's own output stack; stream _k_ is `stack[:, k]`      |
| `attn_stream_collapse` | `d_model`              | the one `d_model` vector attention reads                       |
| `mlp_stream_collapse`  | `d_model`              | the same for the FFN, one norm before `mlp_in`                 |
| `attn_stream_write`    | `n_residual_streams`   | the per-stream weights attention's output is written back with |
| `attn_stream_mix`      | `n_residual_streams`   | the doubly-stochastic matrix that remixes the streams after it |
| `mlp_stream_write`     | `n_residual_streams`   | the MLP's counterpart                                          |
| `mlp_stream_mix`       | `n_residual_streams`   | the MLP's mixing matrix                                        |

The two collapse points are what an SAE or a steering vector wants on such a trunk, since they are
the vectors a sublayer actually reads. The four write and mix rows are coefficients rather than
activations, so they capture but refuse a steer.

Note that `resid_streams` and the `stream` coordinate say different things. `resid_streams.5` is
the whole stack, `[tokens, streams, d_model]`; `resid_post.5.stream-2` qualifies a residual point
with one stream and is eager-only.

## Which backend serves which

This page is the vocabulary, not the support matrix — every point above is capturable eagerly, and
the vLLM backend serves most but not all of them. Ask the model rather than a table if you are
branching on it, with `model.points()`; see [Capabilities](./capabilities.md). The full per-backend
breakdown, with a reason for each refusal, is in
[SUPPORTED_POINTS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/SUPPORTED_POINTS.md),
and the <a href="/">visualizer</a> shows where each point sits in the forward pass.
