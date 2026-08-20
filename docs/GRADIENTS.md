# Gradients

Two rules, and they are the whole design.

**Gradient support never gates loading.** Someone switching from vLLM to `VLLMModel` loads
successfully on any kernel, any quantization scheme, any cudagraph mode, and finds out about
gradients only if they ask. Nothing in `autograd_support.py` is consulted during construction: the
verdict is computed lazily on first read of `model.grad_support`, from configuration alone — it
builds no engine, loads no kernel, and runs no forward.

**And nothing degrades silently.** Asking for gradients where they are unavailable raises
`GradientsUnsupported` naming the specific blockers. It never quietly hands back detached tensors,
and it never quietly flips to a slower kernel to make gradients possible.

### The two kinds of support are not the same question

| | what it means | eager | vLLM |
| --- | --- | --- | --- |
| `downstream` | the captured tensor works in an autograd graph *you* build — fit a probe on it, backprop into a decoder | yes | **yes** |
| `through_forward` | gradients flow back through the model's own forward, to its weights or inputs | with `requires_grad=True` | **never** |

So a linear probe on captured activations trains fine on either backend. Attribution that needs a
gradient *of the model* is eager-only.

### Getting gradients (eager)

```python
model = EagerModel("openai-community/gpt2", requires_grad=True)  # default is False
cache = run_with_cache(model, input_ids, [("resid_post", 5)], detach=False)
cache.get("resid_post", 5).sum().backward()
model.arch.embed.weight.grad  # populated: the graph reached the embedding
```

`requires_grad` defaults to `False` because serving is forward-only and a live graph costs
activation memory on every capture. The async `capture()` takes the same `detach=False`, and returns
device tensors rather than CPU ones in that case (moving them is a graph node nobody asked for).

`detach=False` on a **frozen** model raises rather than returning graph-free tensors. It cannot do
anything useful: the inputs are token ids, so with frozen parameters there is nothing to
differentiate with respect to, and the silent version of this is someone debugging all-zero
gradients for an afternoon.

### float16, and why one architecture is refused outright

A backward only amplifies a marginal forward, so `float16` is where a differentiable model quietly
stops being useful. The verdict gives two different answers on purpose:

- **On an architecture in `facts.FP16_EAGER_OVERFLOW_ARCHS` it is a blocker**, and
  `EagerModel(..., requires_grad=True)` refuses at construction. `pythia-70m-deduped` in fp16 already
  NaNs from layer 3 in transformers' GPT-NeoX eager attention kernel, so its gradients would be NaN
  too — and a NaN gradient is worse than a missing one, because it looks like a result. The error
  names `dtype='float32'` as the fix. A *plain* fp16 load of the same model is untouched: gradient
  support never gates a load that did not ask for gradients.
- **Anywhere else it is a caveat, not a refusal** — logged once at construction and reported in
  `grad_support.caveats`. fp16 gradients are legitimate on a healthy model, and refusing them would be
  the tail wagging the dog.

Worth knowing that `EagerModel.__init__` defaults to `float32` while `from_pretrained` defaults to
`auto` (the checkpoint's native precision), so the entry point you pick decides this for you. The
verdict reads the *loaded* dtype, so `auto` is judged by what the checkpoint turned out to be.

### Quantization: capture does not care, the backward does

Hooks read activations, which `transformers` dequantizes to a compute dtype at module boundaries, so
every point the engine serves is quantization-agnostic down to 4-bit. The backward is a different
question, and it splits by **implementation rather than bit width** — the same two-answer shape as
float16, from the two tables in `facts.py`:

- **A fused forward-only kernel is a blocker** (`facts.FORWARD_ONLY_QUANT_METHODS`: AWQ, GPTQ, FP8 in
  its spellings, native MXFP4). Their dequantize-matmul is registered without an autograd formula, so
  a backward through it raises from inside the op. Refusing up front turns that into a sentence naming
  the scheme, and the remedy says the part people get wrong: capture is unaffected, only the backward
  is.
- **bitsandbytes is a caveat** (`facts.DIFFERENTIABLE_QUANT_METHODS`, 4-bit and 8-bit). Its matmuls
  route through autograd Functions with a real backward, which is why an offline lens fit works on
  these. Qualified twice: the gradient is with respect to activations only — the quantized weights are
  frozen and receive nothing — and dequantization adds noise, so it is not the gradient the
  unquantized model would give.
- **An unrecognised scheme is also a caveat**, not a blocker, and this asymmetry is deliberate. A
  wrongly-listed scheme refuses a request that would have worked and the caller cannot override it;
  an unlisted one fails inside the kernel with a caveat already saying why. `torchao` and `quanto` are
  the reason the tables are short: both support quantization-aware training on some paths, so neither
  belongs in either table without being measured.

The scheme is read from the *loaded* config (`EagerModel.quant_method`), so a pre-quantized checkpoint
and a quantize-on-load are judged the same way. See
[PERFORMANCE.md](PERFORMANCE.md#quantization-support) for which install extra each scheme needs.

### Three more things that will bite you, which are documented rather than guarded

The gradient path itself is architecture-agnostic — one generic hook and one context-manager flip, no
per-family code — but these are not:

- **MoE changes what the gradient means.** Top-k expert routing is a discrete choice: gradient flows
  through the selected experts and the router's gate weights, never through the selection. Which
  layers are affected is reported in `quirks.moe_layers`.
- **On sandwich-norm models the capture point silently picks your attribution target**, because
  `mlp_out` and `mlp_out_post` are different graph nodes there (Gemma-2/3/4, OLMo-2/3).

### Generation is not differentiable, and that is deliberate

`steer.py`'s generation loop is unconditionally `no_grad` with no `detach` escape hatch. A tape over
`max_tokens` sequential forwards retains every step's activations at once, so memory grows with
the generation length and a few hundred tokens will OOM a card that generates the same text fine.
Differentiating a rollout is a real thing to want, but it needs a purpose-built path (a fixed short
rollout, or gradient checkpointing) rather than a flag here.

The lens read-out *is* differentiable, via `decode_residuals(..., detach=False)`:

```python
resid = torch.randn(n, model.d_model, requires_grad=True)
decode_residuals(model, resid, detach=False).sum().backward()
resid.grad  # populated
```

That works on a **frozen** model and deliberately does not consult `grad_support`, because the useful
gradient here is with respect to the residual you passed in — optimizing a steering vector against a
logit objective, say — and it never needs to reach a parameter. `layer_logits` has no such flag on
purpose: it is the serving read-out path, where a graph is pure overhead.

### Why vLLM can never differentiate through its forward

Not a kernel problem, and not fixable by configuration. vLLM decorates `GPUModelRunner.execute_model`
(and `Worker.execute_model`) with `@torch.inference_mode()`, which is strictly stronger than
`no_grad`: tensors created inside are *inference tensors*, and autograd refuses them outright —
`RuntimeError: Inference tensors cannot be saved for backward` — rather than merely not tracking
them. So no attention backend, cudagraph mode or quantization scheme buys gradients through a vLLM
forward. `through_forward` is `False` on every vLLM configuration, and the verdict says so.

The probe still reports the other blockers it can see (`enforce_eager=False`, `cudagraph_mode=FULL`,
`quantization=...`, per-layer attention kernels with no exposed backward), because a caller
debugging this deserves all the reasons at once, and because those are what the verdict would turn
on if the `inference_mode` decorator were ever lifted upstream. `BACKWARD_CAPABLE_ATTENTION_BACKENDS`
is deliberately tiny — `TORCH_SDPA` and `FLEX_ATTENTION`, the two built from differentiable torch
ops. Everything else in vLLM's roster is a hand-written kernel, and **anything unrecognised counts
as unsupported**, so a backend added upstream is a blocker until someone checks it.

Given all that, do not try to add a "capture with gradients" worker RPC. Its precondition — a vLLM
config the probe calls differentiable — is unsatisfiable, so it would be untestable dead code.
Reaching gradients on vLLM would mean stripping `inference_mode` off the hot path *and* replacing
the paged attention kernel *and* dealing with a KV cache written in place: at that point you have
written the eager backend, which is already here.

### Why `downstream` is still true on vLLM

Because captures cross the process boundary as raw bytes. `encode_tensor_payload` /
`decode_tensor_payload` rebuild the tensor on the client, which launders vLLM's inference tensors
into ordinary ones. Verified both ways in `tests/test_autograd_support.py`, and end to end against a
real vLLM capture — the raw worker tensor refuses `backward()`, the decoded one does not. This is
load-bearing for anyone fitting probes against a serving pod, so if you ever change the payload
encoding to pass tensors by reference, that guarantee goes with it.

### Asking, rather than guessing from the backend name

`model.grad_support` is on the protocol, so a caller gates on fact:

```python
if model.grad_support.through_forward:
    ...
else:
    print(model.grad_support.blockers)  # why not
```

`apps/inference` surfaces `grad_support` in `/capabilities` (via `.describe()`) for the same reason —
the webapp should not infer gradient availability from the string `"vllm"`, since an eager pod is
also non-differentiable unless it was loaded with `requires_grad=True`, which serving does not do.


[← back to the interp-engine README](../README.md)
