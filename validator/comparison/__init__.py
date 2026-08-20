"""Cross-engine validator for the interp-engine interpretability core.

Compares activations captured by different execution engines on the same models, over the
*same pre-tokenized input ids*, and emits a results table into the engine README.

Engines, in the order the report lists them — interp-engine's own two capture paths first, then the
third-party engines those are checked against. All six capture the same four points (``resid_post``,
``resid_mid``, ``mlp_out``, ``attn_out``), each by whatever route that engine offers:
  - ``eager``    : interp-engine ``EagerModel`` (raw HF forward) — the reference
  - ``vllm``     : vLLM, captured through ``interp_engine.vllm_plugin`` (the shipped worker extension)
  - ``tlens_v2`` : TransformerLens 2 ``HookedTransformer.from_pretrained_no_processing``
  - ``tlens_v3`` : TransformerLens 3 ``TransformerBridge`` (raw-HF numerics by default)
  - ``nnsight``  : nnterp ``StandardizedTransformer`` on nnsight
  - ``sglang``   : SGLang, forward hooks injected into its scheduler subprocess

Each engine runs in its OWN environment (they don't co-resolve — vLLM and SGLang in
particular conflict), as a subprocess via ``run_engine.py``, dumping activations to disk.
``aggregate.py`` then loads all dumps, computes pairwise diffs, and ``report.py`` updates the
README table only when the numbers moved meaningfully.
"""
