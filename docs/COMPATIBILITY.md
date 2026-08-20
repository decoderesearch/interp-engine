# transformers versions

This engine has no forward pass of its own. It attaches hooks to `transformers` modules and reports
what they compute, which is the property that makes its captures trustworthy — and it means the
version of `transformers` you install is part of your numerical result, the same way the checkpoint
is. `pyproject.toml` asks for `transformers>=4.57.1` with no ceiling. This page is what that floor
does and does not promise.

## What is actually tested

Two versions, and they are tested for different things:

| version | what runs | what it proves |
| --- | --- | --- |
| 4.57.1, the declared floor | the config-facing tests, no weights, no network (`floor` job in `.github/workflows/engine-tests.yml`) | the engine still *reads* an old transformers correctly |
| whatever CI resolves today | the full CPU and GPU suites, including the gpt-2 golden parity gate | the engine is right on the version people install |

Everything between the two is inference from those endpoints. Raising the floor is therefore a
one-line edit plus a CI run, and it should be done whenever the old version stops being read
correctly rather than kept out of sentiment.

## What "supported" means, and what it does not

Supported means the engine resolves this version's configs into the right facts and hooks the right
modules. It does **not** mean two supported versions produce the same numbers. They do not, and the
difference is not always visible:

- **The forward pass changes.** transformers 5.15.0 fixed DeepSeek-V2's YaRN `mscale`: before it, the
  attention softmax ran at the wrong temperature, so every attention-derived activation on that
  family was wrong by 1–5% cosine. A capture on 5.14.1 is a faithful capture of a wrong forward pass.
- **Configs move.** transformers 5.15.0 made Gemma-4's `head_dim` and `num_key_value_heads` per-layer
  attributes. Reading them the old way now raises rather than returning a plausible-looking global
  number, which is the good case; the bad case is a field that silently keeps a stale value.
- **New architectures arrive.** A checkpoint added in 5.15 does not load on 4.57 at all — a loud,
  obvious failure, and the least interesting of the three.

The first two are the reason this page exists. Neither shows up as an error, a shape mismatch, or a
NaN. If you have one engine, you have no second opinion, so the engine has to say something itself.

## Combinations known to be wrong

`facts.TRANSFORMERS_CAVEATS` is the list, and `resolve_facts` warns once per process when the loaded
architecture matches one on the installed version:

```
RuntimeWarning: DeepseekV2ForCausalLM on transformers 5.14.1: attention runs at the wrong
temperature -- the YaRN `mscale_all_dim` factor is missing from the softmax scale ... Fixed in
transformers 5.15.0 (https://github.com/huggingface/transformers/pull/47435) -- upgrade before
trusting these activations.
```

The bar for adding a row is deliberately high: a named upstream fix, the release it landed in, and an
effect on *captured activations*. A deprecation, a speed regression or a bug in a code path the
engine does not read is not a row here — a warning people learn to ignore is worse than no warning.
Rows narrow themselves further where they can: the DeepSeek entry checks for YaRN scaling in the
config, so the same architecture without it stays quiet.

Removing a row is fine once the floor rises above its `fixed_in`, and at that point it is dead code
rather than history worth keeping.

## Practical advice

- Record the transformers version alongside any activations you publish or cache. It belongs with the
  checkpoint revision, not in a footnote.
- Pin it in an environment whose captures are compared over time. Two runs of the same prompt on the
  same checkpoint can differ across a transformers upgrade, and nothing in the output says so.
- When cross-checking against another implementation (vLLM, TransformerLens, someone's paper), rule
  the version out first. On DeepSeek-V2 it was the whole answer, and it took a cross-engine sweep to
  see it — see `interp-engine-validator`'s `docs/ENGINE_DIFFERENCES.md`, "When the reference is the
  one that is wrong".
