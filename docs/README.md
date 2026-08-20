# interp-engine documentation

Which doc answers which question. Start at [USAGE.md](USAGE.md) if you have not run the engine yet.

| doc                                                | when you need it                                                                               |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| [USAGE.md](USAGE.md)                               | install, load a model, capture, generate, steer, lens — start here                             |
| [SUPPORTED_POINTS.md](SUPPORTED_POINTS.md)         | every point, its width, and whether each backend can serve it                                  |
| [ENGINE_HOOK_MAPPINGS.md](ENGINE_HOOK_MAPPINGS.md) | every point mapped across interp-engine, TransformerLens and nnsight                           |
| [PORTING.md](PORTING.md)                           | translating code from TransformerLens, nnsight or nnterp                                       |
| [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md)       | porting code onto the engine: recipes, hard rules, error-to-fix (written for a coding agent)   |
| [ARCHITECTURE_QUIRKS.md](ARCHITECTURE_QUIRKS.md)   | every architecture quirk the engine knows about, and where a per-model fact is allowed to live |
| [GRADIENTS.md](GRADIENTS.md)                       | what is differentiable, on which backend, and what is silently not                             |
| [PERFORMANCE.md](PERFORMANCE.md)                   | vLLM speed/feature tradeoffs and quantization support                                          |
| [COMPATIBILITY.md](COMPATIBILITY.md)               | which transformers versions are tested, and the ones known to compute a model wrongly          |
| [INTERNALS.md](INTERNALS.md)                       | what each module owns, and what the test suite checks about the engine                         |

Editing the engine rather than calling it? [AGENTS.md](../AGENTS.md) at the repo root is the design
decisions and the boundaries between modules.

[← back to the interp-engine README](../README.md)
