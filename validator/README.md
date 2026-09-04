# interp-engine-validator

[interp-engine](https://github.com/decoderesearch/interp-engine) is a new engine for interpretability inference.

**The critical question for interp-engine is: Is it correct?**

This directory, `interp-engine-validator`, compares results between `interp-engine`, `transformerlens` (both v2 and v3), and `nnsight`/`nnterp` across 50+ models to ensure that `interp-engine`'s outputs match the results from other engines across all hook points. It contains:

- **Comparison Results** of the latest validations, updated periodically
- **Sweep Code** for running the validation/comparisons across all engines and models

## Contents

- [Comparison Results](#comparison-results) — the results table, one row per model
- [Documentation](#documentation) — which doc answers which question
- [Model support](#model-support) — which architectures work, and what has been shown about each
- [Layout](#layout) — what each part of this directory owns
- [How the engines differ](#how-the-engines-differ) — hook naming and per-engine limits
- [How to run](#how-to-run) — the full sweep, one column, or one cell
- [Filing an engine bug report](#filing-an-engine-bug-report) — when the fault is the other engine's

<!-- ENGINE-COMPARISON:START -->

## Comparison Results

| model | interp-engine eager<br>v1.6.0 | interp-engine vllm<br>[v0.28.0](https://github.com/vllm-project/vllm/commit/2cf0a6915ce544dc493a0990f2ea38d81601128a) | interp-engine vllm-static<br>[v0.28.0](https://github.com/vllm-project/vllm/commit/2cf0a6915ce544dc493a0990f2ea38d81601128a) | tlens_v2<br>[v3.8.1](https://github.com/TransformerLensOrg/TransformerLens/commit/2a05c15df238caf8b7869188ea27b5d39a903d35) | tlens_v3<br>[v3.8.1](https://github.com/TransformerLensOrg/TransformerLens/commit/2a05c15df238caf8b7869188ea27b5d39a903d35) | nnsight<br>[v0.7.0](https://github.com/ndif-team/nnsight/commit/884bd1c7b0dbd53be01ade6a18843078d90367b4) |
| --- | --- | --- | --- | --- | --- | --- |
| `allenai/OLMo-2-0425-1B`<br>[Results](comparison/results/allenai/OLMo-2-0425-1B/0_result_details.md) | ref [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/eager.json) | ✅ [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/vllm.json) | ✅ [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/vllm-static.json) | ✅ [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/tlens_v2.json) | ✅ [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/tlens_v3.json) | ✅ [09/03/26](comparison/results/allenai/OLMo-2-0425-1B/nnsight.json) |
| `allenai/Olmo-3-1125-32B`<br>[Results](comparison/results/allenai/Olmo-3-1125-32B/0_result_details.md) | ref [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/eager.json) | ✅ [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/vllm.json) | ✅ [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/vllm-static.json) | ✅ [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/tlens_v2.json) | ✅ [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/tlens_v3.json) | ✅ [09/03/26](comparison/results/allenai/Olmo-3-1125-32B/nnsight.json) |
| `bigcode/gpt_bigcode-santacoder`<br>[Results](comparison/results/bigcode/gpt_bigcode-santacoder/0_result_details.md) | ref [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/eager.json) | ✅ [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/vllm.json) | ✅ [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/vllm-static.json) | unsupported [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/tlens_v2.json) | ✅ [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/tlens_v3.json) | ✅ [09/03/26](comparison/results/bigcode/gpt_bigcode-santacoder/nnsight.json) |
| `bigcode/starcoder2-3b`<br>[Results](comparison/results/bigcode/starcoder2-3b/0_result_details.md) | ref [09/03/26](comparison/results/bigcode/starcoder2-3b/eager.json) | ✅ [09/03/26](comparison/results/bigcode/starcoder2-3b/vllm.json) | ✅ [09/03/26](comparison/results/bigcode/starcoder2-3b/vllm-static.json) | unsupported [09/03/26](comparison/results/bigcode/starcoder2-3b/tlens_v2.json) | ✅ [09/03/26](comparison/results/bigcode/starcoder2-3b/tlens_v3.json) | ✅ [09/03/26](comparison/results/bigcode/starcoder2-3b/nnsight.json) |
| `bigscience/bloom-560m`<br>[Results](comparison/results/bigscience/bloom-560m/0_result_details.md) | ref [09/03/26](comparison/results/bigscience/bloom-560m/eager.json) | ✅ [09/03/26](comparison/results/bigscience/bloom-560m/vllm.json) | ✅ [09/03/26](comparison/results/bigscience/bloom-560m/vllm-static.json) | ✅ [09/03/26](comparison/results/bigscience/bloom-560m/tlens_v2.json) | [🐞](https://github.com/TransformerLensOrg/TransformerLens/issues/1639) [09/03/26](comparison/results/bigscience/bloom-560m/tlens_v3.json) | [🐞](https://github.com/ndif-team/nnterp/issues/51) [09/03/26](comparison/results/bigscience/bloom-560m/nnsight.json) |
| `deepseek-ai/DeepSeek-V2-Lite`<br>[Results](comparison/results/deepseek-ai/DeepSeek-V2-Lite/0_result_details.md) | ref [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/eager.json) | ✅ [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/vllm.json) | ✅ [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/vllm-static.json) | unsupported [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/tlens_v2.json) | ✅ [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/tlens_v3.json) | ✅ [09/03/26](comparison/results/deepseek-ai/DeepSeek-V2-Lite/nnsight.json) |
| `deepseek-ai/DeepSeek-V4-Flash-0731`<br>[Results](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/0_result_details.md) | ref [09/04/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/eager.json) | ✅ [09/04/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/vllm.json) | ✅ [09/04/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/vllm-static.json) | unsupported [09/03/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/tlens_v2.json) | ⚠️ [09/04/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/tlens_v3.json) | unsupported [09/04/26](comparison/results/deepseek-ai/DeepSeek-V4-Flash-0731/nnsight.json) |
| `EleutherAI/pythia-70m-deduped`<br>[Results](comparison/results/EleutherAI/pythia-70m-deduped/0_result_details.md) | ref [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/eager.json) | ✅ [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/vllm.json) | ✅ [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/vllm-static.json) | unsupported [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/tlens_v2.json) | unsupported [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/tlens_v3.json) | ✅ [09/03/26](comparison/results/EleutherAI/pythia-70m-deduped/nnsight.json) |
| `facebook/opt-125m`<br>[Results](comparison/results/facebook/opt-125m/0_result_details.md) | ref [09/03/26](comparison/results/facebook/opt-125m/eager.json) | ✅ [09/03/26](comparison/results/facebook/opt-125m/vllm.json) | ✅ [09/03/26](comparison/results/facebook/opt-125m/vllm-static.json) | ✅ [09/03/26](comparison/results/facebook/opt-125m/tlens_v2.json) | ✅ [09/03/26](comparison/results/facebook/opt-125m/tlens_v3.json) | ✅ [09/03/26](comparison/results/facebook/opt-125m/nnsight.json) |
| `google/gemma-2-27b`<br>[Results](comparison/results/google/gemma-2-27b/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-2-27b/eager.json) | ✅ [09/03/26](comparison/results/google/gemma-2-27b/vllm.json) | ✅ [09/03/26](comparison/results/google/gemma-2-27b/vllm-static.json) | ✅ [09/03/26](comparison/results/google/gemma-2-27b/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-2-27b/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-2-27b/nnsight.json) |
| `google/gemma-3-1b-it`<br>[Results](comparison/results/google/gemma-3-1b-it/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-3-1b-it/eager.json) | ✅ [09/03/26](comparison/results/google/gemma-3-1b-it/vllm.json) | ✅ [09/03/26](comparison/results/google/gemma-3-1b-it/vllm-static.json) | ✅ [09/03/26](comparison/results/google/gemma-3-1b-it/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-3-1b-it/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-3-1b-it/nnsight.json) |
| `google/gemma-3-27b-it`<br>[Results](comparison/results/google/gemma-3-27b-it/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-3-27b-it/eager.json) | ✅ [09/03/26](comparison/results/google/gemma-3-27b-it/vllm.json) | ✅ [09/03/26](comparison/results/google/gemma-3-27b-it/vllm-static.json) | ✅ [09/03/26](comparison/results/google/gemma-3-27b-it/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-3-27b-it/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-3-27b-it/nnsight.json) |
| `google/gemma-4-12B-it`<br>[Results](comparison/results/google/gemma-4-12B-it/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-4-12B-it/eager.json) | ⚠️ [09/03/26](comparison/results/google/gemma-4-12B-it/vllm.json) | ⚠️ [09/03/26](comparison/results/google/gemma-4-12B-it/vllm-static.json) | unsupported [09/03/26](comparison/results/google/gemma-4-12B-it/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-4-12B-it/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-4-12B-it/nnsight.json) |
| `google/gemma-4-26B-A4B-it`<br>[Results](comparison/results/google/gemma-4-26B-A4B-it/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/eager.json) | ⚠️ [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/vllm.json) | [🐞](https://github.com/vllm-project/vllm/issues/55238) [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/vllm-static.json) | unsupported [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-4-26B-A4B-it/nnsight.json) |
| `google/gemma-4-31B`<br>[Results](comparison/results/google/gemma-4-31B/0_result_details.md) | ref [09/03/26](comparison/results/google/gemma-4-31B/eager.json) | ✅ [09/03/26](comparison/results/google/gemma-4-31B/vllm.json) | ✅ [09/03/26](comparison/results/google/gemma-4-31B/vllm-static.json) | unsupported [09/03/26](comparison/results/google/gemma-4-31B/tlens_v2.json) | ✅ [09/03/26](comparison/results/google/gemma-4-31B/tlens_v3.json) | ✅ [09/03/26](comparison/results/google/gemma-4-31B/nnsight.json) |
| `HuggingFaceTB/SmolLM3-3B`<br>[Results](comparison/results/HuggingFaceTB/SmolLM3-3B/0_result_details.md) | ref [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/eager.json) | ✅ [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/vllm.json) | ✅ [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/vllm-static.json) | unsupported [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/tlens_v2.json) | ✅ [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/tlens_v3.json) | ✅ [09/03/26](comparison/results/HuggingFaceTB/SmolLM3-3B/nnsight.json) |
| `ibm-granite/granite-3.0-1b-a400m-base`<br>[Results](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/0_result_details.md) | ref [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/eager.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/vllm.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/vllm-static.json) | unsupported [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/tlens_v2.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/tlens_v3.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.0-1b-a400m-base/nnsight.json) |
| `ibm-granite/granite-3.3-2b-instruct`<br>[Results](comparison/results/ibm-granite/granite-3.3-2b-instruct/0_result_details.md) | ref [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/eager.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/vllm.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/vllm-static.json) | unsupported [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/ibm-granite/granite-3.3-2b-instruct/nnsight.json) |
| `LiquidAI/LFM2-8B-A1B`<br>[Results](comparison/results/LiquidAI/LFM2-8B-A1B/0_result_details.md) | ref [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/eager.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/vllm.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/vllm-static.json) | unsupported [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/tlens_v2.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/tlens_v3.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2-8B-A1B/nnsight.json) |
| `LiquidAI/LFM2.5-230M`<br>[Results](comparison/results/LiquidAI/LFM2.5-230M/0_result_details.md) | ref [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/eager.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/vllm.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/vllm-static.json) | unsupported [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/tlens_v2.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/tlens_v3.json) | ✅ [09/03/26](comparison/results/LiquidAI/LFM2.5-230M/nnsight.json) |
| `meta-llama/Llama-3.1-8B`<br>[Results](comparison/results/meta-llama/Llama-3.1-8B/0_result_details.md) | ref [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/eager.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/vllm.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/vllm-static.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/tlens_v2.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/tlens_v3.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.1-8B/nnsight.json) |
| `meta-llama/Llama-3.3-70B-Instruct`<br>[Results](comparison/results/meta-llama/Llama-3.3-70B-Instruct/0_result_details.md) | ref [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/eager.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/vllm.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/vllm-static.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/meta-llama/Llama-3.3-70B-Instruct/nnsight.json) |
| `microsoft/phi-2`<br>[Results](comparison/results/microsoft/phi-2/0_result_details.md) | ref [09/03/26](comparison/results/microsoft/phi-2/eager.json) | ✅ [09/03/26](comparison/results/microsoft/phi-2/vllm.json) | ✅ [09/03/26](comparison/results/microsoft/phi-2/vllm-static.json) | ✅ [09/03/26](comparison/results/microsoft/phi-2/tlens_v2.json) | ✅ [09/03/26](comparison/results/microsoft/phi-2/tlens_v3.json) | ✅ [09/03/26](comparison/results/microsoft/phi-2/nnsight.json) |
| `microsoft/Phi-3-mini-4k-instruct`<br>[Results](comparison/results/microsoft/Phi-3-mini-4k-instruct/0_result_details.md) | ref [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/eager.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/vllm.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/vllm-static.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-3-mini-4k-instruct/nnsight.json) |
| `microsoft/Phi-mini-MoE-instruct`<br>[Results](comparison/results/microsoft/Phi-mini-MoE-instruct/0_result_details.md) | ref [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/eager.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/vllm.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/vllm-static.json) | unsupported [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/microsoft/Phi-mini-MoE-instruct/nnsight.json) |
| `mistralai/Mistral-7B-v0.1`<br>[Results](comparison/results/mistralai/Mistral-7B-v0.1/0_result_details.md) | ref [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/eager.json) | ✅ [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/vllm.json) | ✅ [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/vllm-static.json) | ✅ [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/tlens_v2.json) | ✅ [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/tlens_v3.json) | ✅ [09/03/26](comparison/results/mistralai/Mistral-7B-v0.1/nnsight.json) |
| `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`<br>[Results](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/0_result_details.md) | ref [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/eager.json) | ✅ [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/vllm.json) | ✅ [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/vllm-static.json) | unsupported [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/tlens_v2.json) | ✅ [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/tlens_v3.json) | ✅ [09/03/26](comparison/results/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/nnsight.json) |
| `openai-community/gpt2`<br>[Results](comparison/results/openai-community/gpt2/0_result_details.md) | ref [09/03/26](comparison/results/openai-community/gpt2/eager.json) | ✅ [09/03/26](comparison/results/openai-community/gpt2/vllm.json) | ✅ [09/03/26](comparison/results/openai-community/gpt2/vllm-static.json) | ✅ [09/03/26](comparison/results/openai-community/gpt2/tlens_v2.json) | ✅ [09/03/26](comparison/results/openai-community/gpt2/tlens_v3.json) | ✅ [09/03/26](comparison/results/openai-community/gpt2/nnsight.json) |
| `openai/gpt-oss-20b`<br>[Results](comparison/results/openai/gpt-oss-20b/0_result_details.md) | ref [09/03/26](comparison/results/openai/gpt-oss-20b/eager.json) | ✅ [09/03/26](comparison/results/openai/gpt-oss-20b/vllm.json) | ✅ [09/03/26](comparison/results/openai/gpt-oss-20b/vllm-static.json) | ✅ [09/03/26](comparison/results/openai/gpt-oss-20b/tlens_v2.json) | ✅ [09/03/26](comparison/results/openai/gpt-oss-20b/tlens_v3.json) | ✅ [09/03/26](comparison/results/openai/gpt-oss-20b/nnsight.json) |
| `Qwen/Qwen2.5-7B-Instruct`<br>[Results](comparison/results/Qwen/Qwen2.5-7B-Instruct/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/vllm-static.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen2.5-7B-Instruct/nnsight.json) |
| `Qwen/Qwen3-30B-A3B`<br>[Results](comparison/results/Qwen/Qwen3-30B-A3B/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/vllm-static.json) | unsupported [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-30B-A3B/nnsight.json) |
| `Qwen/Qwen3-32B`<br>[Results](comparison/results/Qwen/Qwen3-32B/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen3-32B/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-32B/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-32B/vllm-static.json) | unsupported [09/03/26](comparison/results/Qwen/Qwen3-32B/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-32B/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-32B/nnsight.json) |
| `Qwen/Qwen3-Next-80B-A3B-Instruct`<br>[Results](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/vllm-static.json) | unsupported [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3-Next-80B-A3B-Instruct/nnsight.json) |
| `Qwen/Qwen3.6-35B-A3B`<br>[Results](comparison/results/Qwen/Qwen3.6-35B-A3B/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/vllm-static.json) | unsupported [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.6-35B-A3B/nnsight.json) |
| `Qwen/Qwen3.8-27B`<br>[Results](comparison/results/Qwen/Qwen3.8-27B/0_result_details.md) | ref [09/03/26](comparison/results/Qwen/Qwen3.8-27B/eager.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.8-27B/vllm.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.8-27B/vllm-static.json) | unsupported [09/03/26](comparison/results/Qwen/Qwen3.8-27B/tlens_v2.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.8-27B/tlens_v3.json) | ✅ [09/03/26](comparison/results/Qwen/Qwen3.8-27B/nnsight.json) |

One row per model, one column per engine. A cell rolls up **every** hook point we ask that engine for against `eager`, the raw-HF reference, in the checkpoint's native dtype: `resid_post`, `resid_mid`, `mlp_out`, `mlp_out_post`, `attn_out`, `attn_out_post`, `attn_in`, `mlp_act`, `q_norm_in`, `q_norm_out`, `k_norm_in`, `k_norm_out`, `value`, `z`, `router_logits`, `embeddings`, `final_norm`, `attn_scores`, plus `mlp_pre`, `mlp_pre_linear` on the eager engines (no fused engine materializes those), and on a trunk carrying several residual streams the hyper-connection rows `resid_streams`, `attn_stream_collapse`, `attn_stream_write`, `attn_stream_mix`, `mlp_stream_collapse`, `mlp_stream_write`, `mlp_stream_mix` (DeepSeek-V4, Motif 3). The first three columns are interp-engine's own capture paths — eager PyTorch, hooked vLLM through `interp_engine.vllm_plugin`, and vLLM CUDA-graph static taps (`vllm-static`); the rest are the third-party engines they are checked against.

| cell | meaning |
| --- | --- |
| ✅ | every hook point agrees at cosine ≈ 1.0 |
| ⚠️ | a hook point differs significantly in value, or was not captured — an absence with an architectural reason written down in `spec.ENGINE_GAPS` does not warn (the model's **Results** page lists it under *Not compared*, with the reason) |
| ❌ | a regression, a structurally wrong tensor (mismatched shape, all-zero, unrelated direction), or a crash |
| 🐞 | a bug in one of the two engines being compared rather than in the capture: investigated, reduced to a repro, and filed — the cell links to the issue. Usually the engine under test; `ref🐞` in the reference column means the baseline is the wrong one |

**Results** under each model is that model's page: every point, every engine, what agreed and what did not, with the reason and the layer — the view between this table's one glyph and the raw JSONs. Each cell is dated when *it* was captured and links its own detail JSON: per-hook-point cosine, relative and absolute diff, the versions (and commits) of the stack that produced it, the commands to reproduce that one cell, and any tolerance waiver that applied (`spec.TOLERANCE_WAIVERS` — for checkpoints whose own bf16 arithmetic explains a difference, which is measured before it is waived). A column's heading carries the version most of its cells ran at; a cell captured against a different one says so under its date. `unsupported` means that engine's loader declines the checkpoint (the reasons are in [docs/COMPARISON.md](docs/COMPARISON.md)), `no ref` that it captured cleanly but `eager` did not, so there is nothing to score it against until that one cell is rerun, and `—` that the pair has never run. `ref*` in the reference column means `eager` ran but declined a point another engine captured, with nothing in `spec.REFERENCE_GAPS` declaring why — those cells score nothing, so without the marker the row would read clean in every column; the reference's own JSON lists the points and which engines produced them. `ref🐞` is the rarer one: a point the reference gets *wrong*, with an issue filed against it (`engine_bugs.REFERENCE_BUGS`), which makes the engines that disagree the ones that are right — their cells carry 🐞 at those points and are scored on the rest.

<!-- ENGINE-COMPARISON:END -->

## Documentation

| doc                                                      | when you need it                                                                          |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [docs/COMPARISON.md](docs/COMPARISON.md)                 | running the validator: venv setup, one cell, the full sweep, and how to read a disagreement |
| [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md)         | adding or validating a new model family, end to end                                       |
| [docs/MODELS_STATUS.md](docs/MODELS_STATUS.md)           | whether a given model works, and what has been shown about it                             |
| [docs/ENGINE_DIFFERENCES.md](docs/ENGINE_DIFFERENCES.md) | how the six engines differ: hook naming, what each can capture, what each refuses         |

## Model support

<!-- MODEL-SUPPORT:START -->

Support is tracked per **architecture**, against vLLM 0.26.0's model registry -- what this
engine has to cover is what a user can serve. Both columns below come from tests, not from a
maintained list:

|  | architectures | what has been shown |
| --- | --- | --- |
| **verified** | 31 | captured on real weights and checked point-by-point against other engines |
| **resolves** | 59 | every point resolves and the arithmetic invariants hold; no independent engine has reproduced it |
| **unaudited** | 46 | not probed: transformers has no class for the family, or its own config defaults do not build |
| **broken** | 2 | a module tree the `(point, layer)` addressing cannot express, with the reason |

Verified today, on 32 checkpoints between them:

`BloomForCausalLM`, `DeepseekV2ForCausalLM`, `GPT2LMHeadModel`, `GPTBigCodeForCausalLM`, `GPTNeoXForCausalLM`, `Gemma2ForCausalLM`, `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`, `Gemma4ForConditionalGeneration`, `GptOssForCausalLM`, `GraniteForCausalLM`, `GraniteMoeForCausalLM`, `Lfm2ForCausalLM`, `Lfm2MoeForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`, `NemotronHForCausalLM`, `OPTForCausalLM`, `Olmo2ForCausalLM`, `Olmo3ForCausalLM`, `Phi3ForCausalLM`, `PhiForCausalLM`, `PhiMoEForCausalLM`, `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `SmolLM3ForCausalLM`, `Starcoder2ForCausalLM`.

Unverified is the absence of a *cross-engine* run, not a prediction of failure: an architecture in the
second tier resolves every hook point against its real module tree and satisfies three arithmetic
invariants on a tiny random model of the family -- `attn_probs @ value == z`, `resid_pre + attn_out_post
+ mlp_out_post == resid_post`, and `down_proj(mlp_act) == mlp_out`. Where a point genuinely does not
exist -- a Mamba block's attention, a latent-attention
model's `value`, the residual between the sublayers of a parallel block -- it is refused with an
explanation rather than returned as a plausible tensor.
[docs/MODELS_STATUS.md](docs/MODELS_STATUS.md) has every architecture by name, with the reason and,
for the unverified, what would promote it.

<!-- MODEL-SUPPORT:END -->

## Layout

- `comparison/` — the validator. `tokenize_inputs.py` writes the shared inputs, `run_engine.py`
  captures one engine into a dump, `aggregate.py` scores the dumps and re-renders the table,
  `report.py` owns the rendering, `spec.py` is the model/point/scoring spec, and `engines/` holds
  one adapter per engine (including the hooks injected into SGLang's scheduler subprocess).
- `comparison/results/` — one JSON per (checkpoint, engine) cell. The table is a pure rendering of
  this directory, which is what makes a partial run safe.
- `tests/` — the scoring, reporting and coverage tests. Pure python, no GPU.

## Correctness

### How the engines differ

Agreeing on the numbers is what the table above measures; everything else about these engines differs.
The two that matter in practice are what a hook point is _called_ and what an engine simply will not do:

| engine                             | Hook Name Convention                                                                       | Notes                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| interp-engine eager                | `cache["mlp_out", 5]` — canonical name + layer                                             | no serving throughput; `attn_probs`/`attn_scores` need `attn_implementation="eager"`; fp16 refused where it overflows                                                                                                                                                                              |
| interp-engine vLLM                 | same names, over `collective_rpc` to the worker plugin                                     | no CUDA graphs or prefix caching while capturing (hooks would not fire); no gradients through the forward, ever; `attn_probs`/`attn_scores` only by off-kernel recompute; `z`/`value` are per-rank under TP; the MLP input branches (`gate_up_proj` is fused) and the MoE selection are eager-only |
| `tlens_v2` (`HookedTransformer`)   | `blocks.5.mlp.hook_out` (raw) vs `blocks.5.hook_mlp_out` (post-norm residual contribution) | only checkpoints in its hardcoded name registry; converts _after_ loading, so ~2x peak memory                                                                                                                                                                                                      |
| `tlens_v3` (`TransformerBridge`)   | same hook strings, plus compatibility aliases                                              | multimodal adapters need optional deps                                                                                                                                                                                                                                                             |
| `nnsight` (nnterp)                 | `mlps_output[5]` — a standardized accessor                                                 | no accessor for `z`/`value`; `attn_probs` only via a source patch; one point per trace; a hybrid trunk needs `RenameConfig(ignore_attn=True)`                                                                                                                                                      |
| `sglang` (paused, no column above) | no hook surface — we inject hooks into its scheduler subprocess                            | bf16 only; no CUDA graphs; no `attn_probs`; its venv no longer starts (`triton.runtime.cache.default_cache_dir` is gone), so the sweep skips it — the adapter is still there and `MODE=engine ENGINE=sglang` still runs it                                                                         |

The trap is that `blocks.5.hook_mlp_out` and `mlps_output[5]` are the same tensor on Llama and different
tensors on Gemma, because TransformerLens' block-level hook fires _after_ the post-sublayer norm.
[interp-engine's ENGINE_HOOK_MAPPINGS.md](https://github.com/decoderesearch/interp-engine/blob/main/docs/ENGINE_HOOK_MAPPINGS.md) maps every hook point across the three
hookable stacks — including the ones TransformerLens has and we do not — and
[docs/ENGINE_DIFFERENCES.md](docs/ENGINE_DIFFERENCES.md) has the per-engine limitations and where each is
enforced in code; `interp_engine.mappers` translates names both directions.

### How to run

Needs a CUDA box and the three venvs ([docs/COMPARISON.md](docs/COMPARISON.md#setting-up-a-fresh-box-to-run-the-sweep)
sets one up). One model at a time, so the HF cache stays bounded.

```bash
AGGREGATE=1 bash comparison/run_all_models.sh                 # every model, every engine, then update the table
MODE=retry bash comparison/run_all_models.sh                  # only the cells that are missing or not ok
MODE=engine ENGINE=nnsight VERSION=latest bash comparison/run_all_models.sh  # refresh one column at a new release
```

`MODE=engine` takes one engine or several (`ENGINE="tlens_v2 tlens_v3"`), and `VERSION` upgrades that
engine in its venv first — which is the point, since a column's version heading is only as good as the
build that produced it. Every other cell keeps its own verdict, date and version.

To score an interp-engine that has not shipped yet, name the checkout instead of moving the pin:

```bash
LOCAL_ENGINE=~/code/neuronpedia/interp-engine AGGREGATE=1 bash comparison/run_all_models.sh
```

That runs the checkout in front of the installed wheel in every venv, and writes the dumps, cells and
table to gitignored scratch paths rather than `comparison/results` — a checkout reports the _released_
version string, so those cells would be indistinguishable from published ones in the table
([docs/COMPARISON.md](docs/COMPARISON.md#setting-up-a-fresh-box-to-run-the-sweep) has the details).

For one model, or one cell, run the engines directly and aggregate; each cell's JSON also carries the exact
commands that reproduce _it_ under `run.replicate`:

```bash
PYTHONPATH=. .venv-cmp/bin/python -m comparison.tokenize_inputs --dumps dumps --models gemma-2-2b
PYTHONPATH=. .venv-cmp/bin/python -m comparison.run_engine --engine eager --dumps dumps --model gemma-2-2b --device cuda
PYTHONPATH=. .venv-cmp/bin/python -m comparison.aggregate --dumps dumps
```

### Filing an engine bug report

When a cell disagrees and the fault looks like the _other_ engine's, this is the procedure — for a person
or for an agent reading comparison output. Rule it out as ours first: the checks that separate the two, and
the four bugs already filed, are in
[docs/COMPARISON.md](docs/COMPARISON.md#bugs-filed-against-the-other-engines). When the two that split are
`vllm` and `vllm-static`, one run settles it —
[`VLLM_BATCH_INVARIANT=1`](docs/COMPARISON.md#is-a-vllm-cell-ours-at-all-vllm_batch_invariant1). Then:

1. **Search that engine's tracker for a duplicate**, issues and PRs, open and closed. Search the mechanism
   and the symptom, not just the model name. If it is already reported, add a row to `ENGINE_BUGS` in
   [comparison/engine_bugs.py](comparison/engine_bugs.py) pointing at that issue — the cell becomes 🐞 and
   links there — and comment on the thread if you have something it lacks.
2. **If there is no duplicate, write the issue and hand it to the user to post.** Print it in the
   conversation for copy-paste; do not commit it to a file, and do not try to file it yourself. Read that
   repo's conventions first — its issue template (`.github/ISSUE_TEMPLATE/`) and `CONTRIBUTING.md`,
   including any policy on AI-assisted contributions — and satisfy every required field, since a missing
   one gets an issue closed unread. As of 2026-08: SGLang's form requires a `[Bug] ` title and
   `python3 -m sglang.check_env` output; FlashInfer's requires `python -m flashinfer.collect_env`
   untrimmed and a standalone script rather than a server command; TransformerLens' asks for a
   `[Bug Report] ` title, a minimal example with the stack trace, and how `transformer_lens` was installed
   plus OS and Python version. Neither SGLang nor FlashInfer nor TransformerLens asks you to disclose AI
   assistance on an issue — FlashInfer's only clause is about pull requests, and it asks that you
   understand and can defend the change, which is the real bar here too.
3. **Write it as a user of that library, not as us.** No mention of interp-engine, the validator, or our
   engine names or glyphs: a maintainer should see only their project, a repro they can paste and run,
   expected vs. actual with numbers, the root cause if you found one, and the versions it reproduces on
   (theirs, including their latest release or `main` — "does it still happen on main" is the first thing
   they will ask). Prefer a repro against plain `transformers`, or none at all — the strongest one we have
   filed loads no model and calls one attention wrapper three ways.
4. **Include nothing private.** No tokens, keys, `.env` contents, internal hostnames or paths, cluster or
   account names, unreleased model names, or anything else that identifies a person or our infrastructure.
   GPU model, driver and library versions are fine and wanted; `$HOME`-rooted paths and usernames are not.

Once it is filed, add the row to `ENGINE_BUGS` with the issue URL and a one-sentence `mechanism`, then
re-aggregate so the cell renders 🐞. A row must not paper over a live pass: if the engine starts agreeing
the cell returns to ✅ on its own and the row can be deleted.
