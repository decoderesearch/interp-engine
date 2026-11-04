# Running interp-engine on a Google Cloud TPU

The eager backend runs on Cloud TPU through [TorchTPU](https://github.com/google-pytorch/torch_tpu),
Google's native PyTorch backend for TPUs (it replaces PyTorch/XLA, which is deprecated). TorchTPU
registers a `tpu` device with PyTorch, so a TPU is one more value of `device=`. Every capture point,
attention pattern, logit lens read-out and steering op that works on CUDA works there too, with no
code of its own in the engine.

```python
from interp_engine import capture, load_model

model = load_model("openai-community/gpt2", device="tpu", dtype="bfloat16")
ids = model.to_tokens("The capital of France is")
cache = capture(model, ids, [("resid_post", 6), ("attn_probs", 6)])
resid = cache.get("resid_post", 6)  # lives on the TPU; `.cpu()` when you want it on the host
```

On a TPU VM, `backend="auto"` picks this by itself: no CUDA plus a registered `tpu` accelerator
resolves to `eager` on `tpu`.

## What does not run on a TPU

- **The vLLM backend.** The engine's vLLM backend patches CUDA vLLM internals (`vllm_capture/`), so
  `backend="vllm"` on a TPU-only machine is refused at selection, and the refusal names
  `backend="eager"`. vLLM itself has a TPU build, but the engine's capture and steering layer is not
  ported to it.
- **`cuda`-only knobs.** `check_cuda_driver`, `check_flashinfer`, `deepgemm_fallback_kwargs` and the
  GPU-Sizer's VRAM arithmetic describe CUDA cards. They are inert or meaningless on a TPU.

## Numerics: fp32 on a TPU is not fp32 unless you ask

TorchTPU's default `float32` matmul precision is `"medium"`, meaning one pass through the MXU with
the inputs **rounded to bfloat16**. CUDA's default is true float32. A float32 load is how a caller
asks for reference numerics, so the engine warns at load when a float32 model lands on a TPU below
`"highest"`. It does not change the setting for you, because the setting is process-wide.

| you want | do this | cost |
| -------- | ------- | ---- |
| numbers that match a CPU/CUDA fp32 run | `torch.set_float32_matmul_precision("highest")` before loading, `dtype="float32"` | ~6x MXU passes |
| the served configuration | `dtype="bfloat16"` | native |
| something between | `torch.set_float32_matmul_precision("high")` | ~3x, ~fp32-grade but not bit-matched |

Measured on a v6e-1 (TorchTPU 0.2.0, torch 2.15, transformers 5.22.1), gpt2 in float32, as relative
error of the TPU capture against the same capture on CPU fp32:

| matmul precision | `resid_post@0` | `resid_post@6` | `resid_post@11` | logits |
| ---------------- | -------------- | -------------- | --------------- | ------ |
| `"medium"` (TorchTPU default) | 1.7e-3 | 6.6e-4 | 4.5e-3 | 3.1e-3 |
| `"high"` | 1.4e-5 | 1.4e-5 | 1.6e-5 | 1.7e-5 |
| `"highest"` | 3.9e-7 | 9.5e-8 | 6.4e-7 | 3.4e-7 |

`tests/test_tpu.py::test_fp32_highest_matches_cpu_fp32` is the check that `"highest"` keeps
reproducing a CPU fp32 capture.

## Execution model, briefly

TorchTPU defers work: an op is queued and compiled by XLA, and runs when something needs the result,
such as `.cpu()`, `.item()`, a `print`, or `torch.accelerator.synchronize()`. Two consequences for
interpretability code:

- **The first call at a new shape compiles.** Expect the first capture of a given prompt length to
  take seconds, and later ones at that length to be fast. A sweep over many prompt lengths
  recompiles per length. Pad or bucket the prompts if that dominates.
- **Reading a captured tensor on the host is a sync point.** Keep activations on the TPU while you
  compute with them (SAE encodes, projections, dot products), and call `.cpu()` once at the end.
  `TORCH_TPU_DEFER_AND_FUSE=1` lets TorchTPU fuse the work between those sync points.

## From your machine to a running TPU

You need the [gcloud CLI](https://cloud.google.com/sdk/docs/install), signed in
(`gcloud auth login`), and a project with billing and the Cloud TPU API on:

```bash
gcloud services enable tpu.googleapis.com --project=PROJECT
```

### 1. Find a zone and create a TPU VM

TorchTPU supports v5e, v5p, v6e (Trillium) and v7 (Ironwood). A single chip is plenty for small and
mid-sized models: `v6e-1` (32 GB HBM) or `v5litepod-1` (16 GB HBM). List what a zone offers:

```bash
gcloud compute tpus accelerator-types list --zone=us-east5-b --project=PROJECT
```

Create a **spot** VM. Spot is far cheaper, can be preempted, and is the right choice for a test.
Drop `--spot` for one that stays up. On-demand v6e capacity is often short, so expect to try more
than one zone:

```bash
gcloud compute tpus tpu-vm create ie-tpu --project=PROJECT --zone=us-east5-b --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e --spot
```

The runtime version is per generation: `v2-alpha-tpuv6e` for v6e, `v2-alpha-tpuv5-lite` for v5e,
`v2-alpha-tpuv5` for v5p. "There is no more capacity in the zone" is a capacity answer, not a
permissions one, so try the next zone. For a long wait, `gcloud compute tpus queued-resources
create` queues the request instead of failing it.

### 2. Connect

```bash
gcloud compute tpus tpu-vm ssh ie-tpu --project=PROJECT --zone=us-east5-b
```

On Windows this opens PuTTY the first time and generates a key. From then on it works the same from
PowerShell, Git Bash or WSL. Add `--command="..."` to run one command without an interactive shell.
To use VS Code Remote-SSH or plain `ssh`, run `gcloud compute config-ssh --project=PROJECT` once.
It writes the host entries for you.

### 3. Install, in this order

The order matters. TorchTPU has to be installed first so that it chooses the torch build. Installing
the engine afterwards must not replace that torch with a CUDA one from PyPI.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa   # TPU VM images are Ubuntu 22.04, whose own Python is 3.10
sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv git make
python3.12 -m venv ~/ie && source ~/ie/bin/activate
pip install --upgrade pip
pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu

git clone https://github.com/decoderesearch/interp-engine.git && cd interp-engine
pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.accelerator.current_accelerator())"   # -> device(type='tpu')
```

Without `--extra-index-url .../whl/cpu`, pip resolves `torch` from PyPI, which is the CUDA build,
and TorchTPU stops loading. Do **not** use `uv sync` or `uv run` here either: both make the venv
match `uv.lock`, whose torch is the CUDA-indexed one.

Gated checkpoints (Gemma, Llama) need a token, the same as anywhere else:
`export HF_TOKEN=...`, or `huggingface-cli login`.

### 4. Run

```bash
make test-tpu                       # tests/test_tpu.py + tests/test_select.py on the TPU
python -m pytest tests -m "not gpu and not xl"   # the CPU suite still runs here too
```

### 5. Delete it

A TPU VM bills until it is deleted. Stopping it is not enough for spot VMs, and a forgotten v6e
costs real money:

```bash
gcloud compute tpus tpu-vm delete ie-tpu --project=PROJECT --zone=us-east5-b --quiet
gcloud compute tpus tpu-vm list --zone=- --project=PROJECT    # confirm nothing is left
```

## Troubleshooting

| symptom | cause | fix |
| ------- | ----- | --- |
| `RuntimeError: Expected one of cpu, cuda, ... device type at start of device string: tpu` | TorchTPU is not installed in this venv, or its torch was replaced | reinstall in the order above, with the CPU extra index |
| `backend='vllm' was requested ... on a Cloud TPU` | the vLLM backend is CUDA-only | `backend="eager"` (or leave `backend="auto"`) |
| warning: `loaded in float32 on a TPU with ... 'medium'` | fp32 matmuls are running at bf16 input precision | `torch.set_float32_matmul_precision("highest")` before loading, or load bf16 |
| the first capture takes many seconds, later ones are fast | XLA compiling a new shape | expected; bucket prompt lengths to compile fewer shapes |
| `There is no more capacity in the zone` | no free chips of that type there | another zone, `--spot`, or a queued resource |

[← back to the docs index](README.md)
