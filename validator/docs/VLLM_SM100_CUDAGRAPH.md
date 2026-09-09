# vLLM CUDA-graph replay on compute capability 10.x

`google/gemma-4-26B-A4B-it` returns different greedy output from vLLM depending on whether CUDA
graphs are on, on B200 and B300 and nowhere else. Filed as
[vllm#55238](https://github.com/vllm-project/vllm/issues/55238). This is what was measured, what it
rules out, and what is still open — written so someone else can pick the kernel hunt up from here.

Nothing in this document needs interp-engine. Every number below is plain `vllm.LLM`.

## The measurement

`comparison/vllm_cudagraph_repro.py` holds compilation mode at `NONE` in both arms and moves only
`cudagraph_mode`, because `enforce_eager` — which the original report toggled — moves torch.compile
and CUDA graphs together.

| box | capability | worst delta | argmax flips | next token |
| --- | --- | --- | --- | --- |
| H200 | 9.0 | 0.000, bit-identical at every position | none | unchanged |
| RTX PRO 6000 | 12.0 | 0.000, bit-identical at every position | none | unchanged |
| B300 | 10.3 | 6.773 nats | 2 of 12 positions | 992 → 9079 |

The B200 the issue was filed from shows the same token move. It is the same architecture family as
the B300 in vLLM's own terms (`is_device_capability_family(100)` is a major-version match), and the
two boxes agree *bitwise* across all 62 points of a validator sweep, on different hosts five days
apart, where any two boxes from different families agree on at most 4.

Each arm is bit-identical to itself across two passes, checked on the H200 and the B300, so this is
deterministic rather than replay noise.

## It follows CUDA-graph padding, not the prompt

Sweeping token-prefix lengths of one prompt on the B300:

| padded to | prompt lengths | result |
| --- | --- | --- |
| 2, 4, 8 | 3, 5–7 | bit-identical |
| **16** | **9–15** | **diverges, up to 6.77 nats** |
| 24 | 17–23 | bit-identical |
| **32** | **25–31** | **diverges, 3.66 nats** |
| 40, 48, 56, 64 | 33–39, 41–63 | bit-identical |

Lengths that exactly fill a captured size are clean, and so is every length in most buckets. Dropping
16 from `cudagraph_capture_sizes` *moves* the failure instead of removing it: length 16 then breaks by
7.098 nats, lengths 11–15 keep breaking, and length 20 stays clean. So it depends on the real token
count and the padded size together, not on either alone.

This is also why the earlier "one prompt position with a massive activation" reading was dropped. The
position that fails is fixed for a given prefix, but which lengths fail is decided by padding, and a
14x-norm token cannot explain a failure that switches off at length 16 and on again at 25.

## Ruled out

Each of these was measured on the B300 and changed nothing:

- **Three capture warmups** instead of the default zero. Not a first-call initialisation baked in at
  capture time.
- **A restricted capture set.** Bucket 16 is not wrong because of what was captured before it.
- **Reversed run order**, and **a fresh engine running only the failing length**. Both return
  bit-identical *wrong* numbers, so nothing stale is being read from a previous request.
- **Batch composition with graphs off.** Batching the same prompt alongside another request is
  bit-identical, so this is not plain batch-size sensitivity.
- **The MoE router GEMM pinned to `torch.mm`.** Same failing lengths, so the router dispatch tier
  (`ll_bf16` at `M<=16`, cuBLAS above) is not the cause on its own.

## Controls

Same box, same script, the two dense Gemma 4 checkpoints — same hybrid sliding/full attention layout,
no MoE block:

| model | worst delta | argmax flips | next token |
| --- | --- | --- | --- |
| `gemma-4-12B-it` | 0.000, bit-identical | 0 of 13 | unchanged |
| `gemma-4-31B` | 0.249 | 1 of 14, at a margin of exactly 0.000 | unchanged |
| `gemma-4-26B-A4B-it` | 6.773 | 2 of 12 | 992 → 9079 |

A quarter of a nat that reorders an exact tie is the ordinary picture. This rules out "capability 10.x
perturbs everything" and "Gemma 4 numerics", and leaves the MoE path.

## Not a near-tie artifact

Worth stating because it is the natural first objection, and because a reviewer raised it upstream.
The argmax flips do sit on narrow margins — 0.250 and 0.125 — which is exactly what a small
perturbation on a near-tie predicts. The size of the perturbation is the anomaly: the largest delta
that did *not* move an argmax was 6.773 nats at a position whose eager margin is 2.375. A near-tie
story explains which positions flip; it does not explain a multi-nat shift on a wide-margin position,
and it does not explain bitwise agreement at other lengths on the same box.

## Still open

`VLLM_BATCH_INVARIANT=1` makes every length bit-identical. The overrides that flag installs are
fixed-reduction GEMMs and softmaxes (`aten::mm`, `addmm`, `matmul`, `linear`, the softmaxes,
`mean.dim`), so a reduction-order dependence is the natural first place to look — but the flag may
gate other kernel choices too, so that is a lead rather than a conclusion.

Which kernel stops being padding-invariant is not established. It is not FlashAttention, which was
the first guess: on this checkpoint the B300 logs FA4 disabled at both head sizes (256 temporarily,
512 for TMEM capacity) and settles on TRITON_ATTN with the TRITON MoE backend.

## What this repo does about it

- `comparison/engine_bugs.py` carries the row, scoped to capability 10.x via `capabilities=("10",)`,
  so the annotation does not appear on boxes where the cell is measured clean.
- `interp_engine/vllm_capture/static.py::sm100_cudagraph_refusal_reason` refuses this checkpoint on
  10.x under a static set, naming the flag and the issue. A static set requires the same graph replay
  with torch.compile off, so its taps would report vLLM's wrong forward as an interp result.
