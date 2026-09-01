/**
 * Does `lib/size.ts` still agree with `interp_engine.memory`?
 *
 * The browser sizer holds a port of the engine's memory arithmetic, for the latency reason
 * `lib/size.ts` opens with. A port is only worth having while it is provably the same port, so this
 * prices the same models through both implementations and compares them term by term: every option
 * `fit.py` reports for a model must appear here on the same card with the same count, the same
 * derived utilization and context, the same per-term bytes and the same KV capacity.
 *
 * Run it with `make size-check`, or `npm run size:check` from this directory. It needs the repo's
 * Python environment (it shells out to `gpu-sizer/fit.py`) and a few KB off the Hub per model,
 * which is why it is not part of `viz-check`.
 *
 * The models are chosen for the branches they exercise rather than for coverage: a small dense
 * checkpoint, a quantized one whose packed and dequantized sizes differ threefold, and a
 * hybrid-attention trunk where the sliding-window layers change both the KV arithmetic and the eager
 * attention term. `HF_TOKEN` is read from the environment when set, and the gated model is skipped
 * rather than failed when it is not.
 */

import { execFile } from "node:child_process";
import { promises as fs } from "node:fs";
import path, { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { resolveModel, type ModelMemoryFacts } from "../lib/hub";
import {
  BACKENDS,
  concurrentSequences,
  fitAcross,
  offeredStaticPoints,
  resolvedStaticPoints,
  totalBytes,
  type Backend,
  type FitResult,
} from "../lib/size";

const run = promisify(execFile);

const REPO = path.resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** The backends `fit.py` reports on by default, in its order. */
const REPORT_BACKENDS: Backend[] = ["vllm", "vllm-static", "eager"];

/** The context both sides are pinned to when the model's own advertised one cannot be read. */
const PINNED_CONTEXT = 8192;

/**
 * A static set worth comparing, run on `vllm-static` beside the default one.
 *
 * Every width rule in one set, deliberately, filtered per model to the ones that trunk offers:
 * `mlp_act` is the widest thing a tap can name and `router_logits` the narrowest, `attn` is three
 * buffers rather than one and read-only where the rest are read plus write, and `resid_post` is the
 * plain `d_model` case. A set of residual points alone would agree across a bug in any of those
 * rules, which is how the whole term came to be priced at one width for as long as it was.
 */
const STATIC_POINTS = ["attn", "mlp_act", "router_logits", "resid_post"];

const MODELS = [
  { id: "openai-community/gpt2", why: "small, dense, unquantized" },
  { id: "openai/gpt-oss-20b", why: "MXFP4: packed and dequantized differ 3x" },
  {
    id: "google/gemma-3-12b-pt",
    why: "hybrid attention, and gated",
    gated: true,
  },
  {
    id: "nvidia/Llama-3.3-70B-Instruct-FP4",
    why: "NVFP4 weights with an FP8 KV cache, declared only in hf_quant_config.json -- the pair that has to be read from a sidecar, and the one this check was blind to while both sides were wrong in the same direction",
  },
  {
    id: "deepseek-ai/DeepSeek-V4-Flash",
    why: "mixed precision: fp4 routed experts under an fp8 scheme, MLA, and ue8m0 scale tensors. The two sides reach its parameter count from opposite metadata -- headers here, the Hub's aggregate there -- and drifted 1.84x apart unnoticed while it was absent from this list",
  },
];

interface PyTerm {
  name: string;
  bytes: number;
  side: string;
}

interface PyOption {
  backend: string;
  gpu: string;
  num_gpus: number;
  dtype: string;
  gpu_memory_utilization: number;
  max_model_len: number;
  max_num_batched_tokens: number;
  static_points: string[];
  estimated_bytes: number;
  headroom_bytes: number;
  kv_capacity_tokens: number;
  concurrent_sequences: number;
  terms: PyTerm[];
}

interface PyReport {
  model_id: string;
  parameters: number;
  on_disk_bytes: number;
  quant_method: string;
  options: PyOption[];
}

let failures = 0;
let checks = 0;

function check(label: string, got: unknown, want: unknown): void {
  checks += 1;
  const same = JSON.stringify(got) === JSON.stringify(want);
  if (!same) {
    failures += 1;
    console.log(`  FAIL ${label}`);
    console.log(`       python: ${JSON.stringify(want)}`);
    console.log(`       ts:     ${JSON.stringify(got)}`);
  }
}

/**
 * The one figure compared with a tolerance rather than exactly, and the bound is the point.
 *
 * The **logical** parameter count is the only quantity the two sides reach by genuinely different
 * routes: Python reads the shards' own headers through `huggingface_hub`, while the browser reads
 * the Hub's aggregated `safetensors.parameters` and only falls back to headers on a repo with
 * variant shards. The two disagree in kind as well as in detail — the aggregate is already unpacked
 * where the headers are not — so agreement here is corroboration rather than a tautology, and a
 * repo can hold a few small tensors one counts and the other does not. On
 * `nvidia/Llama-3.3-70B-Instruct-FP4` that is 160 BF16 elements plus 960 F32 out of 74.8 billion --
 * 1.5e-8 relative, and 2,240 bytes on the "if this dequantized" figure, which is the only number it
 * feeds.
 *
 * A tolerance is honest here and would not be anywhere else: every *memory* term is compared to the
 * byte, and this is a count neither side uses to size anything. The bound is four orders of magnitude
 * tighter than the smallest difference that could move a GiB, so a real divergence still fails.
 */
const PARAM_COUNT_TOLERANCE = 1e-6;

function checkParamCount(label: string, got: number, want: number): void {
  checks += 1;
  const drift = want ? Math.abs(got - want) / want : got === want ? 0 : 1;
  if (drift > PARAM_COUNT_TOLERANCE) {
    failures += 1;
    console.log(`  FAIL ${label} (drift ${drift.toExponential(2)})`);
    console.log(`       python: ${want}`);
    console.log(`       ts:     ${got}`);
  }
}

async function python(
  modelId: string,
  maxModelLen = 0,
  staticPoints: string[] = [],
): Promise<PyReport> {
  // `--json` redirects the human table away from stdout rather than interleaving it, so what comes
  // back is the report and nothing else. The warning about a config-less repo goes to stderr.
  const { stdout } = await run(
    "uv",
    [
      "run",
      "python",
      "gpu-sizer/fit.py",
      modelId,
      "--json",
      ...(maxModelLen ? ["--max-model-len", String(maxModelLen)] : []),
      ...staticPoints.flatMap((point) => ["--static-point", point]),
    ],
    { cwd: REPO, maxBuffer: 64 * 1024 * 1024 },
  );
  return JSON.parse(stdout) as PyReport;
}

/**
 * The dtype `fit.py` picks when the caller names none, which the TS has to be given explicitly:
 * `auto` for a quantized checkpoint so it is priced as served, bf16 otherwise.
 */
function defaultDtype(facts: ModelMemoryFacts): string {
  return facts.weights.quantMethod ? "auto" : "bfloat16";
}

function compare(
  facts: ModelMemoryFacts,
  backend: Backend,
  ours: FitResult[],
  theirs: PyOption[],
  label = "",
): void {
  const modelId = facts.modelId + label;
  check(
    `${modelId} ${backend}: number of fitting options`,
    ours.length,
    theirs.length,
  );

  for (const want of theirs) {
    const got = ours.find(
      (option) =>
        option.gpu.name === want.gpu && option.count === want.num_gpus,
    );
    if (!got) {
      failures += 1;
      checks += 1;
      console.log(
        `  FAIL ${modelId} ${backend}: python fits ${want.gpu} x${want.num_gpus}, ts finds no fit`,
      );
      continue;
    }
    const where = `${modelId} ${backend} ${want.gpu} x${want.num_gpus}`;
    const est = got.estimate;

    check(`${where}: dtype`, est.spec.dtype, want.dtype);
    check(
      `${where}: utilization`,
      est.spec.gpuMemoryUtilization,
      want.gpu_memory_utilization,
    );
    check(`${where}: max_model_len`, est.spec.maxModelLen, want.max_model_len);
    check(
      `${where}: max_num_batched_tokens`,
      est.spec.maxNumBatchedTokens,
      want.max_num_batched_tokens,
    );
    check(
      `${where}: static points`,
      resolvedStaticPoints(est.spec, facts),
      want.static_points,
    );
    check(`${where}: total bytes`, totalBytes(est), want.estimated_bytes);
    check(`${where}: headroom`, est.headroomBytes, want.headroom_bytes);
    check(
      `${where}: kv capacity`,
      est.kvCapacityTokens,
      want.kv_capacity_tokens,
    );
    check(
      `${where}: concurrent sequences`,
      concurrentSequences(est),
      want.concurrent_sequences,
    );
    // Term by term and in order, because a total that matches while the terms do not means two
    // errors that happen to cancel -- and the UI shows the terms, not just the total.
    check(
      `${where}: terms`,
      est.terms.map((term) => [term.name, term.bytes, term.side]),
      want.terms.map((term) => [term.name, term.bytes, term.side]),
    );
  }
}

async function main(): Promise<void> {
  const token = process.env.HF_TOKEN || (await envToken());
  if (token) console.log("using HF_TOKEN for gated repos\n");

  for (const model of MODELS) {
    if (model.gated && !token) {
      console.log(`${model.id}: skipped, gated and no HF_TOKEN\n`);
      continue;
    }
    console.log(`${model.id} (${model.why})`);

    const [report, facts] = await Promise.all([
      python(model.id),
      resolveModel(model.id, { token }),
    ]);

    // The facts first: an arithmetic comparison is meaningless if the two started from different
    // dimensions, and a mismatch here is a `lib/hub.ts` bug rather than a `lib/size.ts` one.
    checkParamCount(
      `${model.id}: parameters`,
      facts.weights.paramCount,
      report.parameters,
    );
    check(
      `${model.id}: on-disk bytes`,
      facts.weights.onDiskBytes,
      report.on_disk_bytes,
    );

    const dtype = defaultDtype(facts);

    // A family that keeps its advertised context in a model class rather than in `config.json` --
    // gemma-3 is the case -- leaves the browser with no context to step down from, while the Python
    // reads 131,072 off `Gemma3TextConfig`. That is a documented and unfixable difference in the
    // *input*: the Hub serves no file that carries the number, and the API extract and
    // `tokenizer_config.json` were both checked. So the context is pinned on both sides here, which
    // is what makes this a test of the arithmetic rather than of the guess. The UI's answer is to ask
    // for a context instead of defaulting one, and the assertion below is that the gap stays declared.
    const blind = facts.derivedDims.includes("max_position_embeddings");
    if (blind) {
      check(
        `${model.id}: declares max_position_embeddings as unread`,
        facts.derivedDims.includes("max_position_embeddings"),
        true,
      );
      console.log(
        `  note: advertised context is unreadable from the browser; pinning ${PINNED_CONTEXT} on both sides`,
      );
    }
    const maxModelLen = blind ? PINNED_CONTEXT : 0;

    for (const backend of REPORT_BACKENDS) {
      const ours = fitAcross(facts, { backend, dtype, maxModelLen, maxGpus: 8 });
      const theirs = (
        maxModelLen ? await python(model.id, maxModelLen) : report
      ).options.filter((option) => option.backend === backend);
      compare(facts, backend, ours, theirs);
    }

    // The named-point arithmetic, which the loop above never reaches: with no set given, both sides
    // take the `"auto"` branch, and the per-point widths could be wrong on both sides in different
    // ways without a single check failing.
    const points = STATIC_POINTS.filter((point) =>
      offeredStaticPoints(facts).includes(point),
    );
    const staticOurs = fitAcross(facts, {
      backend: "vllm-static",
      dtype,
      maxModelLen,
      staticPoints: points,
      maxGpus: 8,
    });
    const staticTheirs = (
      await python(model.id, maxModelLen, points)
    ).options.filter((option) => option.backend === "vllm-static");
    compare(facts, "vllm-static", staticOurs, staticTheirs, ` [${points}]`);

    console.log();
  }

  console.log(
    failures
      ? `${failures} of ${checks} checks failed`
      : `all ${checks} checks passed`,
  );
  if (failures) process.exitCode = 1;
}

/** The repo keeps a token in `.env` for the Python side; reuse it rather than ask for a second. */
async function envToken(): Promise<string> {
  try {
    const text = await fs.readFile(path.join(REPO, ".env"), "utf8");
    return /^HF_TOKEN=(.*)$/m.exec(text)?.[1]?.trim() ?? "";
  } catch {
    return "";
  }
}

// Sanity: the port and the CLI have to at least agree on what a backend is called.
for (const backend of REPORT_BACKENDS) {
  if (!BACKENDS.includes(backend)) {
    throw new Error(`${backend} is not a backend lib/size.ts knows`);
  }
}

void main();
