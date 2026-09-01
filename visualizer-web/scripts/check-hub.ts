/**
 * The Hub client, checked against the Hub — and against the Python it is a port of.
 *
 * `lib/hub.ts` exists to be exact about weight bytes, and every rung of its ladder depends on a
 * response shape the Hub does not promise us. So rather than trusting it, this resolves real repos
 * and asserts the figures, including the ones that took measuring to get right:
 *
 * - a plain bf16 repo, where the index and the file sizes must agree
 * - `openai/gpt-oss-20b`, the variant-shard and silent-dequantization case, whose dequantized size
 *   must come out at the documented 41.18 GiB rather than the 12.82 GiB it occupies packed
 * - a gated repo with no token, which must still size exactly while reporting no trunk dims
 *
 * The expected numbers are `interp_engine.memory`'s, so a divergence between the two
 * implementations of this ladder fails here rather than in a fit somebody trusted.
 *
 * Usage:
 *   npx tsx scripts/check-hub.ts
 */

import { bytesForLoad, resolveModel, type ModelMemoryFacts } from "@/lib/hub";

const GIB = 1024 ** 3;

interface Expectation {
  id: string;
  /** Rung that must answer. A change here means the ladder shifted under us. */
  source: ModelMemoryFacts["weights"]["source"];
  onDiskGib: number;
  /** Logical parameters, matching `memory.logical_param_count`. */
  paramCount: number;
  /**
   * The scheme this client resolves. It may differ in *name* from the Python's, which is fine and
   * deliberate: the Python reaches gpt-oss-20b through header inference and lands on `int4`, while
   * this reads `config.json` and gets the checkpoint's own `mxfp4`. Both are labels for the same
   * 0.5 bytes per parameter, which is what the figures below actually depend on.
   */
  quantMethod: string;
  /** `bytes_for_load("bfloat16", dequantizes=True)`: what transformers expands the weights to. */
  dequantizedGib: number;
  /** `bytes_for_load("bfloat16", dequantizes=False)`: what vLLM holds, which keeps packing. */
  vllmGib: number;
  dims: { nLayers: number; dModel: number; nKvHeads: number; headDim: number } | null;
  why: string;
}

/**
 * Figures verified against `interp_engine.memory` on the same repos.
 *
 * One divergence is expected and is not a bug: the Python found `google/gemma-3-12b-pt`'s index and
 * config because `hf_hub_download` tries `local_files_only=True` first and this machine had them
 * cached from an authenticated run. Unauthenticated, both files are 401 — which is the path a
 * browser is always on, and the one asserted here.
 */
const CASES: Expectation[] = [
  {
    id: "openai-community/gpt2",
    source: "file-sizes",
    onDiskGib: 0.5105,
    paramCount: 137_022_720,
    quantMethod: "",
    // Stored fp32, so asking for bf16 *halves* it. The same arithmetic that catches eager doubling
    // a bf16 checkpoint catches this, which is why the term is keyed on load dtype either way.
    dequantizedGib: 0.2552,
    vllmGib: 0.2552,
    dims: { nLayers: 12, dModel: 768, nKvHeads: 12, headDim: 64 },
    why: "a single-file repo with no index, so the file sizes have to carry it",
  },
  {
    id: "openai/gpt-oss-20b",
    source: "safetensors-index",
    onDiskGib: 12.8162,
    paramCount: 22_109_150_784,
    quantMethod: "mxfp4",
    dequantizedGib: 41.1815,
    vllmGib: 12.8162,
    dims: { nLayers: 24, dModel: 2880, nKvHeads: 8, headDim: 64 },
    why: "variant shards under original/, and the silent-dequantization trap this module exists for",
  },
  {
    id: "google/gemma-3-12b-pt",
    source: "file-sizes",
    onDiskGib: 22.7007,
    paramCount: 12_187_325_040,
    quantMethod: "",
    dequantizedGib: 22.7007,
    vllmGib: 22.7007,
    dims: null,
    why: "gated: weights must resolve with no token, and the dims must be absent rather than zero",
  },
];

let failures = 0;

/** Four decimals for a GiB figure, exact for a count, so a diff is readable either way. */
function show(value: unknown): string {
  if (typeof value !== "number") return JSON.stringify(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

function check(label: string, actual: unknown, expected: unknown, tolerance = 0): void {
  const ok =
    typeof actual === "number" && typeof expected === "number"
      ? Math.abs(actual - expected) <= tolerance
      : actual === expected;
  if (ok) {
    console.log(`    ok    ${label}: ${show(actual)}`);
    return;
  }
  failures += 1;
  console.log(`    FAIL  ${label}: got ${show(actual)}, expected ${show(expected)}`);
}

async function main(): Promise<void> {
  if (process.env.HF_TOKEN) {
    // The gated case asserts the no-token path. An ambient token would make it pass for the wrong
    // reason, so refuse rather than quietly test something else.
    console.error("HF_TOKEN is set. Unset it: this script checks the unauthenticated path.");
    process.exit(2);
  }

  for (const expected of CASES) {
    console.log(`\n${expected.id}`);
    console.log(`  ${expected.why}`);
    const facts = await resolveModel(expected.id);

    check("source", facts.weights.source, expected.source);
    check("on-disk GiB", facts.weights.onDiskBytes / GIB, expected.onDiskGib, 0.001);
    check("quant method", facts.weights.quantMethod, expected.quantMethod);
    // 0.5%, because this parses the safetensors headers itself where the Python uses
    // huggingface_hub's reader. An exact match is the goal; a rounding difference is not a bug.
    check(
      "logical params",
      facts.weights.paramCount,
      expected.paramCount,
      Math.round(expected.paramCount * 0.005),
    );
    check(
      "dequantized GiB (bf16, transformers)",
      bytesForLoad(facts.weights, "bfloat16") / GIB,
      expected.dequantizedGib,
      0.001,
    );
    // A quantized checkpoint stays packed under vLLM whatever dtype is asked for, and an
    // unquantized one does not care about the flag. Four live Neuronpedia pods depend on the first
    // half of that, so both are asserted rather than assumed.
    check(
      "vLLM GiB (bf16, packed)",
      bytesForLoad(facts.weights, "bfloat16", { dequantizes: false }) / GIB,
      expected.vllmGib,
      0.001,
    );

    check("trunk dims known", facts.trunkDimsKnown, expected.dims !== null);
    if (expected.dims) {
      check("layers", facts.nLayers, expected.dims.nLayers);
      check("d_model", facts.dModel, expected.dims.dModel);
      check("kv heads", facts.nKvHeads, expected.dims.nKvHeads);
      check("head dim", facts.headDim, expected.dims.headDim);
    }
    for (const note of facts.notes) console.log(`    note  ${note}`);
  }

  console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
  process.exit(failures ? 1 : 0);
}

void main();
