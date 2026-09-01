/**
 * What a configuration costs on a card, and which cards will run it.
 *
 * A port of `interp_engine.memory`'s `estimate`, `fit` and `fit_across`, and the only reason a port
 * exists is latency: the finder reprices on every keystroke and every slider drag, and a round trip
 * to Python for each would make the VRAM bar lag the control that moves it. The numbers it spends
 * are not duplicated — `data/gpus.generated.ts` is written from the Python catalog and calibration —
 * so what lives here is arithmetic and nothing else.
 *
 * **The arithmetic is checked, not trusted.** `scripts/check-size.ts` prices the same matrix of
 * models, cards and backends through this file and through `gpu-sizer/fit.py`, and fails on any
 * disagreement. Run it with `make size-check` after touching anything below. Every `Math.trunc` and
 * `Math.floor` here mirrors an `int()` or `//` in the Python, and the expression order is kept the
 * same on purpose: both languages are IEEE-754 doubles, so identical operations in an identical
 * order give identical results, and a "harmless" algebraic simplification is how a port starts
 * drifting in the last digit.
 *
 * The one thing worth understanding before reading on is the utilization line, which is what makes
 * the vLLM arm two budgets rather than one. vLLM claims `card x gpu_memory_utilization` as a pool
 * and fills it; the weights, the CUDA context and the KV cache come out of that pool, while the
 * warmup overshoot, fragmentation and anything the caller allocates after startup have to fit in the
 * slice *outside* it. Both have to hold. They fail differently, and they are fixed by moving
 * utilization in opposite directions, which is why a single total against the card would be useless
 * advice even when it is an accurate number.
 */

import {
  CALIBRATION,
  GPUS,
  VERIFIED_RUNS,
  type Gpu,
  type VerifiedRun,
} from "@/data/gpus.generated";
import {
  bytesForLoad,
  dtypeBytes,
  dtypeBytesOrNull,
  fullAttentionLayers,
  kvCacheWidth,
  kvCachingLayers,
  recurrentLayers,
  type ModelMemoryFacts,
} from "@/lib/hub";

export const GIB = 1024 ** 3;

export type Backend = "vllm" | "vllm-static" | "vllm-generate" | "eager";

/** In the order the finder offers them: the default first, then the two that cost more. */
export const BACKENDS: Backend[] = [
  "vllm",
  "vllm-static",
  "vllm-generate",
  "eager",
];

const VLLM_BACKENDS: Backend[] = ["vllm", "vllm-static", "vllm-generate"];

export function isVllm(backend: Backend): boolean {
  return VLLM_BACKENDS.includes(backend);
}

/**
 * Batch widths `fit` steps `max_num_batched_tokens` down through, and contexts it steps
 * `max_model_len` down through. Both descend, and both are searched context-major: too small a
 * context refuses requests outright, while a narrow prefill batch only makes prefill slower.
 */
const CAPTURE_SIZES = [16384, 8192, 4096, 2048, 1024];
const CONTEXT_LADDER = [131072, 65536, 32768, 16384, 8192, 4096, 2048];

// ------------------------------------------------------------------ reservations

/**
 * VRAM the engine must not touch, split by how it scales with tensor parallelism.
 *
 * The split is the point: a number replicated on every rank and a number that exists once behave
 * completely differently at TP > 1. `beforeEngine` decides which side of the utilization line it
 * lands on, and it is easy to get backwards — memory allocated *before* startup is charged against
 * the pool and just shrinks the KV cache, while memory allocated *after* sits on top of a pool vLLM
 * has already filled and eats the margin instead. The default is the latter, because loading the
 * engine and then loading your own weights is both the common case and the dangerous one.
 */
export interface Reservations {
  /** Replicated on every GPU. A Jacobian lens is the case that matters. */
  perRankBytes: number;
  /** Exists once, on rank 0. A preloaded SAE cache in the serving process is the usual one. */
  hostBytes: number;
  /** Peak bytes concurrent requests need outside the pool, over and above the two above. */
  transientBytes: number;
  beforeEngine: boolean;
  note: string;
}

export function reservations(
  overrides: Partial<Reservations> = {},
): Reservations {
  return {
    perRankBytes: 0,
    hostBytes: 0,
    transientBytes: 0,
    beforeEngine: false,
    note: "",
    ...overrides,
  };
}

function forRank(res: Reservations, rank = 0): number {
  return (
    res.perRankBytes + res.transientBytes + (rank === 0 ? res.hostBytes : 0)
  );
}

/**
 * Reservations sized for a Jacobian lens read-out: `n_layers x d_model^2 x itemsize` per rank.
 *
 * Per rank rather than sharded, because the read-out holds the whole matrix on each worker device.
 * On a 70B in fp32 that is ~10 GiB a card, and a utilization derived without it hands vLLM memory
 * the lens is about to want.
 */
export function jacobianLens(
  facts: ModelMemoryFacts,
  dtype = "float32",
): Reservations {
  const width = dtypeBytes(dtype);
  const perRank = Math.trunc(
    Math.max(facts.nLayers, 1) * Math.max(facts.dModel, 1) ** 2 * width,
  );
  return reservations({
    perRankBytes: perRank,
    note: `jacobian lens: ${facts.nLayers} layers x ${facts.dModel}^2 x ${width}B per rank`,
  });
}

// ---------------------------------------------------------------- workload specs

/**
 * A configuration to price: the arguments a caller would pass, plus the load they intend.
 *
 * Field names match `load_model` and the vLLM engine arguments they become, so an estimate can be
 * turned back into runnable code without a translation table.
 */
export interface WorkloadSpec {
  backend: Backend;
  /** The dtype weights load at. `"auto"` means as stored. */
  dtype: string;
  /** `"auto"` follows the model dtype. */
  kvCacheDtype: string;
  maxModelLen: number;
  maxNumBatchedTokens: number;
  maxNumSeqs: number;
  gpuMemoryUtilization: number;
  numGpus: number;
  /**
   * Static tap sites, read and write together. `0` resolves to `2 * n_layers` on `vllm-static`. A
   * raw count prices every site at the residual width; name the points to price them at theirs.
   */
  staticSites: number;
  /** Static tap points to declare at every layer. Empty is `"auto"`. `staticSites` overrides it. */
  staticPoints: string[];
  enforceEager: boolean | null;
  batchSize: number;
  /** Eager only: the prompt being priced. This is what the quadratic terms scale with. */
  seqLen: number;
  requiresGrad: boolean;
  attnImplementation: string;
  nCapturePoints: number;
}

export function workload(overrides: Partial<WorkloadSpec> = {}): WorkloadSpec {
  return {
    backend: "vllm",
    dtype: "auto",
    kvCacheDtype: "auto",
    maxModelLen: 0,
    maxNumBatchedTokens: 0,
    maxNumSeqs: 0,
    gpuMemoryUtilization: 0,
    numGpus: 1,
    staticSites: 0,
    staticPoints: [],
    enforceEager: null,
    batchSize: 1,
    seqLen: 0,
    requiresGrad: false,
    attnImplementation: "",
    nCapturePoints: 0,
    ...overrides,
  };
}

/**
 * Whether a CUDA graph pool will be captured.
 *
 * `vllm` runs `enforce_eager=True` by default, because graph replay skips the Python forward the
 * capture hooks live on. The two graph backends force it False.
 */
export function graphsOn(spec: WorkloadSpec): boolean {
  if (spec.enforceEager !== null) return !spec.enforceEager;
  return spec.backend === "vllm-static" || spec.backend === "vllm-generate";
}

/**
 * The static tap points this sizer can price, in forward order, each with the trunk dimension its
 * buffer is wide along.
 *
 * Not the whole vocabulary — the engine serves about twenty-five names, and the ones missing here
 * are missing because their width is not on `ModelMemoryFacts`: the QK-norm points need a per-head
 * shape that depends on how the norm was written, and `value` is the fused qkv output on most
 * families rather than the `d_model` its name suggests. Pricing one of those at `d_model` would be a
 * guess wearing a number's clothes.
 */
const STATIC_POINT_WIDTHS: Record<string, string> = {
  resid_pre: "d_model",
  resid_streams: "streams",
  attn: "qkv",
  z: "heads",
  attn_out: "d_model",
  mlp_act: "neurons",
  router_logits: "experts",
  mlp_out: "d_model",
  resid_post: "d_model",
};

/**
 * Points that allocate a read buffer and nothing else. `attn` is refused as a write outright: it is
 * a copy of the kernel's q/k/v, and adding to it means nothing.
 *
 * Everything else is priced with a write buffer beside the read, which is a choice worth naming.
 * `"auto"` declares a write at every read and that is what doubles the default set — but the engine
 * leaves an *explicit* list read-only unless `static_writes` restates it. Pricing a named set the
 * expensive way is deliberate: a tap set is asked for in order to steer at the same addresses, and
 * quoting the read-only figure to someone who then adds a steer is the direction that OOMs. `snippet`
 * emits the matching `static_writes` so the code and this figure describe the same pod.
 */
const STATIC_CAPTURE_ONLY = new Set(["attn"]);

/**
 * Points a hyper-connection trunk refuses, mirroring the engine's `_SINGLE_STREAM_POINTS`: with
 * several streams in flight there is no single residual vector for these to name.
 */
const SINGLE_STREAM_ONLY = new Set(["resid_pre", "resid_mid", "resid_post"]);

/**
 * The points that can be declared on *this* trunk, in forward order.
 *
 * Two trunk properties gate the list, and both are refusals in the engine rather than tidying here.
 * A hyper-connection block has no single residual vector, so the three names for one are out and
 * `resid_streams` is in. And a sparse block's MLP is a fused kernel with no activation tensor to tap
 * and a router that has one, so `mlp_act` and `router_logits` trade places.
 */
export function offeredStaticPoints(facts: ModelMemoryFacts): string[] {
  const streams = Math.max(facts.nResidualStreams, 1) > 1;
  const moe = facts.nExperts > 0;
  return Object.keys(STATIC_POINT_WIDTHS).filter((name) => {
    if (SINGLE_STREAM_ONLY.has(name)) return !streams;
    if (name === "resid_streams") return streams;
    if (name === "mlp_act") return !moe;
    if (name === "router_logits") return moe;
    return true;
  });
}

/** What `static_points="auto"` resolves to, which the trunk decides rather than the caller. */
export function defaultStaticPoint(facts: ModelMemoryFacts): string {
  return Math.max(facts.nResidualStreams, 1) > 1 ? "resid_streams" : "resid_post";
}

/**
 * Buffers one point allocates per layer.
 *
 * Three for `attn`, which is not a tensor but a request for the kernel's q, k and v; two for
 * everything else, a read and the write `"auto"` declares beside it.
 */
export function staticPointBuffers(point: string): number {
  if (point === "attn") return 3;
  return STATIC_CAPTURE_ONLY.has(point) ? 1 : 2;
}

/**
 * Elements per token that one point costs at one layer, across every buffer it allocates.
 *
 * The widths are the engine's: the residual and sublayer points are `d_model`, `z` is the o_proj's
 * input and so head-shaped, `mlp_act` is the down_proj's input and so `intermediate_size`,
 * `router_logits` is the gate's output and so as wide as the expert bank, and `attn` is three buffers
 * at `n_heads` and twice `n_kv_heads` head widths rather than one at `d_model`.
 *
 * Pricing every point at `d_model` is what this exists to stop being: on Qwen3-32B `mlp_act` is 5.0x
 * a residual tap and `attn` is 0.6x, so a set of three is anywhere from 1.6x to 11x the default
 * depending on which three.
 */
export function staticPointElements(
  point: string,
  facts: ModelMemoryFacts,
): number {
  const dModel = Math.max(facts.dModel, 1);
  const headDim = Math.max(facts.headDim, 1);
  const kind = STATIC_POINT_WIDTHS[point];
  let width = dModel;
  if (kind === "streams") {
    width = dModel * Math.max(facts.nResidualStreams, 1);
  } else if (kind === "heads") {
    width = Math.max(facts.nHeads, 1) * headDim;
  } else if (kind === "neurons") {
    width = Math.max(facts.intermediateSize || 4 * dModel, 1);
  } else if (kind === "experts") {
    width = Math.max(facts.nExperts, 1);
  } else if (kind === "qkv") {
    // One buffer per role, so the three are summed rather than multiplied by `staticPointBuffers`.
    width =
      (Math.max(facts.nHeads, 1) + 2 * Math.max(facts.nKvHeads, 1)) * headDim;
  }
  return STATIC_CAPTURE_ONLY.has(point) ? width : width * 2;
}

/**
 * The points a spec will really declare, at every layer.
 *
 * A named point this trunk refuses is dropped rather than refused, because the caller who chose it
 * was looking at a different model a moment ago: switching from Qwen3 to DeepSeek-V4 should not
 * error, it should fall back to the stack that trunk does carry.
 *
 * Repeats are dropped too, and that one is a correctness fix rather than tidying. A tap is per
 * `(point, layer)`, so declaring `mlp_act` twice declares the same buffers once — but
 * `resolvedStaticSites` and `staticElements` both sum over this list, so a repeat charged the model
 * twice for memory the engine allocates once. `Set` keeps first insertion order, so the points stay
 * in the forward order `snippet` prints them in.
 */
export function resolvedStaticPoints(
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): string[] {
  if (spec.backend !== "vllm-static") return [];
  const offered = new Set(offeredStaticPoints(facts));
  const chosen = [
    ...new Set(spec.staticPoints.filter((point) => offered.has(point))),
  ];
  return chosen.length ? chosen : [defaultStaticPoint(facts)];
}

/** Static buffers a spec will actually allocate, reads and writes together. */
export function resolvedStaticSites(
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): number {
  if (spec.backend !== "vllm-static") return 0;
  if (spec.staticSites) return spec.staticSites;
  const points = resolvedStaticPoints(spec, facts);
  return (
    Math.max(facts.nLayers, 1) *
    points.reduce((total, point) => total + staticPointBuffers(point), 0)
  );
}

/**
 * Elements per token across every static buffer, which is what the bytes scale with.
 *
 * Not `sites * width`: a named set has no single width, and averaging one is how a set carrying
 * `mlp_act` comes to look as cheap as a set of residual taps.
 */
export function staticElements(
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): number {
  if (spec.backend !== "vllm-static") return 0;
  if (spec.staticSites) return spec.staticSites * staticWidth(facts);
  const points = resolvedStaticPoints(spec, facts);
  return (
    Math.max(facts.nLayers, 1) *
    points.reduce(
      (total, point) => total + staticPointElements(point, facts),
      0,
    )
  );
}

/**
 * The engine's own defaults, filled in so a spec prices what would really happen.
 *
 * The vLLM engine arguments are left at zero on the eager backend rather than defaulted, so a spec
 * turned back into code does not carry settings that backend has never heard of.
 */
export function withDefaults(
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): WorkloadSpec {
  const maxLen = spec.maxModelLen || facts.maxPositionEmbeddings || 4096;
  if (spec.backend === "eager") {
    return {
      ...spec,
      maxModelLen: maxLen,
      // On eager there is no paged cache, so `maxModelLen` is simply the longest prompt -- and the
      // prompt is what the quadratic attention term and the logits scale with. Pricing a 512-token
      // forward for a caller who said 8192 would miss the entire risk.
      seqLen: spec.seqLen || maxLen,
      dtype: spec.dtype || "float32",
      attnImplementation: spec.attnImplementation || "eager",
    };
  }
  return {
    ...spec,
    maxModelLen: maxLen,
    maxNumBatchedTokens:
      spec.maxNumBatchedTokens ||
      (spec.backend === "vllm-static" ? 8192 : 2048),
    maxNumSeqs: spec.maxNumSeqs || 256,
    gpuMemoryUtilization: spec.gpuMemoryUtilization || 0.9,
    dtype: spec.dtype || "auto",
  };
}

// -------------------------------------------------------------------- the terms

/** Which budget a term is charged against. `"eager"` is the whole card, there being no pool. */
export type Side = "pool" | "outside" | "eager";

export interface MemoryTerm {
  name: string;
  bytes: number;
  side: Side;
  note: string;
}

export interface MemoryEstimate {
  spec: WorkloadSpec;
  gpu: Gpu;
  facts: ModelMemoryFacts;
  terms: MemoryTerm[];
  fits: boolean;
  /** Bytes left over; negative is the shortfall. The tighter of the two budgets, on vLLM. */
  headroomBytes: number;
  /** Kept separately because the two fail differently and are fixed in opposite directions. */
  poolHeadroomBytes: number;
  outsideHeadroomBytes: number;
  /** What the pool has to spend, i.e. `card x utilization`. Zero on eager. */
  poolBytes: number;
  /** Context the KV cache the pool leaves will hold, in tokens. Zero on eager. */
  kvCapacityTokens: number;
  advice: string[];
  warnings: string[];
  /** `"measured"` when a verification record backs this exact spec, else `"estimated"`. */
  evidence: "measured" | "estimated";
}

export function totalBytes(est: MemoryEstimate): number {
  return est.terms.reduce((sum, term) => sum + term.bytes, 0);
}

export function sideBytes(est: MemoryEstimate, side: Side): number {
  return est.terms
    .filter((term) => term.side === side)
    .reduce((sum, term) => sum + term.bytes, 0);
}

/** Full-length sequences the KV cache holds at once. */
export function concurrentSequences(est: MemoryEstimate): number {
  if (!est.spec.maxModelLen) return 0;
  return Math.floor(est.kvCapacityTokens / est.spec.maxModelLen);
}

/**
 * KV bytes for one token of context, across every layer.
 *
 * The flat, model-wide figure: every layer charged for the full context. **Zero means unknown, not
 * free** — when the config could not be read there is no honest number, and callers must treat 0 as
 * "cannot size this" rather than adding it to a total.
 */
/**
 * How many ways tensor parallelism actually divides the KV cache.
 *
 * **Not the rank count**, which is the assumption to get rid of. vLLM shards the cache by *KV head*,
 * so how far it divides is a property of the attention shape rather than of the machine:
 * Llama-3.3-70B's 8 KV heads go 2-per-rank at TP=4 and the cache really is a quarter on each card,
 * while a DeepSeek MLA trunk caches a single 512-wide latent head, which cannot be cut at all — vLLM
 * replicates it, and four cards hold four copies of the same cache.
 *
 * `min` covers both ends, including the case past the second one: vLLM pads a head count up to the
 * rank count by duplicating heads, so 8 heads across 16 ranks still costs what 8 ranks cost rather
 * than half as much.
 *
 * One when the head dims were never read, matching `kvCacheWidth`'s `2 * d_model` fallback. That
 * figure is a whole-model worst case with no head structure behind it, and dividing a worst case by
 * a rank count would invent exactly the precision the fallback exists because we lack.
 */
export function kvShards(facts: ModelMemoryFacts, numGpus: number): number {
  const tp = Math.max(Math.trunc(numGpus), 1);
  if (facts.nKvHeads <= 0 || facts.headDim <= 0) return 1;
  return Math.min(tp, facts.nKvHeads);
}

/**
 * KV bytes for one token of context, across every layer that caches tokens.
 *
 * Recurrent layers are excluded, because they allocate no cache to hold: what they keep is a
 * fixed-size state per sequence, sized by `max_num_seqs` rather than by context. **That state pool
 * is not priced anywhere in this module**, so on a hybrid-linear trunk the capacity this feeds is an
 * upper bound by an unmeasured margin. `estimate` warns when that is the case.
 *
 * Zero means unknown, not free — see `trunkDimsKnown`, and the all-recurrent trunk that reaches the
 * same zero by having no cache at all.
 */
export function kvBytesPerToken(
  facts: ModelMemoryFacts,
  kvDtype = "auto",
  modelDtype = "bfloat16",
): number {
  if (!facts.trunkDimsKnown) return 0;
  let width: number;
  if (kvDtype && kvDtype !== "auto") {
    width = dtypeBytes(kvDtype);
  } else {
    // `auto` means "whatever the engine will do", and for a checkpoint that ships a quantized KV
    // cache that is the checkpoint's own scheme rather than the model dtype — vLLM honours the
    // declaration whether or not the caller mentioned it. Narrower of the two wins, since an
    // explicit request cannot widen what the weights already store.
    const declared = dtypeBytesOrNull(facts.kvQuantAlgo);
    const model = dtypeBytes(modelDtype);
    width = declared === null ? model : Math.min(declared, model);
  }
  return kvCachingLayers(facts) * Math.max(kvCacheWidth(facts), 1) * width;
}

/**
 * KV bytes vLLM will actually spend to serve `maxModelLen` tokens.
 *
 * Sliding-window layers get no discount, and that is a measurement rather than a simplification: on
 * gemma-3-1b the windowed arithmetic came out 4.2x optimistic against what vLLM really built, because
 * the hybrid allocator pages whole blocks across both layer groups and reports capacity governed by
 * the full-attention group. The residual 8% is charged as `hybrid_kv_overhead` on a mixed trunk.
 *
 * A recurrent layer is a different case and *is* discounted, in {@link kvBytesPerToken} rather than
 * here: it requests no blocks from that allocator, so there is no group for it to be paged alongside
 * and the gemma-3 measurement says nothing about it. The multiplier still applies to such a trunk,
 * which is the conservative direction while no linear-attention trunk has been measured.
 */
export function kvBytesForContext(
  facts: ModelMemoryFacts,
  maxModelLen: number,
  kvDtype = "auto",
  modelDtype = "bfloat16",
): number {
  const flat = kvBytesPerToken(facts, kvDtype, modelDtype) * maxModelLen;
  return isHybridTrunk(facts) ? flat * CALIBRATION.hybrid_kv_overhead : flat;
}

/**
 * Whether some layers cache a window rather than the whole context.
 *
 * The per-layer table answers it outright when there is one. When there is not, a declared
 * `sliding_window` is enough: a trunk does not carry a window it never uses, and the Python reaches
 * the same conclusion from the layer pattern its model class fills in. Inferring it matters because
 * the overhead it selects makes the KV capacity *smaller* — leaving it off is the optimistic
 * direction, and 10% of optimism on a cache figure is a pod that accepts work it cannot hold.
 */
function isHybridTrunk(facts: ModelMemoryFacts): boolean {
  if (facts.layerTypes) return fullAttentionLayers(facts) < facts.nLayers;
  return facts.slidingWindow !== null;
}

/**
 * Elements per token per static site, for a set declared as a raw count rather than by name. A
 * hyper-connection trunk carries several streams.
 */
function staticWidth(facts: ModelMemoryFacts): number {
  return Math.max(facts.dModel, 1) * Math.max(facts.nResidualStreams, 1);
}

/**
 * Static tap buffer bytes from a per-token element count summed over every buffer.
 *
 * The general form of `staticBufferBytes`, which assumes one width for every site. A named point set
 * does not have one: `mlp_act` is `intermediate_size` wide where `resid_post` is `d_model`, so the
 * sum is taken before the multiply rather than after.
 */
export function staticTapBytes(
  elementsPerToken: number,
  maxN: number,
  elementBytes = 2,
): number {
  return (
    Math.trunc(elementsPerToken) * Math.trunc(maxN) * Math.trunc(elementBytes)
  );
}

/**
 * Static tap buffer bytes: one `maxN`-row buffer of `width` per site.
 *
 * Does **not** shard with tensor parallelism — a `d_model`-wide tap is replicated on every rank —
 * which is why a static set costs the same on eight cards as on one.
 */
export function staticBufferBytes(
  nSites: number,
  maxN: number,
  width: number,
  elementBytes = 2,
): number {
  return Math.trunc(nSites) * Math.trunc(maxN) * Math.trunc(width) * elementBytes;
}

/**
 * Peak activation bytes for one eager forward, per term, in the order they are charged.
 *
 * **Weights are not what OOMs the eager backend**, and this is the function that says why. Two terms
 * grow with the prompt rather than with the model: the logits, which materialize `[batch, seq, vocab]`
 * and are usually upcast to fp32 beside themselves, and the attention matrix, which is quadratic and
 * per layer under `attn_implementation="eager"`. On a 262k-vocab model at 8k tokens the logits pair
 * alone is ~12 GiB. `requiresGrad` is the third trap: the graph retains every layer's activations
 * rather than one layer's.
 *
 * `logitsFp32` defaults to true as the conservative reading. Whether a family upcasts is a per-family
 * matter and the one measurement on hand says Qwen3 does not, so this runs 1.2-1.8x pessimistic
 * where the upcast is absent — one rung of prompt length, against an OOM for the opposite error.
 */
export function eagerActivationBytes(
  facts: ModelMemoryFacts,
  {
    batchSize = 1,
    seqLen = 512,
    dtype = "float32",
    nCapturePoints = 0,
    requiresGrad = false,
    attnImplementation = "eager",
    logitsFp32 = true,
  }: {
    batchSize?: number;
    seqLen?: number;
    dtype?: string;
    nCapturePoints?: number;
    requiresGrad?: boolean;
    attnImplementation?: string;
    logitsFp32?: boolean;
  } = {},
): [string, number][] {
  const width = dtypeBytes(dtype);
  const tokens = Math.max(batchSize, 1) * Math.max(seqLen, 1);

  let logits = Math.trunc(tokens * Math.max(facts.vocabSize, 1) * width);
  if (logitsFp32 && width < 4) {
    // The upcast holds both copies at once; `logits.float()` is a new tensor.
    logits += Math.trunc(tokens * Math.max(facts.vocabSize, 1) * 4);
  }

  let attn = 0;
  if (attnImplementation.toLowerCase().includes("eager")) {
    // Scores plus the softmax result, and two layers' worth: a layer's matrix is freed only once the
    // next has allocated. A sliding-window layer attends over its window rather than the whole
    // prompt, so its matrix is `seq x window` -- on gemma-3 that is 40 layers of 48, and squaring the
    // wrong number at 32k tokens is the difference between 128 GiB and 8.
    const heads = Math.max(facts.nHeads, 1);
    const rows = Math.max(seqLen, 1);
    const full = fullAttentionLayers(facts);
    const windowed = Math.max(facts.nLayers - full, 0);
    const window = Math.min(facts.slidingWindow || rows, rows);
    const widest = full ? rows : window;
    const perLayer = Math.max(batchSize, 1) * heads * rows * widest * width;
    attn = Math.trunc(perLayer * 2 * 2);
    if (!full && !windowed) {
      attn = Math.trunc(
        Math.max(batchSize, 1) * heads * rows * rows * width * 2 * 2,
      );
    }
  }

  let hidden = Math.trunc(tokens * Math.max(facts.dModel, 1) * width);
  let mlp = Math.trunc(
    tokens *
      Math.max(facts.intermediateSize, 4 * Math.max(facts.dModel, 1)) *
      width *
      2,
  );
  const capture = Math.trunc(
    Math.max(nCapturePoints, 0) * tokens * Math.max(facts.dModel, 1) * width,
  );

  if (requiresGrad) {
    // Every layer's activations are retained rather than one layer's, which is why gradients plus
    // eager attention is the worst combination available.
    const depth = Math.max(facts.nLayers, 1);
    hidden *= depth;
    mlp *= depth;
    if (attn) attn = Math.trunc((attn / 2) * depth);
  }

  return [
    ["logits", logits],
    ["attention", attn],
    ["hidden_states", hidden],
    ["mlp_intermediate", mlp],
    ["capture_buffers", capture],
    ["workspace", Math.trunc(CALIBRATION.eager_workspace_gib * GIB)],
  ];
}

// ----------------------------------------------------------------- the estimate

export function supportsFp8(gpu: Gpu): boolean {
  return atLeast(gpu.computeCapability, [8, 9]);
}

export function supportsFp4(gpu: Gpu): boolean {
  return atLeast(gpu.computeCapability, [10, 0]);
}

/** The Triton MXFP4 path transformers needs to avoid dequantizing to bf16. */
export function supportsMxfp4Kernels(gpu: Gpu): boolean {
  return atLeast(gpu.computeCapability, [7, 5]);
}

function atLeast(have: [number, number], want: [number, number]): boolean {
  return have[0] !== want[0] ? have[0] > want[0] : have[1] >= want[1];
}

export function totalGib(gpu: Gpu): number {
  return Math.round((gpu.totalBytes / GIB) * 100) / 100;
}

const gib = (bytes: number, places = 1) => (bytes / GIB).toFixed(places);
const num = (value: number) => Math.round(value).toLocaleString("en-US");

/**
 * What a quantized checkpoint costs if its kernels are missing and it silently falls back to bf16.
 *
 * The gpt-oss / MXFP4 trap as a number rather than a warning: transformers needs `kernels`, Triton
 * and compute capability 7.5, and if any is missing it *warns* and expands the weights instead of
 * failing. Two bytes a logical parameter, which is why `paramCount` has to be logical rather than
 * stored — a packed repo reports containers.
 */
export function dequantizedBytes(facts: ModelMemoryFacts): number {
  if (!facts.weights.quantMethod) return 0;
  return Math.trunc(facts.weights.paramCount * 2);
}

/**
 * Price a configuration on a card, term by term.
 *
 * The vLLM arm splits every term across the utilization line; the eager arm has no such line, so it
 * charges everything against the card. Both add the same reservations, and both report the same
 * shape of answer.
 */
export function estimate(
  facts: ModelMemoryFacts,
  gpu: Gpu,
  rawSpec: WorkloadSpec,
  res: Reservations = reservations(),
): MemoryEstimate {
  const spec = withDefaults(rawSpec, facts);
  const tp = Math.max(Math.trunc(spec.numGpus), 1);
  const terms: MemoryTerm[] = [];
  const warnings: string[] = [];
  const advice: string[] = [];

  // Every row in `gpu-sizer/VERIFIED.md` was measured on one card. The single-GPU arithmetic is
  // calibrated against hardware; how it divides across ranks is not, and the two terms it is most
  // likely to be wrong about -- the weights, which TP shards unevenly, and the cache, which it
  // shards for some attention shapes and replicates for others -- are the two largest.
  if (tp > 1) {
    warnings.push(
      `${tp}-GPU figures are unverified: every configuration measured so far ran on a single card, so tensor parallelism here is arithmetic no hardware has checked`,
    );
  }

  // Only the eager backend expands a quantized checkpoint to the requested dtype; vLLM reads the
  // same argument as an activation dtype and serves the packed weights.
  const weightsTotal = bytesForLoad(facts.weights, spec.dtype, {
    dequantizes: spec.backend === "eager",
  });
  if (!weightsTotal) {
    warnings.push(
      "weight bytes are unknown, so every figure below is only the non-weight terms",
    );
  }
  if (facts.weights.quantMethod) {
    const dequantized = dequantizedBytes(facts);
    if (
      !supportsMxfp4Kernels(gpu) &&
      facts.weights.quantMethod.includes("fp4")
    ) {
      warnings.push(
        `${gpu.name} is compute ${gpu.computeCapability[0]}.${gpu.computeCapability[1]}, below the 7.5 the MXFP4 Triton path needs, so transformers will warn and dequantize: ${gib(dequantized)} GiB rather than ${gib(facts.weights.onDiskBytes)} GiB`,
      );
    } else if (spec.backend === "eager" && dequantized > weightsTotal * 1.5) {
      // Only on the eager arm, because the accident this prices is a transformers one: the kernels
      // are missing, so the checkpoint is expanded on the way in. vLLM never takes that path --
      // which is why `weightsTotal` above is the packed size on that arm -- and a figure four times
      // the real one, attached to a configuration that cannot reach it, reads as a risk rather than
      // as the aside it would be.
      warnings.push(
        `quantized checkpoint (${facts.weights.quantMethod}): ${gib(weightsTotal)} GiB served natively, but ~${gib(dequantized)} GiB if the kernels are missing and transformers dequantizes silently`,
      );
    }
    if (facts.weights.quantMethod.includes("fp8") && !supportsFp8(gpu)) {
      warnings.push(
        `${gpu.name} has no FP8 tensor cores, so an FP8 checkpoint runs emulated: correct, but slower than the same weights on Ada/Hopper or newer`,
      );
    }
  }

  if (spec.backend === "eager") {
    // One process, no pool: weights land wherever `device_map` puts them, and the activation peak
    // sits beside them on the same card.
    if (!facts.trunkDimsKnown) {
      warnings.push(
        `cannot size the activation peak for ${facts.modelId}: its config gave no dimensions, and on eager the prompt-driven terms usually exceed the weights. Only the weights below are real.`,
      );
    }
    const perCardWeights = Math.floor(weightsTotal / tp);
    terms.push({
      name: "weights",
      bytes: perCardWeights,
      side: "eager",
      note:
        `${(facts.weights.paramCount / 1e9).toFixed(1)}B params at ${spec.dtype}` +
        (tp > 1 ? `, spread over ${tp} GPUs` : "") +
        ` [${facts.weights.source}]`,
    });

    const activation = eagerActivationBytes(facts, {
      batchSize: spec.batchSize,
      seqLen: spec.seqLen,
      dtype:
        spec.dtype !== "auto"
          ? spec.dtype
          : facts.weights.storedDtype || "bfloat16",
      nCapturePoints: spec.nCapturePoints,
      requiresGrad: spec.requiresGrad,
      attnImplementation: spec.attnImplementation || "eager",
    });
    for (const [name, value] of activation) {
      if (value) {
        terms.push({
          name,
          bytes: value,
          side: "eager",
          note: eagerNote(name, spec, facts),
        });
      }
    }
    terms.push({
      name: "cuda_context",
      bytes: Math.trunc(CALIBRATION.cuda_context_gib * GIB),
      side: "eager",
      note: "process CUDA context",
    });
    if (forRank(res)) {
      terms.push({
        name: "reserved",
        bytes: forRank(res),
        side: "eager",
        note: res.note || "caller reservations",
      });
    }

    const total = terms.reduce((sum, term) => sum + term.bytes, 0);
    const headroom = gpu.totalBytes - total;
    const fits = headroom >= 0 && facts.trunkDimsKnown;
    if (!fits && facts.trunkDimsKnown) {
      advice.push(...eagerAdvice(activation, spec, facts));
    }
    return {
      spec,
      gpu,
      facts,
      terms,
      fits,
      headroomBytes: headroom,
      poolHeadroomBytes: headroom,
      outsideHeadroomBytes: headroom,
      poolBytes: 0,
      kvCapacityTokens: 0,
      advice,
      warnings,
      evidence: "estimated",
    };
  }

  // --- vLLM: two sides of the utilization line ---------------------------------
  if (!facts.trunkDimsKnown) {
    // The KV cache is the whole question on a vLLM backend -- it is what the pool spends whatever
    // the weights leave -- so with no attention dims there is nothing to weigh it against. Said out
    // loud rather than silently priced at zero, because a KV term of zero reads as "it fits".
    warnings.push(
      `cannot size the KV cache for ${facts.modelId}: its config gave no layer or head dimensions, so only the weights below are real. Every figure that depends on the cache is omitted rather than guessed.`,
    );
  } else if (recurrentLayers(facts)) {
    // Charging these layers per token was a 4x over-estimate on a 3:1 trunk, so they are no longer
    // charged -- but what replaces it is nothing rather than the right number. The state pool is real
    // memory this module does not price, and the direction of that omission is optimistic.
    warnings.push(
      `${recurrentLayers(facts)} of ${facts.nLayers} layers are recurrent and cache no tokens, so only ${kvCachingLayers(facts)} are charged for the KV cache. The fixed-size state those layers hold is sized by max_num_seqs rather than by context and is NOT priced here, so the capacity below is an upper bound by an unmeasured margin`,
    );
  }
  if (facts.trunkDimsKnown && !kvCachingLayers(facts)) {
    // Every layer recurrent: the cache is genuinely zero, and zero is also this module's word for
    // "unknown". Neither reading may be allowed to come out as room to spare.
    warnings.push(
      `every layer of ${facts.modelId} is recurrent, so there is no KV cache to size and the state pool that replaces it is not priced here. No fit is claimed`,
    );
  }
  const context = Math.trunc(CALIBRATION.cuda_context_gib * GIB);
  const overshoot = Math.trunc(CALIBRATION.vllm_overshoot_gib * GIB);
  const frag = Math.trunc(CALIBRATION.frag_fraction * gpu.totalBytes);
  const reservedOutside = res.beforeEngine ? 0 : forRank(res);
  const reservedInside = res.beforeEngine ? forRank(res) : 0;

  terms.push({
    name: "vllm_overshoot",
    bytes: overshoot,
    side: "outside",
    note: "what vLLM allocates past its own budget during warmup",
  });
  terms.push({
    name: "fragmentation",
    bytes: frag,
    side: "outside",
    note: `${Math.round(CALIBRATION.frag_fraction * 100)}% of the card`,
  });
  if (reservedOutside) {
    terms.push({
      name: "reserved",
      bytes: reservedOutside,
      side: "outside",
      note:
        res.note ||
        "caller reservations, allocated after the engine (SAEs, lens, other processes)",
    });
  }

  // Inside the pool. The CUDA context is here rather than outside because vLLM's budget is measured
  // against what the process is ALREADY using.
  terms.push({
    name: "cuda_context",
    bytes: context,
    side: "pool",
    note: "process CUDA context, charged against vLLM's budget",
  });
  if (reservedInside) {
    terms.push({
      name: "reserved",
      bytes: reservedInside,
      side: "pool",
      note:
        res.note ||
        "caller reservations, allocated before the engine so vLLM sees them as used",
    });
  }

  const perCardWeights = Math.floor(weightsTotal / tp);
  terms.push({
    name: "weights",
    bytes: perCardWeights,
    side: "pool",
    note:
      `${(facts.weights.paramCount / 1e9).toFixed(1)}B params at ${spec.dtype}` +
      (tp > 1 ? `, sharded over TP=${tp}` : "") +
      ` [${facts.weights.source}]`,
  });

  const sites = resolvedStaticSites(spec, facts);
  const buffers = sites
    ? staticTapBytes(staticElements(spec, facts), spec.maxNumBatchedTokens)
    : 0;
  if (sites) {
    const named = resolvedStaticPoints(spec, facts);
    terms.push({
      name: "static_buffers",
      bytes: buffers,
      side: "pool",
      note:
        (named.length && !spec.staticSites
          ? `${named.join(", ")} at ${Math.max(facts.nLayers, 1)} layers`
          : `${sites} sites x ${staticWidth(facts)} wide`) +
        ` = ${sites} buffers x ${spec.maxNumBatchedTokens} rows` +
        (tp > 1 ? " (not sharded by TP)" : ""),
    });
  }

  const graphs = graphsOn(spec)
    ? Math.trunc(CALIBRATION.graph_pool_gib * GIB)
    : 0;
  if (graphs) {
    terms.push({
      name: "graph_pool",
      bytes: graphs,
      side: "pool",
      note: "CUDA graph capture pool",
    });
  }

  const shards = kvShards(facts, tp);
  const kvFloor = Math.trunc(
    (kvBytesPerToken(facts, spec.kvCacheDtype, spec.dtype) * spec.maxModelLen) /
      shards,
  );
  terms.push({
    name: "kv_cache_floor",
    bytes: kvFloor,
    side: "pool",
    note:
      `one sequence of ${num(spec.maxModelLen)} tokens, every caching layer at full context` +
      (recurrentLayers(facts)
        ? ` (${kvCachingLayers(facts)} of ${facts.nLayers}; the rest are recurrent)`
        : "") +
      (tp > 1
        ? shards > 1
          ? `, sharded ${shards} ways`
          : ", replicated on every rank"
        : ""),
  });

  const poolAvailable = Math.trunc(spec.gpuMemoryUtilization * gpu.totalBytes);
  const outsideNeeded = overshoot + frag + reservedOutside;
  const poolNeeded =
    context + reservedInside + perCardWeights + buffers + graphs + kvFloor;

  const poolHeadroom = poolAvailable - poolNeeded;
  const outsideHeadroom = gpu.totalBytes - poolAvailable - outsideNeeded;
  const headroom = Math.min(poolHeadroom, outsideHeadroom);
  // An unsizable model is never reported as fitting. `kvFloor` is 0 when the dims are unknown, so the
  // arithmetic above would otherwise weigh the weights against the pool and find room to spare -- a
  // confident yes built on the one term nobody could measure. An all-recurrent trunk reaches the same
  // zero by a different road -- there really is no cache -- and is refused for the same reason: what
  // it holds instead is a state pool nothing here prices.
  const fits =
    poolHeadroom >= 0 &&
    outsideHeadroom >= 0 &&
    facts.trunkDimsKnown &&
    kvCachingLayers(facts) > 0;

  const kvRoom = Math.max(
    poolAvailable -
      context -
      reservedInside -
      perCardWeights -
      buffers -
      graphs,
    0,
  );
  const perToken =
    kvBytesForContext(facts, spec.maxModelLen, spec.kvCacheDtype, spec.dtype) /
    shards;
  const kvCapacity =
    perToken > 0 ? Math.trunc(kvRoom / (perToken / spec.maxModelLen)) : 0;

  if (!fits) {
    advice.push(
      ...vllmAdvice({
        spec,
        facts,
        gpu,
        poolHeadroom,
        outsideHeadroom,
        weights: perCardWeights,
        buffers,
      }),
    );
  }
  if (spec.gpuMemoryUtilization > CALIBRATION.max_util) {
    warnings.push(
      `gpu_memory_utilization=${spec.gpuMemoryUtilization} is above vLLM's own default of ${CALIBRATION.max_util}; the margin left outside the pool is thinner than the overshoot measured during warmup, so a fit here is not reliable`,
    );
  }

  return {
    spec,
    gpu,
    facts,
    terms,
    fits,
    headroomBytes: headroom,
    poolHeadroomBytes: poolHeadroom,
    outsideHeadroomBytes: outsideHeadroom,
    poolBytes: poolAvailable,
    kvCapacityTokens: kvCapacity,
    advice,
    warnings,
    evidence: "estimated",
  };
}

function eagerNote(
  name: string,
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): string {
  if (name === "logits") {
    return `${spec.batchSize}x${spec.seqLen} x vocab ${num(facts.vocabSize)}, plus the fp32 upcast`;
  }
  if (name === "attention") {
    return `quadratic in the prompt: ${facts.nHeads} heads x ${spec.seqLen}^2, attn_implementation='eager'`;
  }
  if (name === "capture_buffers") {
    return `${spec.nCapturePoints} points x ${spec.batchSize}x${spec.seqLen} x ${facts.dModel}`;
  }
  if (name === "mlp_intermediate") {
    return `intermediate ${num(facts.intermediateSize)}`;
  }
  if (name === "hidden_states" && spec.requiresGrad) {
    return `retained across all ${facts.nLayers} layers (requires_grad=True)`;
  }
  return "";
}

/** What to change, cheapest first, naming the term each fixes. */
function eagerAdvice(
  activation: [string, number][],
  spec: WorkloadSpec,
  facts: ModelMemoryFacts,
): string[] {
  const at = (name: string) =>
    activation.find(([key]) => key === name)?.[1] ?? 0;
  const out: string[] = [];
  if (at("attention") > GIB) {
    out.push(
      `attn_implementation='sdpa' removes the ${gib(at("attention"))} GiB attention matrix entirely; it is quadratic in the prompt and load_model defaults to 'eager' here`,
    );
  }
  if (at("logits") > GIB) {
    out.push(
      `the logits are ${gib(at("logits"))} GiB at ${spec.seqLen} tokens over a ${num(facts.vocabSize)} vocab; ${Math.floor(spec.seqLen / 2)} tokens halves it`,
    );
  }
  if (spec.requiresGrad) {
    out.push(
      "requires_grad=False if you are not fitting a lens: gradients retain every layer's activations",
    );
  }
  if (spec.dtype === "float32" || spec.dtype === "fp32") {
    out.push(
      "dtype='bfloat16' halves the weights and every activation term (EagerModel defaults to float32)",
    );
  }
  out.push(
    "backend='vllm' pages the KV cache and never materializes all-position logits",
  );
  return out;
}

function vllmAdvice({
  spec,
  facts,
  gpu,
  poolHeadroom,
  outsideHeadroom,
  weights,
  buffers,
}: {
  spec: WorkloadSpec;
  facts: ModelMemoryFacts;
  gpu: Gpu;
  poolHeadroom: number;
  outsideHeadroom: number;
  weights: number;
  buffers: number;
}): string[] {
  const out: string[] = [];
  if (outsideHeadroom < 0) {
    out.push(
      `lower gpu_memory_utilization: ${gib(-outsideHeadroom)} GiB more is needed OUTSIDE the pool, which is where the CUDA context, the warmup overshoot and your own reservations live`,
    );
  }
  if (poolHeadroom < 0) {
    const shortfall = gib(-poolHeadroom);
    if (buffers && buffers > -poolHeadroom) {
      out.push(
        `static buffers are ${gib(buffers)} GiB: a smaller max_num_batched_tokens, or static_writes=[] to drop the write half, recovers most of ${shortfall} GiB`,
      );
    }
    if (spec.maxModelLen > 2048) {
      // Per card, like every other figure in this list: the reader is looking at one card's
      // shortfall, so an unsharded saving would not be the saving they are being offered.
      const freed =
        (Math.floor(spec.maxModelLen / 2) *
          kvBytesPerToken(facts, spec.kvCacheDtype, spec.dtype)) /
        kvShards(facts, spec.numGpus);
      out.push(
        `max_model_len=${spec.maxModelLen} sets the KV floor; halving it frees about ${gib(freed)} GiB`,
      );
    }
    if (
      ["float32", "fp32", "auto"].includes(spec.dtype) &&
      ["float32", "fp32"].includes(facts.weights.storedDtype)
    ) {
      out.push(
        "dtype='bfloat16' halves the weights, which are the largest term in the pool",
      );
    }
    if (weights > gpu.totalBytes * 0.7) {
      const need = weights / (gpu.totalBytes * 0.6);
      out.push(
        `the weights alone are ${gib(weights)} GiB of a ${totalGib(gpu).toFixed(1)} GiB card; num_gpus=${Math.max(2, Math.trunc(need) + 1)} shards them, or use a quantized checkpoint`,
      );
    }
    if (graphsOn(spec)) {
      out.push(
        `backend='vllm' instead of '${spec.backend}' frees the ${CALIBRATION.graph_pool_gib.toFixed(0)} GiB graph pool and every static buffer, at 4-11x lower decode throughput`,
      );
    }
  }
  return out;
}

// ---------------------------------------------------------------------- the fit

export interface FitOptions {
  backend?: Backend;
  res?: Reservations;
  maxModelLen?: number;
  dtype?: string;
  kvCacheDtype?: string;
  numGpus?: number;
  staticSites?: number;
  staticPoints?: string[];
  maxNumBatchedTokens?: number;
  minKvSequences?: number;
  batchSize?: number;
  seqLen?: number;
  nCapturePoints?: number;
  requiresGrad?: boolean;
  attnImplementation?: string;
}

/**
 * The largest configuration of this shape that fits on one card, or null when none does.
 *
 * Solves in the order the constraints actually bind. **Utilization** is arithmetic rather than
 * search — everything outside the pool is fixed by the card and the reservations — and it is
 * *truncated* to two decimals rather than rounded, because rounding up is the direction that OOMs.
 * **Batch width** then steps down until the static buffers fit beside the weights and the KV floor.
 * And a configuration holding fewer than `minKvSequences` full-length sequences is refused outright:
 * it would serve one request at a time and stall the rest, which reads as a hang rather than as the
 * capacity problem it is.
 */
export function fit(
  facts: ModelMemoryFacts,
  gpu: Gpu,
  options: FitOptions = {},
): MemoryEstimate | null {
  const {
    backend = "vllm",
    res = reservations(),
    maxModelLen = 0,
    dtype = "auto",
    kvCacheDtype = "auto",
    numGpus = 1,
    staticSites = 0,
    staticPoints = [],
    maxNumBatchedTokens = 0,
    minKvSequences = 2,
    batchSize = 1,
    seqLen = 0,
    nCapturePoints = 0,
    requiresGrad = false,
    attnImplementation = "",
  } = options;

  // `estimate` refuses to report `fits` on a model whose dims are unknown, so every rung of every
  // ladder below would come back false. Callers wanting to tell "cannot size" from "does not fit"
  // should read `facts.trunkDimsKnown`, which is the only thing that distinguishes them.
  if (!facts.trunkDimsKnown) return null;

  const advertised = facts.maxPositionEmbeddings || 4096;

  if (backend === "eager") {
    // Prompt length is to eager what context is to vLLM: the term that decides whether it fits, and
    // the one to step down. Sizing only for the advertised context is how this came to report that
    // gemma-3-12b runs eagerly on *no* card in the catalog -- its 131k context puts 206 GiB of logits
    // on the card, while the same model at a 4k prompt fits one A40 with room to spare.
    const pinned = seqLen || maxModelLen;
    const prompts = pinned
      ? [pinned]
      : [advertised, ...CONTEXT_LADDER.filter((n) => n < advertised)];
    for (const prompt of prompts) {
      // `sdpa` rather than the engine's `eager` default: the quadratic attention matrix is the
      // largest avoidable term here, and a sizer recommending a configuration should recommend the
      // one that works. `estimate` still prices the default when asked for it.
      const est = estimate(
        facts,
        gpu,
        workload({
          backend: "eager",
          dtype,
          numGpus,
          batchSize,
          seqLen: prompt,
          nCapturePoints,
          requiresGrad,
          attnImplementation: attnImplementation || "sdpa",
          maxModelLen,
        }),
        res,
      );
      if (est.fits) return est;
    }
    return null;
  }

  // Only what genuinely lands outside the pool sets the ceiling on utilization. The CUDA context does
  // not: vLLM charges it against its own budget, so counting it here would lower utilization to buy
  // margin that is not needed and shrink the KV cache for nothing.
  let outside =
    Math.trunc(CALIBRATION.vllm_overshoot_gib * GIB) +
    Math.trunc(CALIBRATION.frag_fraction * gpu.totalBytes);
  if (!res.beforeEngine) outside += forRank(res);

  const ceiling = (gpu.totalBytes - outside) / gpu.totalBytes;
  // Truncate, never round: one step of 0.01 is ~0.44 GiB on a 44 GiB card, and the estimate feeding
  // this has more error than that in the optimistic direction.
  const util = Math.min(CALIBRATION.max_util, Math.trunc(ceiling * 100) / 100);
  if (util < CALIBRATION.min_util) return null;

  // Pinned by the caller means pinned: a sizer that quietly served less context than was asked for
  // would be answering a different question, and the caller would find out at request time.
  const contexts = maxModelLen
    ? [maxModelLen]
    : [advertised, ...CONTEXT_LADDER.filter((size) => size < advertised)];

  const asked =
    maxNumBatchedTokens || (backend === "vllm-static" ? 8192 : 2048);
  const batchWidths = [asked, ...CAPTURE_SIZES.filter((size) => size < asked)];

  // Context-major: the largest context that fits is worth more than the widest prefill batch,
  // because too small a context refuses requests outright while a narrow batch only slows prefill.
  for (const context of contexts) {
    for (const batched of batchWidths) {
      const est = estimate(
        facts,
        gpu,
        workload({
          backend,
          dtype,
          kvCacheDtype,
          maxModelLen: context,
          // A prefill batch wider than the context is waste: nothing can fill it.
          maxNumBatchedTokens: Math.min(batched, context),
          gpuMemoryUtilization: util,
          numGpus,
          staticSites,
          staticPoints,
        }),
        res,
      );
      if (est.fits && concurrentSequences(est) >= minKvSequences) return est;
    }
  }
  return null;
}

export interface FitResult {
  gpu: Gpu;
  count: number;
  estimate: MemoryEstimate;
}

/**
 * Every `(gpu, count)` that fits, smallest card and fewest cards first.
 *
 * Multi-GPU is tried in powers of two only, because that is what tensor parallelism requires: the
 * attention heads have to divide evenly across ranks, and vLLM refuses a count that does not divide
 * the KV heads.
 */
export function fitAcross(
  facts: ModelMemoryFacts,
  {
    gpus = GPUS,
    maxGpus = 8,
    ...options
  }: FitOptions & { gpus?: Gpu[]; maxGpus?: number } = {},
): FitResult[] {
  const out: FitResult[] = [];
  const ordered = [...gpus].sort((a, b) => a.totalBytes - b.totalBytes);
  for (const gpu of ordered) {
    for (let count = 1; count <= maxGpus; count *= 2) {
      const est = fit(facts, gpu, { ...options, numGpus: count });
      if (est) {
        out.push({ gpu, count, estimate: est });
        break;
      }
    }
  }
  return out;
}

// ------------------------------------------------------------------- provenance

export function findGpu(name: string): Gpu | undefined {
  const wanted = name.trim().toLowerCase();
  return GPUS.find(
    (gpu) =>
      gpu.name.toLowerCase() === wanted ||
      gpu.aliases.some((alias) => alias.toLowerCase() === wanted),
  );
}

export interface Evidence {
  kind: "verified" | "fails" | "estimated";
  /** What to show on the badge. */
  label: string;
  /** The run it rests on, when there is one. */
  run?: VerifiedRun;
}

/** How much a spec asks for, as a tuple that can be ordered against another spec's. */
function width(spec: {
  maxModelLen: number;
  maxNumBatchedTokens: number;
  seqLen: number;
}): [number, number, number] {
  return [spec.maxModelLen || 0, spec.maxNumBatchedTokens || 0, spec.seqLen || 0];
}

/**
 * What real hardware has to say about a configuration: verified, known to fail, or estimated.
 *
 * Matched on the card as well as the model, because a verified row on a *different* GPU is not
 * evidence for this one — and matched on the **settings** too, which is the part that is easy to skip
 * and produces a confident lie when skipped: gemma-3-12b's `vllm-static` crash was recorded at 16,384
 * batched tokens while the recommended configuration uses 8,192, so keying on the backend alone
 * stamped the recommendation KNOWN TO FAIL on the strength of a run twice as wide.
 *
 * Where the settings differ, one run can still speak about another, in one direction only. Every knob
 * in {@link width} costs more memory as it grows, so a run that **failed** condemns anything asking
 * for at least as much, and one that **passed** vouches for anything asking for no more. The reverse
 * of either is not evidence.
 */
export function evidenceFor(
  facts: ModelMemoryFacts,
  gpu: Gpu,
  spec: WorkloadSpec,
): Evidence {
  let inexact: Evidence | null = null;
  const ours = width(spec);
  for (const run of VERIFIED_RUNS) {
    if (
      run.modelId !== facts.modelId ||
      run.gpu !== gpu.name ||
      run.backend !== spec.backend ||
      run.dtype !== spec.dtype
    ) {
      continue;
    }
    const theirs = width(run);
    const failed = run.outcome !== "pass";
    if (theirs.every((value, index) => value === ours[index])) {
      // An exact match settles it outright, so return rather than keep looking.
      return failed
        ? { kind: "fails", label: `known to fail (${run.outcome})`, run }
        : { kind: "verified", label: "verified", run };
    }
    if (failed && ours.every((value, index) => value >= theirs[index])) {
      inexact ??= {
        kind: "fails",
        label: `fails at ${num(theirs[0])} ctx`,
        run,
      };
    } else if (!failed && ours.every((value, index) => value <= theirs[index])) {
      inexact ??= {
        kind: "verified",
        label: `verified at ${num(theirs[0])} ctx`,
        run,
      };
    }
  }
  return inexact ?? { kind: "estimated", label: "estimated" };
}

/**
 * Runnable `load_model(...)` for a fitted spec.
 *
 * Only the arguments that matter are emitted. A snippet restating every default is one nobody reads,
 * and one carrying arguments the chosen backend ignores teaches the wrong thing.
 */
export function snippet(
  facts: ModelMemoryFacts,
  spec: WorkloadSpec,
  gpu: Gpu,
  count: number,
): string {
  const modelId = facts.modelId;
  const lines = [`# ${modelId} on ${count}x ${gpu.name}`];
  const args = [`    "${modelId}"`, `    backend="${spec.backend}"`];
  if (spec.dtype && spec.dtype !== "auto") args.push(`    dtype="${spec.dtype}"`);
  if (count > 1) args.push(`    num_gpus=${count}`);
  // `"auto"` *is* the default point at every layer, so a set equal to it is spelled the short way.
  // Anything else has to be named per layer: a static tap is per (point, layer), and there is no
  // wildcard for "this point everywhere".
  //
  // `static_writes` is restated rather than left out, and that is the difference between this
  // snippet and the estimate above it agreeing or not. `"auto"` declares a write beside every read;
  // an explicit list does not, so a bare `static_points=` would allocate half the buffers this was
  // priced for and quietly refuse every steer at the addresses it just tapped.
  const points = resolvedStaticPoints(spec, facts);
  const named =
    points.length > 1 ||
    (points.length === 1 && points[0] !== defaultStaticPoint(facts));
  const perLayer = (chosen: string[]) =>
    `[\n        Address(point, layer)\n` +
    `        for point in (${chosen.map((point) => `"${point}"`).join(", ")}${chosen.length === 1 ? "," : ""})\n` +
    `        for layer in range(${Math.max(facts.nLayers, 1)})\n    ]`;
  if (named) {
    args.push(`    static_points=${perLayer(points)}`);
    // `attn` is capture-only: it names the kernel's own q/k/v, and the engine refuses a write there.
    const writable = points.filter((point) => staticPointBuffers(point) !== 3);
    if (writable.length) {
      args.push(`    static_writes=${perLayer(writable)}`);
    }
  }
  if (isVllm(spec.backend)) {
    args.push(`    gpu_memory_utilization=${spec.gpuMemoryUtilization}`);
    args.push(`    max_model_len=${spec.maxModelLen}`);
    if (spec.maxNumBatchedTokens) {
      args.push(
        `    extra_vllm_kwargs={"max_num_batched_tokens": ${spec.maxNumBatchedTokens}}`,
      );
    }
  } else {
    args.push(
      `    attn_implementation="${spec.attnImplementation || "sdpa"}"`,
    );
    lines.push(
      `# sized for prompts up to ${spec.seqLen} tokens -- longer ones grow quadratically`,
    );
  }
  lines.push(
    named
      ? "from interp_engine import Address, load_model"
      : "from interp_engine import load_model",
    "",
    "model = load_model(",
  );
  lines.push(args.join(",\n") + ",");
  lines.push(")");
  return lines.join("\n");
}
