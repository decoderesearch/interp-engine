/**
 * The GPU sizer, without the UI.
 *
 * Same question the sizer dialog answers — "what will run this model, and with what settings" — and
 * deliberately the same arithmetic: this route takes the inputs the sizer's controls produce, hands
 * them to the same `fitAcross` the component calls, and serializes what comes back. There is no
 * second estimator here to drift away from the first one, and `scripts/check-size.ts` already holds
 * `lib/size.ts` against `gpu-sizer/fit.py`, so the answer a caller gets is the answer the page and
 * the CLI give.
 *
 * Model resolution reuses the sizer's first two tiers and stops there. The build-time cache in
 * `data/models.generated.ts` answers most ids for free; anything else costs a Hub read on this
 * project's shared token, metered by `checkHubLimit` for the reason `/api/hub` gives at length. The
 * third tier — a reader's own token, sent from their browser straight to huggingface.co — has no
 * analogue on a server, and inviting callers to put a token in a query string would be worse than
 * not offering it, so a gated repo with no cached entry is a 422 pointing at the CLI.
 */

import { CALIBRATION, GPUS } from "@/data/gpus.generated";
import { HubError, isRepoId, resolveModel } from "@/lib/hub";
import type { ModelMemoryFacts } from "@/lib/hub";
import { cachedModel } from "@/lib/models";
import { checkHubLimit, clientKey, configured, exempt } from "@/lib/ratelimit";
import {
  BACKENDS,
  concurrentSequences,
  estimate,
  evidenceFor,
  fitAcross,
  GIB,
  isVllm,
  jacobianLens,
  offeredStaticPoints,
  reservations,
  resolvedStaticPoints,
  snippet,
  totalGib,
  workload,
  type Backend,
  type FitResult,
  type MemoryEstimate,
  type Reservations,
} from "@/lib/size";
import { byTier, defaultTier, shortGpuName, tierGib } from "@/lib/tiers";

/** `lib/hub.ts` is isomorphic, but a cold model is six URLs and wants a real Node fetch. */
export const runtime = "nodejs";

/** A cache hit is arithmetic and returns in microseconds; a cold repo waits on the Hub. */
export const maxDuration = 30;

/**
 * The dtypes the sizer prices, matching the control in `components/sizer/Sizer.tsx`.
 *
 * A closed list rather than anything `dtypeBytes` would accept, because that function matches by
 * substring and would quietly price a typo: `bflot16` contains none of its tags and falls through to
 * the 2-byte default, which is right by luck rather than by reading.
 */
const DTYPES = ["auto", "bfloat16", "float16", "float32"];

/** Reservations are entered in GiB, and past this the answer is not a sizing question. */
const MAX_RESERVE_GIB = 1024;

/** Above the largest context any catalog model advertises, and past the ladder `fit` searches. */
const MAX_CONTEXT = 4_194_304;

export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;

  const model = params.get("model")?.trim() ?? "";
  if (!isRepoId(model)) {
    return problem(400, "not a Hugging Face model id");
  }

  const backend = params.get("backend")?.trim() || "vllm";
  if (!BACKENDS.includes(backend as Backend)) {
    return problem(400, `backend must be one of ${BACKENDS.join(", ")}`);
  }

  const dtypeAsked = params.get("dtype")?.trim() ?? "";
  if (dtypeAsked && !DTYPES.includes(dtypeAsked)) {
    return problem(400, `dtype must be one of ${DTYPES.join(", ")}`);
  }

  const context = digits(params.get("max_model_len"), MAX_CONTEXT);
  if (context instanceof Error) {
    return problem(400, `max_model_len: ${context.message}`);
  }

  const reserveGib = decimal(params.get("reserve_gib"), MAX_RESERVE_GIB);
  if (reserveGib instanceof Error) {
    return problem(400, `reserve_gib: ${reserveGib.message}`);
  }

  const lens = flag(params.get("jacobian_lens"));
  if (lens instanceof Error) {
    return problem(400, `jacobian_lens: ${lens.message}`);
  }

  // De-duplicated here as well as in `resolvedStaticPoints`, which is where it decides the price.
  // This copy is what `request.staticPoints` echoes, and an echo showing a point twice beside an
  // answer that charged for it once reads as the sizer having missed one.
  const asked = [
    ...new Set(
      params.getAll("static_point").flatMap((value) =>
        value
          .split(",")
          .map((point) => point.trim())
          .filter(Boolean),
      ),
    ),
  ];
  // Bounded before the names are echoed in an error below, so a junk list cannot turn a 400 into a
  // response the size of the request. No trunk offers close to this many.
  if (asked.length > 32) {
    return problem(400, "static_point: at most 32 distinct points");
  }
  // Refused rather than ignored. A caller who named tap points has priced a pod with tap buffers in
  // it, and every other backend allocates none -- serving the cheaper answer under the settings that
  // asked for the dearer one is the direction that OOMs.
  if (asked.length && backend !== "vllm-static") {
    return problem(
      400,
      "static_point applies to backend=vllm-static, which is the only one that allocates tap buffers",
    );
  }

  let facts: ModelMemoryFacts | null = await cachedModel(model);

  if (!facts) {
    const token = process.env.HF_TOKEN;
    // Both guards are the sibling route's, and only reached on a cache miss: a model in
    // `data/models.generated.ts` needs no token and spends no allowance, so an unconfigured
    // deployment still answers for the sixty-odd models this project has run.
    if (!token || (!configured && !exempt)) {
      return problem(503, "model lookup is not configured on this deployment");
    }

    const verdict = await checkHubLimit(clientKey(request.headers));
    if (!verdict.ok) {
      return problem(429, verdict.reason, {
        "retry-after": String(verdict.retryAfter),
      });
    }

    try {
      facts = await resolveModel(model, { token });
    } catch (cause) {
      if (cause instanceof HubError) {
        // A network failure reaching the Hub is 502 rather than the 0 it carries: from the caller's
        // side an upstream that did not answer is this deployment's problem, not theirs.
        return problem(cause.status || 502, cause.message);
      }
      console.error("[sizer] resolve failed", cause);
      return problem(502, "could not reach the Hub");
    }
  }

  // "Cannot size" is not "does not fit", and reporting the second for the first is the failure worth
  // guarding: without `config.json` there is no attention shape, so the KV term has no honest value
  // and every card would come back false for a reason that has nothing to do with the cards.
  if (!facts.trunkDimsKnown) {
    return problem(
      422,
      `${facts.modelId} resolved its weights but not its trunk dimensions, so the KV cache cannot be ` +
        "sized. This is normally a gated repo: `python gpu-sizer/fit.py --token ...` sizes it with a " +
        "token of your own.",
      {},
      { model: describe(facts) },
    );
  }

  const offered = offeredStaticPoints(facts);
  const unknown = asked.filter((point) => !offered.includes(point));
  if (unknown.length) {
    return problem(
      400,
      `${facts.modelId} has no static tap at ${unknown.join(", ")}. It offers ${offered.join(", ")}.`,
    );
  }

  // The two derived defaults the component derives, so an omitted parameter means what it means on
  // the page. `recommendedDtype`: as stored for a quantized checkpoint, bf16 otherwise. And a model
  // whose advertised context was never stated gets a length worth pricing rather than a class
  // default nothing read.
  const dtype =
    dtypeAsked || (facts.weights.quantMethod ? "auto" : "bfloat16");
  const maxModelLen =
    context ??
    (facts.derivedDims.includes("max_position_embeddings") ? 8192 : 0);

  const res = withReserve(
    lens ? jacobianLens(facts) : reservations(),
    reserveGib ?? 0,
  );

  const results = fitAcross(facts, {
    backend: backend as Backend,
    dtype,
    maxModelLen,
    staticPoints: asked,
    res,
  });

  const best = defaultTier(byTier(results))?.result;

  return Response.json(
    {
      model: describe(facts),
      request: {
        backend,
        dtype,
        maxModelLen,
        staticPoints: asked,
        reserveGib: reserveGib ?? 0,
        jacobianLens: lens ?? false,
      },
      recommended: best ? { gpu: best.gpu.name, count: best.count } : null,
      results: results.map((result) => serialize(facts, result, res)),
      // Empty whenever something fits, since advice there is per result. When nothing does, this is
      // the only thing the response has to say, priced the way the page prices it.
      advice: results.length
        ? []
        : shortfall(facts, backend as Backend, dtype, maxModelLen, asked, res),
    },
    { headers: { "cache-control": "no-store" } },
  );
}

/** The knob the page adds on top of the lens, in the units it is entered in. */
function withReserve(base: Reservations, gib: number): Reservations {
  if (!gib) return base;
  return reservations({
    perRankBytes: base.perRankBytes + Math.round(gib * GIB),
    note: [base.note, `${gib} GiB reserved for your own tensors`]
      .filter(Boolean)
      .join(" + "),
  });
}

/** The facts a caller needs to check the arithmetic against their own, and nothing else. */
function describe(facts: ModelMemoryFacts) {
  return {
    id: facts.modelId,
    architecture: facts.architecture,
    nLayers: facts.nLayers,
    dModel: facts.dModel,
    nHeads: facts.nHeads,
    nKvHeads: facts.nKvHeads,
    headDim: facts.headDim,
    nExperts: facts.nExperts,
    nResidualStreams: facts.nResidualStreams,
    maxPositionEmbeddings: facts.maxPositionEmbeddings,
    vocabSize: facts.vocabSize,
    intermediateSize: facts.intermediateSize,
    weights: {
      paramCount: facts.weights.paramCount,
      onDiskBytes: facts.weights.onDiskBytes,
      storedDtype: facts.weights.storedDtype,
      quantMethod: facts.weights.quantMethod,
      source: facts.weights.source,
    },
    kvQuantAlgo: facts.kvQuantAlgo,
    gated: facts.gated,
    trunkDimsKnown: facts.trunkDimsKnown,
    /** Dims the config never stated. Each one widens the margin the figures deserve. */
    derivedDims: facts.derivedDims,
    notes: facts.notes,
    staticPoints: facts.trunkDimsKnown ? offeredStaticPoints(facts) : [],
  };
}

/**
 * One fitting `(gpu, count)`, as the detail panel shows it plus the code it prints.
 *
 * Repriced from the fitted spec rather than read off the search, which is what the panel does and for
 * the same reason: `fit` returns the first rung that worked, and the snippet has to describe exactly
 * the spec the terms were priced from or the two can drift apart.
 */
function serialize(
  facts: ModelMemoryFacts,
  result: FitResult,
  res: Reservations,
) {
  const { gpu, count } = result;
  const est = estimate(facts, gpu, result.estimate.spec, res);
  const evidence = evidenceFor(facts, gpu, est.spec);

  return {
    gpu: {
      name: gpu.name,
      shortName: shortGpuName(gpu.name),
      /** Usable bytes, which is what every figure here was weighed against. */
      totalBytes: gpu.totalBytes,
      totalGib: totalGib(gpu),
      /** The board size on the box, and how the page groups rows. */
      tierGib: tierGib(gpu),
      computeCapability: gpu.computeCapability,
      bandwidthGibS: gpu.bandwidthGibS,
    },
    count,
    evidence: { kind: evidence.kind, label: evidence.label },
    /** The `load_model` arguments the snippet below prints, as data. */
    spec: {
      backend: est.spec.backend,
      dtype: est.spec.dtype,
      numGpus: count,
      maxModelLen: est.spec.maxModelLen,
      maxNumBatchedTokens: est.spec.maxNumBatchedTokens,
      gpuMemoryUtilization: est.spec.gpuMemoryUtilization,
      attnImplementation: est.spec.attnImplementation,
      seqLen: est.spec.seqLen,
      staticPoints: resolvedStaticPoints(est.spec, facts),
    },
    // Null on eager rather than zero, there being no paged cache to report the capacity of.
    kv: isVllm(est.spec.backend)
      ? {
          capacityTokens: est.kvCapacityTokens,
          concurrentSequences: concurrentSequences(est),
        }
      : null,
    memory: memory(est),
    warnings: est.warnings,
    advice: est.advice,
    snippet: snippet(facts, est.spec, gpu, count),
  };
}

/**
 * Where the bytes went, in bytes.
 *
 * Two budgets on vLLM and not one, because they fail differently and are fixed by moving utilization
 * in opposite directions: the pool holds the weights, the context and the KV cache, while the warmup
 * overshoot, fragmentation and anything allocated after startup have to fit in the slice outside it.
 */
function memory(est: MemoryEstimate) {
  return {
    poolBytes: est.poolBytes,
    headroomBytes: est.headroomBytes,
    poolHeadroomBytes: est.poolHeadroomBytes,
    outsideHeadroomBytes: est.outsideHeadroomBytes,
    terms: est.terms.map((term) => ({
      name: term.name,
      bytes: term.bytes,
      side: term.side,
      note: term.note,
    })),
  };
}

/**
 * Why nothing fit, priced on the largest card in the catalog so the shortfall named is the smallest
 * one available.
 *
 * At vLLM's own default utilization rather than a derived ceiling: a fit search would have lowered it
 * to buy margin, and the question at this point is what the configuration costs rather than how to
 * squeeze it onto a card it does not fit.
 */
function shortfall(
  facts: ModelMemoryFacts,
  backend: Backend,
  dtype: string,
  maxModelLen: number,
  staticPoints: string[],
  res: Reservations,
): string[] {
  const biggest = [...GPUS].sort((a, b) => b.totalBytes - a.totalBytes)[0];
  const est = estimate(
    facts,
    biggest,
    workload({
      backend,
      dtype,
      maxModelLen,
      staticPoints,
      gpuMemoryUtilization: isVllm(backend) ? CALIBRATION.max_util : 0,
    }),
    res,
  );
  return [
    `nothing in the catalog fits, up to 8x ${biggest.name} at ${totalGib(biggest).toFixed(1)} GiB`,
    ...est.advice,
  ];
}

// ------------------------------------------------------------------- parameters

/**
 * A non-negative integer, or `null` when the parameter was absent.
 *
 * `Number` is not used on its own anywhere below: it reads `""` as 0, `"1e5"` as 100000 and `" 12 "`
 * as 12, so a caller's typo becomes a number the response then reports as though it had been asked
 * for. Absent and zero are also different here — zero is what "let the fit choose" is spelled as, and
 * absent is what makes a default apply.
 */
function digits(raw: string | null, max: number): number | null | Error {
  if (raw === null) return null;
  const value = raw.trim();
  if (!value) return null;
  if (!/^\d+$/.test(value)) return new Error("expected a non-negative integer");
  const parsed = Number(value);
  if (parsed > max) return new Error(`must be at most ${max}`);
  return parsed;
}

/** A non-negative decimal, for the one parameter entered in GiB. */
function decimal(raw: string | null, max: number): number | null | Error {
  if (raw === null) return null;
  const value = raw.trim();
  if (!value) return null;
  if (!/^\d+(\.\d+)?$/.test(value)) {
    return new Error("expected a non-negative number");
  }
  const parsed = Number(value);
  if (parsed > max) return new Error(`must be at most ${max}`);
  return parsed;
}

const TRUE = new Set(["1", "true", "yes", "on"]);
const FALSE = new Set(["0", "false", "no", "off"]);

/**
 * A boolean, spelled any of the usual ways.
 *
 * A bare `?jacobian_lens` is true, which is how a flag reads in a URL. Anything unrecognised is an
 * error rather than false: `jacobian_lens=maybe` silently meaning "no lens" would under-price a
 * read-out that is ~10 GiB a card on a 70B.
 */
function flag(raw: string | null): boolean | null | Error {
  if (raw === null) return null;
  const value = raw.trim().toLowerCase();
  if (!value) return true;
  if (TRUE.has(value)) return true;
  if (FALSE.has(value)) return false;
  return new Error("expected true or false");
}

function problem(
  status: number,
  error: string,
  headers: Record<string, string> = {},
  extra: Record<string, unknown> = {},
): Response {
  return Response.json(
    { error, ...extra },
    { status, headers: { "cache-control": "no-store", ...headers } },
  );
}
