/**
 * Resolve any Hugging Face model to the facts VRAM arithmetic needs, from the browser.
 *
 * The port of `interp_engine.memory`'s `weight_bytes` / `model_memory_facts` ladder to `fetch`, so
 * the sizer can size a model nobody generated data for. Nothing here downloads a shard: the whole
 * resolution is one repo-metadata call plus at most two JSON files, a few hundred KB.
 *
 * **The user's token never reaches our server, and we never use one of ours.** The Hub reflects the
 * caller's origin in `access-control-allow-origin` on both `api/models/{id}` and `resolve/main/*`
 * (verified through the 307 to `api/resolve-cache/...`, which carries the header too), and its
 * preflight answers `access-control-allow-headers: authorization`. So the browser talks to the Hub
 * directly and there is no API route to proxy, log or leak anything. See {@link setHubToken} for
 * what happens to a token that is supplied.
 *
 * The ladder, best first, each rung degrading to a **wider** margin rather than to a wrong number,
 * and every answer labelled with the rung that produced it ({@link WeightSource}) so a caller can
 * widen its own margin when the number came from arithmetic rather than from the checkpoint.
 *
 * Two things about the Hub's metadata are worth knowing before trusting it, both learned by
 * checking rather than by reading docs:
 *
 * - **`safetensors.parameters` is repo-wide, not checkpoint-wide.** `openai/gpt-oss-20b` ships the
 *   sharded MXFP4 checkpoint *and* `original/model.safetensors`, the same weights in another
 *   layout. The API's aggregate covers both, reporting `U8: 19.1e9` where the served checkpoint
 *   holds about half that, and summing every `.safetensors` in the repo gives 25.63 GiB for a
 *   12.82 GiB model. Hence {@link rootShards}: shards in a subdirectory are a variant, not a shard.
 * - **A gated repo hides its index as well as its config.** The plan for this client had only
 *   `config.json` behind the licence; `model.safetensors.index.json` is a 401 too. File sizes are
 *   not, which is the rung that keeps a gated model sizable at all.
 */

/** Which rung of the ladder produced a weight figure. The last two are lower bounds. */
export type WeightSource =
  | "safetensors-index"
  | "file-sizes"
  | "safetensors-headers"
  | "config-count"
  | "dense-guess"
  | "unknown";

/**
 * What a checkpoint's weights cost, keyed on how it will be *loaded*.
 *
 * The parts rather than one number, for the reason the Python says at length: a dequantized load
 * costs `paramCount` times the requested width, a native quantized load costs `onDiskBytes`, and
 * collapsing the two is how the engine's own estimator came to be 2x optimistic on eager.
 */
export interface WeightBytes {
  /** **Logical** parameters, containers unpacked. What a dequantized load costs. */
  paramCount: number;
  /** What the checkpoint occupies as stored. What a native quantized load costs. */
  onDiskBytes: number;
  /**
   * `config.dtype` (`torch_dtype` before transformers v5), e.g. `bfloat16`. Empty when the config
   * declares none, which happens — gpt-oss-20b does not.
   *
   * The dtype the checkpoint is stored in **only when it is not quantized**. On a packed repo this
   * is the *compute* dtype and the two differ: DeepSeek V4 declares `bfloat16` here and ships fp8
   * and fp4 weights, and both are true — bf16 is the residual stream, the norms and the unquantized
   * parameters. It is not every multiply: a repo with `activation_scheme: dynamic` casts the inputs
   * of its quantized GEMMs down. {@link WeightBytes.elementsByDtype} is what the tensors really are.
   */
  storedDtype: string;
  /** `quantization_config.quant_method`, lowercased. Empty on an unquantized checkpoint. */
  quantMethod: string;
  /**
   * `config.expert_dtype` when an MoE checkpoint stores its routed experts narrower than the rest of
   * itself. DeepSeek V4 is the case: `quant_method: fp8` with `expert_dtype: fp4`, and the experts
   * are the large majority of the weights, so the scheme alone names the smaller half. Empty
   * everywhere else.
   */
  expertDtype: string;
  /** Stored elements per safetensors dtype tag. Containers, not parameters, on a packed repo. */
  elementsByDtype: Record<string, number>;
  source: WeightSource;
}

/** Everything about a model that VRAM arithmetic needs, and nothing else. */
export interface ModelMemoryFacts {
  modelId: string;
  weights: WeightBytes;
  nLayers: number;
  dModel: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  /** Width of one *value* head, which a few families make different from `headDim`. */
  vHeadDim: number;
  vocabSize: number;
  intermediateSize: number;
  /**
   * Routed experts per sparse layer; 0 on a dense trunk. Feeds the static tap widths —
   * `router_logits` is as wide as the expert bank — and says whether a layer's MLP is a fused kernel
   * rather than three Linears, which decides whether `mlp_act` exists to tap at all.
   */
  nExperts: number;
  /** Per-layer attention kinds when the config states them. Sliding layers cache far less KV. */
  layerTypes: string[] | null;
  slidingWindow: number | null;
  /** Parallel residual streams. >1 multiplies every static tap buffer. */
  nResidualStreams: number;
  /** The model's advertised context, which is the `max_model_len` default. */
  maxPositionEmbeddings: number;
  architecture: string;
  /**
   * The scheme the **checkpoint** stores its KV cache in, when it declares one — `"fp8"` on NVIDIA's
   * FP4 exports, which pack weights at NVFP4 and the cache at FP8.
   *
   * Not the caller's `kvCacheDtype`: this is a property of the weights on disk, and vLLM honours it
   * whether or not anyone asked. Ignoring it costs a factor of two in the conservative direction,
   * which is why it went unnoticed for so long — `nvidia/Llama-3.3-70B-Instruct-FP4` on a B200 was
   * predicted to hold 394,295 tokens and built 784,896.
   */
  kvQuantAlgo: string;
  /** The repo's gating, from the API: `false`, `"auto"` or `"manual"`. */
  gated: string | false;
  /**
   * Whether `config.json` was actually read, as opposed to defaulted to zeros.
   *
   * Weight bytes and trunk dims come from different places, so a repo can be sized to the byte with
   * nothing known about its attention — which is the *normal* state for a gated repo with no token.
   * Anything needing the KV term has to check this and decline, because the arithmetic fails
   * quietly otherwise: guards written to avoid dividing by zero turn the unknown into a very small
   * number, and 4 bytes per token divides into any budget as billions of tokens of context.
   */
  trunkDimsKnown: boolean;
  /**
   * Dims the config did **not** state, which transformers would fill from a class default.
   *
   * The one thing reading raw JSON cannot do, and it is not hypothetical: `google/gemma-3-12b-pt`
   * omits `head_dim`, `max_position_embeddings` and `layer_types` altogether, and `Gemma3TextConfig`
   * supplies 256, 131072 and the 5:1 sliding pattern from its own defaults. Deriving `head_dim` as
   * `hidden_size / num_attention_heads` gives 240 against a true 256, which under-states the KV
   * width by 7% — the direction that OOMs.
   *
   * So it is named rather than hidden. A consumer that needs the KV term exactly should widen its
   * margin when `head_dim` appears here, and {@link ModelMemoryFacts.notes} says which way each
   * missing field errs.
   */
  derivedDims: string[];
  /** What the resolution could not reach or had to derive, in the words a UI can show. */
  notes: string[];
}

/** A Hub request that failed in a way the caller may be able to act on. */
export class HubError extends Error {
  readonly status: number;
  /** True when a token would plausibly fix it: the repo is gated or private. */
  readonly needsToken: boolean;

  constructor(message: string, status: number, needsToken = false) {
    super(message);
    this.name = "HubError";
    this.status = status;
    this.needsToken = needsToken;
  }
}

// ------------------------------------------------------------------ dtype widths

/**
 * Bytes per element, matched by substring, narrowest family first.
 *
 * Order matters: `float16` contains neither `fp4` nor `fp8`, but `nvfp4` must not fall through to
 * the 4-byte row via `fp32`-like matching, so the narrow tags are tested first. Mirrors
 * `memory._DTYPE_BYTES`.
 */
const DTYPE_BYTES: readonly (readonly [readonly string[], number])[] = [
  [["fp4", "nvfp4", "mxfp4", "int4", "uint4", "nf4"], 0.5],
  [["fp8", "int8", "uint8", "e4m3", "e5m2"], 1.0],
  [["float32", "fp32"], 4.0],
  [["bfloat16", "float16", "bf16", "fp16", "half"], 2.0],
];

/**
 * Bytes per element for a dtype name, or null when the name is unrecognized.
 *
 * Null rather than a default, so a caller falls back to something it knows instead of a guess: this
 * reads fields no schema constrains, and a wrong *narrower* answer is the one that OOMs.
 */
export function dtypeBytesOrNull(name: unknown): number | null {
  if (name === null || name === undefined || name === "") return null;
  const text = String(name).toLowerCase();
  for (const [tags, size] of DTYPE_BYTES) {
    if (tags.some((tag) => text.includes(tag))) return size;
  }
  return null;
}

/** Bytes per element, falling back to `fallback` (bf16) for an unrecognized name. */
export function dtypeBytes(name: unknown, fallback = 2.0): number {
  return dtypeBytesOrNull(name) ?? fallback;
}

/** Container widths for the integer safetensors tags. Mirrors `memory._CONTAINER_DTYPES`. */
const CONTAINER_DTYPES: Record<string, number> = {
  U8: 1.0,
  I8: 1.0,
  F8_E4M3: 1.0,
  F8_E5M2: 1.0,
  U16: 2.0,
  I16: 2.0,
  U32: 4.0,
  I32: 4.0,
  U64: 8.0,
  I64: 8.0,
};

/**
 * The two fp8 tags. Whether these hold weights or *scales* depends on the scheme around them: in an
 * fp8 checkpoint they are the payload, in a 4-bit one they are the per-block scales beside a `U8`
 * payload. {@link logicalParamCount} is where that distinction is paid for.
 */
const FP8_TAGS = new Set(["F8_E4M3", "F8_E5M2"]);

/**
 * Tags that are **only ever scales**. `F8_E8M0` is eight bits of exponent and no mantissa, so
 * nothing stores a weight in one; it carries the per-block scale of an MX or ue8m0 payload. Mirrors
 * `memory._SCALE_TAGS`.
 */
const SCALE_TAGS = new Set(["F8_E8M0"]);

/**
 * 64-bit integer tags, which are indices and bookkeeping — a rope table, an MTP map — and never a
 * packed weight. They are containers by width, so {@link CONTAINER_DTYPES} needs them for the byte
 * arithmetic, but unpacking one at a 4-bit scheme's width invents parameters sixteen at a time.
 * Mirrors `memory._INDEX_TAGS`.
 */
const INDEX_TAGS = new Set(["I64", "U64"]);

/**
 * Bytes per logical parameter per scheme, keyed on substrings of `quantMethod`.
 *
 * `awq` and `gptq` sit at 4-bit because that is what they are in practice; an 8-bit GPTQ export
 * exists but is rare, and reading it as 4-bit only over-states the dequantized size, which is the
 * safe direction for a warning.
 */
const SCHEME_WIDTH: readonly (readonly [readonly string[], number])[] = [
  [["mxfp4", "nvfp4", "fp4", "int4", "uint4", "nf4", "awq", "gptq"], 0.5],
  [
    ["fp8", "int8", "compressed-tensors", "finegrained_fp8", "modelopt_fp8"],
    1.0,
  ],
];

/** Bytes per logical parameter for a scheme, or null when it is not recognized. */
export function schemeWidth(quantMethod: string): number | null {
  const text = (quantMethod || "").toLowerCase();
  if (!text) return null;
  for (const [tags, width] of SCHEME_WIDTH) {
    if (tags.some((tag) => text.includes(tag))) return width;
  }
  return null;
}

/**
 * Logical parameters, unpacking whatever the containers hold.
 *
 * Safetensors headers count *stored elements*, and on a packed checkpoint that is not the parameter
 * count: the `U8` buckets are MXFP4 blocks holding two values per byte, so each stored byte is two
 * parameters. Reading the sum as a parameter count under-states such a model by nearly half, and it
 * matters in exactly one place that matters a lot — the size the checkpoint becomes when
 * transformers cannot find its kernels and *silently* dequantizes to bf16.
 *
 * `expertDtype` is what a byte container holds on a **mixed-precision** checkpoint, and passing it
 * is only correct when the counts came from the shards' own headers — see {@link resolveModel},
 * where the Hub's aggregate arrives already unpacked and must not be unpacked twice.
 *
 * An unquantized checkpoint is unaffected: its buckets are all float, so containers and parameters
 * are the same thing.
 */
export function logicalParamCount(
  elementsByDtype: Record<string, number>,
  quantMethod = "",
  expertDtype = "",
): number {
  const widths = [schemeWidth(quantMethod), schemeWidth(expertDtype)].filter(
    (width): width is number => width !== null,
  );
  const native = widths.length ? Math.min(...widths) : null;
  let total = 0;
  for (const [tag, count] of Object.entries(elementsByDtype)) {
    const upper = tag.toUpperCase();
    if (SCALE_TAGS.has(upper)) continue;
    const container = INDEX_TAGS.has(upper)
      ? undefined
      : CONTAINER_DTYPES[upper];
    if (container === undefined || native === null) {
      // A float bucket, or a scheme we do not recognize: one element is one parameter.
      total += count;
      continue;
    }
    if (native < 1.0 && FP8_TAGS.has(upper)) {
      // A 4-bit checkpoint packs its payload into bytes and keeps its per-block *scales* in fp8
      // beside it. Scales are not parameters, so unpacking them two-to-a-byte invents weights that
      // do not exist. Counting them one-for-one leaves them slightly over-counted instead, which
      // keeps the total a little conservative without inflating it by a whole tensor group.
      total += count;
      continue;
    }
    total += Math.round(count * (container / native));
  }
  return total;
}

/**
 * Bytes on the device after loading at `loadDtype`. **This is the correction that matters.**
 *
 * The obvious implementation returns `onDiskBytes`, and it is wrong in both directions. Eager's
 * dtype default is `float32`, so a bf16 checkpoint loaded at the default costs `paramCount * 4` —
 * double the file size, which is the difference between a 12B model fitting a 24 GiB card and not.
 * And a quantized checkpoint asked for at a float dtype *dequantizes*: gpt-oss-20b is 12.8 GiB of
 * MXFP4 on disk and about 41 GiB once transformers expands it.
 *
 * `dequantizes` is what the two backends disagree about, and getting it wrong is a 3x error in
 * whichever direction the caller was not expecting. To transformers, `dtype` is the dtype the
 * *weights* are materialized in, so asking bf16 of an MXFP4 checkpoint expands it. To vLLM the same
 * argument sets the **activation** dtype and the weights stay packed. So pass `false` for vLLM.
 *
 * Defaults to `true` because it is the pessimistic reading, and because the accident this exists to
 * price — transformers expanding a checkpoint because its kernels are missing — is a transformers
 * accident.
 */
export function bytesForLoad(
  weights: WeightBytes,
  loadDtype = "auto",
  { dequantizes = true }: { dequantizes?: boolean } = {},
): number {
  const wanted = dtypeBytesOrNull(loadDtype);
  // "auto" means "as stored", which is the only case where the file size is the answer.
  if (!loadDtype || loadDtype === "auto" || wanted === null)
    return weights.onDiskBytes;
  if (weights.quantMethod) {
    if (!dequantizes) return weights.onDiskBytes;
    const native = schemeWidth(weights.quantMethod);
    // Asking for the width it is already stored at, or narrower than transformers will give you:
    // the checkpoint is served natively and the file size stands.
    if (native !== null && wanted <= native) return weights.onDiskBytes;
  }
  return Math.round(weights.paramCount * wanted);
}

// ------------------------------------------------------------------- the token

/**
 * The user's Hub token, held in this module and nowhere else.
 *
 * A module variable rather than `localStorage` or `sessionStorage` on purpose: the commitment is
 * that the token is never *persisted*, so it dies with the tab. It is also never sent anywhere but
 * `huggingface.co` — every request in this file is built from {@link HUB} — and there is no API
 * route in this app that takes one.
 *
 * That last clause survives `/api/hub`, which is where lookups go when nobody has typed a token
 * here. The route accepts a model id and nothing else; holding a token is precisely what makes
 * `lib/resolve.ts` bypass it and call this file from the browser instead. The two tokens never meet.
 */
let hubToken: string | null = null;

/** Hold a token for this tab. Pass null or an empty string to clear it. */
export function setHubToken(token: string | null): void {
  hubToken = token?.trim() || null;
}

/** Whether a token is currently held. The value itself is deliberately not readable. */
export function hasHubToken(): boolean {
  return hubToken !== null;
}

/** Forget the token. */
export function clearHubToken(): void {
  hubToken = null;
}

// ------------------------------------------------------------------- fetching

const HUB = "https://huggingface.co";

/**
 * Every origin this file reaches, for `connect-src`. **Imported by `next.config.ts` rather than
 * typed into the policy**, for the reason `lib/assets.ts` gives about the demo recording: a CSP that
 * does not name a host fails silently, and here it fails in the one place nobody is looking — the
 * request is refused in the console and the panel shows "could not reach the Hub", which reads as the
 * Hub being down.
 *
 * Three entries, and the second two are not optional. A `resolve/main/*` request for a real blob
 * answers 302 to a regional CDN — `us.aws.cdn.hf.co` today, `eu.aws.cdn.hf.co` from Europe, and
 * `cdn-lfs*.huggingface.co` on repos not yet migrated to Xet — and CSP is enforced against **each
 * URL in a redirect chain**, not just the one the code asked for. So naming only `huggingface.co`
 * passes the config fetches and blocks the safetensors header reads, which is the subtlest possible
 * version of this bug: weights resolve, and only the dims that need a header go missing.
 */
export const HUB_ORIGINS = [
  HUB,
  "https://*.huggingface.co",
  "https://*.hf.co",
] as const;

/** The shape of `api/models/{id}?blobs=true` that this file reads. */
interface RepoInfo {
  id?: string;
  gated?: string | false;
  siblings?: { rfilename: string; size?: number }[];
  safetensors?: { parameters?: Record<string, number>; total?: number };
  /**
   * The Hub's own extract of `config.json`, and better than it looks: it carries `architectures`
   * and `quantization_config.quant_method` **on a gated repo with no token**, which is where the
   * scheme would otherwise be unreachable. A subset, so it supplements the real config rather than
   * replacing it.
   */
  config?: Record<string, unknown>;
}

interface FetchOptions {
  token?: string | null;
  signal?: AbortSignal;
  /** Injected by the verifier script; defaults to the global. */
  fetchImpl?: typeof fetch;
}

function authHeaders(
  token: string | null | undefined,
): HeadersInit | undefined {
  const resolved = token === undefined ? hubToken : token;
  return resolved ? { Authorization: `Bearer ${resolved}` } : undefined;
}

async function hubFetch(
  url: string,
  options: FetchOptions,
  extraHeaders?: Record<string, string>,
): Promise<Response> {
  const impl = options.fetchImpl ?? fetch;
  const auth = authHeaders(options.token);
  return impl(url, {
    headers: {
      ...(auth as Record<string, string> | undefined),
      ...extraHeaders,
    },
    signal: options.signal,
  });
}

/** Repo metadata: file sizes, gating, the dtype breakdown and the Hub's config extract. */
async function fetchRepoInfo(
  modelId: string,
  options: FetchOptions,
): Promise<RepoInfo> {
  const url = `${HUB}/api/models/${modelId}?blobs=true`;
  let response: Response;
  try {
    response = await hubFetch(url, options);
  } catch {
    // A network-level failure, which in a browser is also what a blocked request looks like.
    throw new HubError(`could not reach the Hub for ${modelId}`, 0);
  }
  if (response.status === 401 || response.status === 403) {
    // The Hub answers 401 for a repo that does not exist as well as for one that is merely out of
    // reach, so it cannot be read as "this is private" -- a mistyped id lands here too, and the 404
    // branch below is very nearly unreachable. Naming the likelier cause first, since a reader who
    // needs a token usually knows they do.
    throw new HubError(
      `could not read ${modelId} — check the id, or add a token if the repo is private`,
      response.status,
      true,
    );
  }
  if (response.status === 404)
    throw new HubError(`no model called ${modelId}`, 404);
  if (!response.ok) {
    throw new HubError(
      `the Hub answered ${response.status} for ${modelId}`,
      response.status,
    );
  }
  return (await response.json()) as RepoInfo;
}

/**
 * A JSON file from the repo, or null when it is absent or gated.
 *
 * Null rather than throwing, because both callers want to *degrade*: a gated `config.json` costs
 * the KV precision and nothing else, and a missing index just moves the weight figure down a rung.
 */
async function fetchRepoJson<T>(
  modelId: string,
  path: string,
  options: FetchOptions,
): Promise<{ data: T | null; status: number }> {
  const url = `${HUB}/${modelId}/resolve/main/${path}`;
  try {
    const response = await hubFetch(url, options);
    if (!response.ok) return { data: null, status: response.status };
    return { data: (await response.json()) as T, status: response.status };
  } catch {
    return { data: null, status: 0 };
  }
}

/**
 * The `.safetensors` files that make up *this* checkpoint: the ones at the repo root.
 *
 * The filter is the whole point. `openai/gpt-oss-20b` keeps `original/model.safetensors` beside its
 * three sharded files, the same weights in another layout, and summing all four reports 25.63 GiB
 * for a 12.82 GiB model. Anything in a subdirectory is a variant — a second precision, a
 * framework-specific export, a consolidated copy — and never part of the shard set.
 */
function rootShards(info: RepoInfo): { rfilename: string; size?: number }[] {
  return (info.siblings ?? []).filter(
    (s) => s.rfilename.endsWith(".safetensors") && !s.rfilename.includes("/"),
  );
}

/** Whether the repo ships a `.safetensors` outside the root, which makes the API's totals unusable. */
function hasVariantShards(info: RepoInfo): boolean {
  return (info.siblings ?? []).some(
    (s) => s.rfilename.endsWith(".safetensors") && s.rfilename.includes("/"),
  );
}

/**
 * Stored elements per dtype, read from the shards' own safetensors headers.
 *
 * Only needed when the repo ships variant shards and the Hub's aggregate therefore describes more
 * than the checkpoint. On `openai/gpt-oss-20b` the aggregate says `{BF16: 1.80e9, U8: 19.11e9}`
 * while the served checkpoint holds `{BF16: 1.80e9, U8: 10.15e9}` — the `BF16` bucket is identical
 * and only `U8` inflates, so no scaling of the aggregate can recover it and the headers are the
 * only honest source. Reading them turns the dequantized figure from a guess into 41.18 GiB, which
 * is what the checkpoint actually expands to.
 *
 * Two range requests per shard, a few KB each: eight bytes for the header length, then the header.
 * The Hub redirects these to its CDN, which answers `access-control-allow-origin: *` on the 206, so
 * the browser can do this cross-origin — and `Accept-Ranges` / `Content-Range` are named in the
 * API's `access-control-expose-headers`, so it is intended rather than incidental.
 *
 * Returns an empty object rather than throwing: a failure here costs precision on one term, and the
 * caller has a cruder path.
 */
async function readShardHeaders(
  modelId: string,
  shards: string[],
  options: FetchOptions,
): Promise<ShardHeaders> {
  const totals: Record<string, number> = {};
  const shapes: Record<string, number[]> = {};
  const perShard = await Promise.all(
    shards.map(async (shard) => {
      const url = `${HUB}/${modelId}/resolve/main/${shard}`;
      try {
        const lead = await hubFetch(url, options, { Range: "bytes=0-7" });
        if (!lead.ok) return null;
        const view = new DataView(await lead.arrayBuffer());
        if (view.byteLength < 8) return null;
        // Little-endian u64. Read as two u32s because a JS number cannot hold a full u64, and a
        // safetensors header is kilobytes, so the high word is always zero on a real file.
        const low = view.getUint32(0, true);
        const high = view.getUint32(4, true);
        if (high !== 0 || low === 0) return null;
        const body = await hubFetch(url, options, {
          Range: `bytes=8-${7 + low}`,
        });
        if (!body.ok) return null;
        return JSON.parse(await body.text()) as Record<string, unknown>;
      } catch {
        return null;
      }
    }),
  );
  for (const header of perShard) {
    if (!header) continue;
    for (const [name, entry] of Object.entries(header)) {
      if (name === "__metadata__") continue;
      const tensor = asRecord(entry);
      const dtype = tensor?.dtype;
      if (typeof dtype !== "string") continue;
      const shape = Array.isArray(tensor?.shape)
        ? (tensor.shape as unknown[])
        : [];
      // A zero-dimensional tensor is one element, not zero.
      const count = shape.reduce<number>(
        (product, dim) => product * Number(dim),
        1,
      );
      totals[dtype] = (totals[dtype] ?? 0) + count;
      if (DIM_BEARING.test(name)) shapes[name] = shape.map(Number);
    }
  }
  return { elements: totals, shapes };
}

/**
 * The tensors whose shapes say what a config left out, and the only ones worth keeping.
 *
 * Filtered rather than kept wholesale because a header can carry a thousand entries per shard and
 * two numbers are wanted from it: the embedding matrix is `[vocab, hidden]`, and a key projection is
 * `[kv_heads x head_dim, hidden]`.
 */
const DIM_BEARING =
  /(^|\.)(embed_tokens|wte|word_embeddings|tok_embeddings|embed_in|lm_head|k_proj|key_proj|wk)\.weight$/;

interface ShardHeaders {
  /** Stored elements per safetensors dtype tag. */
  elements: Record<string, number>;
  /** Shapes of the {@link DIM_BEARING} tensors, by tensor name. */
  shapes: Record<string, number[]>;
}

/**
 * `vocab_size` and `head_dim` read off the checkpoint, for the configs that do not state them.
 *
 * A multimodal `config.json` like gemma-3's carries a `text_config` holding seven fields, and
 * `vocab_size`, `head_dim`, `max_position_embeddings` and `layer_types` are none of them —
 * transformers fills all four from `Gemma3TextConfig`, which raw JSON cannot see. Two of the four are
 * recoverable from the weights themselves, and it is worth the round trip because the direction of
 * the error is dangerous rather than merely imprecise: a `vocab_size` of zero prices the eager logits
 * term — usually the *largest* term on that backend, ~50 GiB for a 12B at 32k tokens — at nothing,
 * so the finder would call a card roomy when it is nowhere near.
 *
 * One or two shards, not the set: the index's `weight_map` says which shard holds which tensor, so
 * the embedding and a key projection can be fetched directly. A repo without an index falls back to
 * its first root shard, which is where an embedding usually is.
 */
function dimsFromShapes(
  shapes: Record<string, number[]>,
  nKvHeads: number,
): { vocabSize: number; headDim: number } {
  let vocabSize = 0;
  let headDim = 0;
  for (const [name, shape] of Object.entries(shapes)) {
    if (shape.length !== 2) continue;
    if (
      /(embed_tokens|wte|word_embeddings|tok_embeddings|embed_in)\.weight$/.test(
        name,
      )
    ) {
      vocabSize = Math.max(vocabSize, shape[0]);
    } else if (/lm_head\.weight$/.test(name)) {
      // Same `[vocab, hidden]` shape, and present even where the embedding is under a name this does
      // not know. Second choice because a tied-embedding repo may not ship it at all.
      if (!vocabSize) vocabSize = shape[0];
    } else if (nKvHeads && /(k_proj|key_proj|wk)\.weight$/.test(name)) {
      const derived = Math.trunc(shape[0] / nKvHeads);
      if (derived > 0) headDim = derived;
    }
  }
  return { vocabSize, headDim };
}

/** Shards holding the tensors {@link dimsFromShapes} wants, or the first root shard. */
function dimBearingShards(info: RepoInfo, index: ShardIndex | null): string[] {
  const map = index?.weight_map ?? {};
  const wanted = [
    ...new Set(
      Object.entries(map)
        .filter(([name]) => DIM_BEARING.test(name))
        .map(([, shard]) => shard),
    ),
  ];
  if (wanted.length) return wanted.slice(0, 2);
  const roots = rootShards(info).map((shard) => shard.rfilename);
  return roots.slice(0, 1);
}

// -------------------------------------------------------------- reading a config

/** `config.json` is free-form JSON; every read goes through these rather than a cast. */
type Json = Record<string, unknown>;

function asRecord(value: unknown): Json | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Json)
    : null;
}

/**
 * The text sub-config, where a `*ForConditionalGeneration` checkpoint keeps its dims.
 *
 * On a multimodal config the top-level `num_hidden_layers` is absent and the real dims are under
 * `text_config`. The Python calls `get_text_config()`, a transformers accessor this has no access
 * to, so it reads the field — which is why a composite config that names its text half something
 * else resolves to no dims here and says so through {@link ModelMemoryFacts.trunkDimsKnown}.
 */
function textConfig(config: Json): Json {
  return asRecord(config.text_config) ?? config;
}

function firstInt(config: Json, keys: readonly string[]): number {
  for (const key of keys) {
    const value = config[key];
    if (typeof value === "number" && Number.isFinite(value))
      return Math.trunc(value);
  }
  return 0;
}

/**
 * How many KV heads the *forward pass* uses, which is not always what the config field says.
 *
 * Three spellings of the field, and one flag that overrides all of them: `multi_query`. Falcon
 * needs that flag because `FalconConfig` fills an unset `num_kv_heads` with `num_attention_heads`,
 * so Falcon-7B says 71 while attending with a single KV head. `new_decoder_architecture` opts back
 * out, making the field authoritative again.
 */
function effectiveKvHeads(cfg: Json, nHeads: number): number {
  let kv = firstInt(cfg, ["num_key_value_heads", "num_kv_heads", "n_head_kv"]);
  if (!kv) {
    const attn = asRecord(cfg.attn_config);
    if (attn) kv = firstInt(attn, ["kv_n_heads"]);
  }
  if (cfg.multi_query === true && cfg.new_decoder_architecture !== true)
    return 1;
  return kv || nHeads;
}

/** One-character-per-block layer patterns, for the configs that spell `layer_types` that way. */
const LAYER_PATTERN_KINDS: Record<string, string> = {
  M: "linear_attention",
  E: "moe",
  "*": "full_attention",
  "-": "mlp",
};

const LAYER_PATTERN_ATTRS = [
  "hybrid_override_pattern",
  "layers_block_type",
] as const;

function layerTypes(cfg: Json): string[] | null {
  const declared = cfg.layer_types;
  if (Array.isArray(declared) && declared.length) return declared.map(String);
  for (const attr of LAYER_PATTERN_ATTRS) {
    const pattern = cfg[attr];
    if (typeof pattern === "string" && pattern) {
      return [...pattern].map((char) => LAYER_PATTERN_KINDS[char] ?? char);
    }
    if (Array.isArray(pattern) && pattern.length) return pattern.map(String);
  }
  return null;
}

/**
 * Parallel residual streams (1 on a conventional transformer).
 *
 * Only an explicit `false` on the switch disables the mechanism: DeepSeek-V4 has no such flag, and
 * an absent flag is not a disabled one — reading it that way reports one stream for the family this
 * exists for.
 */
function residualStreams(cfg: Json): number {
  if (cfg.mhc_enabled === false) return 1;
  return Math.max(1, firstInt(cfg, ["hc_mult", "mhc_expansion_rate"]) || 1);
}

/** The scheme a config declares, from either the root or the text sub-config. */
function configQuantMethod(config: Json): string {
  for (const holder of [config, textConfig(config)]) {
    const q = asRecord(holder.quantization_config);
    if (!q) continue;
    const method = q.quant_method ?? q.quantization;
    if (typeof method === "string" && method) return method.toLowerCase();
  }
  return "";
}

/** The stored dtype. `dtype` first: transformers v5 renamed `torch_dtype` and warns on the old name. */
function configStoredDtype(config: Json): string {
  for (const holder of [config, textConfig(config)]) {
    const dt = holder.dtype ?? holder.torch_dtype;
    if (typeof dt === "string" && dt) return dt.replace("torch.", "");
  }
  return "";
}

/**
 * The routed experts' own dtype, which an MoE family states in a field of its own rather than in
 * `quantization_config` — the mixed precision is a property of the checkpoint, not of one scheme.
 * `interp_engine.memory._expert_dtype` and `vllm_capture.static` read the same field.
 */
function configExpertDtype(config: Json): string {
  for (const holder of [config, textConfig(config)]) {
    const dt = holder.expert_dtype;
    if (typeof dt === "string" && dt)
      return dt.replace("torch.", "").toLowerCase();
  }
  return "";
}

/**
 * The scheme implied by the tensor dtypes themselves, when nothing in the repo declares one.
 *
 * A last resort, and deliberately so: the safe reading of an undeclared checkpoint is that it is
 * dense. But the headers are real evidence, and there is one shape they can only have for one
 * reason — most of the model's elements sitting in a *byte* container, which no dense checkpoint
 * does. Requires a **majority**, so a small integer side-table cannot make a dense model look
 * packed. Both answers are labels for {@link schemeWidth} rather than claims about a vendor format.
 */
function schemeFromHeaders(elementsByDtype: Record<string, number>): string {
  const counts = Object.entries(elementsByDtype);
  const total = counts.reduce((sum, [, n]) => sum + n, 0);
  if (!total) return "";
  const bytePacked = counts
    .filter(([tag]) => ["U8", "I8"].includes(tag.toUpperCase()))
    .reduce((sum, [, n]) => sum + n, 0);
  const fp8 = counts
    .filter(([tag]) => FP8_TAGS.has(tag.toUpperCase()))
    .reduce((sum, [, n]) => sum + n, 0);
  if (bytePacked > total / 2) return fp8 ? "nvfp4" : "int4";
  if (fp8 > total / 2) return "fp8";
  return "";
}

/**
 * Logical parameter count from config dims, counting MoE experts. A **lower bound**.
 *
 * Counts attention, MLP and embeddings, so it misses norms, biases, quantization scales, MTP heads
 * and attention sinks; across the families on hand the Python's equivalent lands between 0.48x and
 * 1.0x of the truth. Reached only when neither the index nor the file sizes answered, which on the
 * Hub means a repo shipping no safetensors at all.
 *
 * Every layer is counted as sparse when the config declares experts, which over-states a hybrid
 * trunk whose first layers are dense. That is deliberate here: this rung exists to be a *lower*
 * bound, and over-stating weights makes the sizer refuse a card rather than recommend one that
 * OOMs. The Python resolves the per-layer question properly because it has the config object; this
 * has raw JSON and a dozen family-specific spellings, so it takes the safe direction instead.
 */
/** Every spelling of the routed-expert count, mirroring the engine's `facts._N_EXPERTS_FIELDS`. */
const EXPERT_COUNT_FIELDS = [
  "num_local_experts",
  "num_experts",
  "n_routed_experts",
  "moe_num_experts",
];

function configParamCount(config: Json, facts: TrunkDims): number {
  if (!facts.nLayers || !facts.dModel) return 0;
  const cfg = textConfig(config);
  const inter = firstInt(cfg, ["intermediate_size"]) || 4 * facts.dModel;
  const moeInter = firstInt(cfg, ["moe_intermediate_size"]) || inter;
  const nExperts = firstInt(cfg, EXPERT_COUNT_FIELDS);
  const nShared = firstInt(cfg, ["n_shared_experts", "num_shared_experts"]);
  const qDim = facts.nHeads * facts.headDim;
  const kvDim = facts.nKvHeads * facts.headDim;
  const attn = facts.dModel * (qDim + 2 * kvDim) + qDim * facts.dModel;
  const mlp = nExperts
    ? nExperts * 3 * facts.dModel * moeInter +
      nShared * 3 * facts.dModel * moeInter
    : 3 * facts.dModel * inter;
  const tied =
    cfg.tie_word_embeddings === true || config.tie_word_embeddings === true;
  const embeddings = facts.vocabSize * facts.dModel * (tied ? 1 : 2);
  return Math.round(facts.nLayers * (attn + mlp) + embeddings);
}

interface TrunkDims {
  nLayers: number;
  dModel: number;
  nHeads: number;
  nKvHeads: number;
  headDim: number;
  vHeadDim: number;
  vocabSize: number;
  intermediateSize: number;
  nExperts: number;
  layerTypes: string[] | null;
  slidingWindow: number | null;
  nResidualStreams: number;
  maxPositionEmbeddings: number;
  architecture: string;
  derivedDims: string[];
}

const NO_DIMS: TrunkDims = {
  nLayers: 0,
  dModel: 0,
  nHeads: 0,
  nKvHeads: 0,
  headDim: 0,
  vHeadDim: 0,
  vocabSize: 0,
  intermediateSize: 0,
  nExperts: 0,
  layerTypes: null,
  slidingWindow: null,
  nResidualStreams: 1,
  maxPositionEmbeddings: 0,
  architecture: "",
  derivedDims: [],
};

/** Mirrors the memory-relevant half of `interp_engine.facts.resolve_facts`. */
function trunkDims(config: Json): TrunkDims {
  const cfg = textConfig(config);
  const architectures = config.architectures ?? cfg.architectures;
  const nHeads = firstInt(cfg, ["num_attention_heads", "n_head"]);
  // `n_embed` is not a typo of `n_embd`: the first is BLOOM's spelling and the second is GPT-2's,
  // and transformers hides the difference behind `attribute_map` where the Python reads it. Raw JSON
  // has no such map, so a missing spelling here reads as `d_model = 0` -- which makes
  // `trunkDimsKnown` false and takes the whole family out of the sizer.
  const dModel = firstInt(cfg, ["hidden_size", "n_embd", "n_embed", "d_model"]);
  const statedHeadDim = firstInt(cfg, ["head_dim"]);
  const headDim = statedHeadDim || (nHeads ? Math.trunc(dModel / nHeads) : 0);
  const slidingWindow = firstInt(cfg, ["sliding_window"]);
  const kinds = layerTypes(cfg);
  // `n_positions` is the GPT-2 family's spelling, and transformers exposes it *as*
  // `max_position_embeddings` through `GPT2Config.attribute_map` -- so reading only the modern name
  // leaves gpt2 with no advertised context at all, and the sizer then recommends a 4096 default for
  // a model whose real limit is 1024.
  const context = firstInt(cfg, [
    "max_position_embeddings",
    "n_positions",
    "n_ctx",
  ]);
  const derivedDims: string[] = [];
  if (!statedHeadDim && headDim) derivedDims.push("head_dim");
  if (!context) derivedDims.push("max_position_embeddings");
  // Only worth naming on a model that says it has a window: a plain full-attention trunk needs no
  // per-layer table and loses nothing by having none.
  if (!kinds && slidingWindow) derivedDims.push("layer_types");
  return {
    nLayers: firstInt(cfg, ["num_hidden_layers", "n_layer"]),
    dModel,
    nHeads,
    nKvHeads: effectiveKvHeads(cfg, nHeads),
    headDim,
    vHeadDim: firstInt(cfg, ["v_head_dim"]) || headDim,
    vocabSize:
      firstInt(cfg, ["vocab_size"]) || firstInt(config, ["vocab_size"]),
    // `moe_intermediate_size` after the dense name rather than instead of it: a sparse family that
    // states both means the dense one for its dense layers, and one that states only the sparse name
    // is a trunk with no dense layer at all. DeepSeek-V4 is the second case, and reading only the
    // dense name left this at `4 * d_model` -- 8x the real per-expert width, and 8x the Python's.
    intermediateSize:
      firstInt(cfg, ["intermediate_size", "moe_intermediate_size"]) ||
      4 * dModel,
    nExperts: firstInt(cfg, EXPERT_COUNT_FIELDS),
    layerTypes: kinds,
    slidingWindow: slidingWindow || null,
    nResidualStreams: residualStreams(cfg),
    maxPositionEmbeddings: context,
    architecture:
      Array.isArray(architectures) && architectures.length
        ? String(architectures[0])
        : "",
    derivedDims,
  };
}

// ------------------------------------------------------------------ the ladder

/**
 * `hf_quant_config.json`, written by NVIDIA's ModelOpt. The one place a checkpoint states that its
 * **KV cache** is quantized as well as its weights, since these exports leave `config.json` with no
 * `quantization_config` at all.
 */
interface HfQuantConfig {
  quantization?: { quant_algo?: string; kv_cache_quant_algo?: string };
}

/** `model.safetensors.index.json`: the checkpoint's true size, and which files it is made of. */
interface ShardIndex {
  metadata?: { total_size?: number };
  /** Tensor name to shard filename. The distinct values are exactly this checkpoint's shards. */
  weight_map?: Record<string, string>;
}

/**
 * Resolve a model id to the facts a fit needs, without downloading weights.
 *
 * Makes three requests at most: the repo metadata, `config.json`, and the shard index. The first is
 * public even for a gated repo, so **weights resolve exactly with no token** and only the trunk
 * dims are lost — which is a far better prompt for a token than demanding one up front.
 *
 * Throws {@link HubError} only when the repo itself could not be read. A gated or missing
 * `config.json` is not an error: it returns facts with `trunkDimsKnown` false and a note saying so.
 */
export async function resolveModel(
  modelId: string,
  options: FetchOptions = {},
): Promise<ModelMemoryFacts> {
  const id = modelId.trim().replace(/^\/+|\/+$/g, "");
  if (!id) throw new HubError("no model id given", 0);

  const info = await fetchRepoInfo(id, options);
  const notes: string[] = [];

  const [configResult, indexResult] = await Promise.all([
    fetchRepoJson<Json>(id, "config.json", options),
    fetchRepoJson<ShardIndex>(id, "model.safetensors.index.json", options),
  ]);

  const config = configResult.data;
  const dims = config ? trunkDims(config) : { ...NO_DIMS };
  if (!config) {
    notes.push(
      configResult.status === 401 || configResult.status === 403
        ? "config.json is behind this repo's licence, so the KV-cache term falls back to the " +
            "pre-GQA worst case. A token sharpens it; the weight figure needs none."
        : `config.json could not be read (HTTP ${configResult.status}), so no trunk dims are known.`,
    );
  }

  /**
   * Fold what the tensor shapes know back into `dims`.
   *
   * Called wherever headers get read, so the two reasons to read them share one result. Anything it
   * recovers is dropped from `derivedDims`, which is why the notes below are emitted afterwards: a
   * field that has been read off the checkpoint is no longer derived, and warning about it would be
   * a caveat on a number that is now exact.
   */
  const recoverDims = (shapes: Record<string, number[]>) => {
    const found = dimsFromShapes(shapes, dims.nKvHeads);
    if (!dims.vocabSize && found.vocabSize) {
      dims.vocabSize = found.vocabSize;
      notes.push(
        `config.json states no vocab_size, so it was read from the checkpoint: the embedding ` +
          `matrix is ${found.vocabSize.toLocaleString("en-US")} rows. This is the term the eager ` +
          `logits are priced from, and pricing it at zero is the direction that hides an OOM.`,
      );
    }
    if (found.headDim && dims.derivedDims.includes("head_dim")) {
      const wasMirroring = dims.vHeadDim === dims.headDim;
      dims.headDim = found.headDim;
      if (wasMirroring) dims.vHeadDim = found.headDim;
      dims.derivedDims = dims.derivedDims.filter((f) => f !== "head_dim");
      notes.push(
        `config.json omits head_dim; the key projection in the checkpoint gives ${found.headDim}, ` +
          `so the KV figures are exact rather than derived from hidden_size / heads.`,
      );
    }
  };

  // A config that omits a dim is not a config that has no dim: transformers fills it from the
  // model class, which raw JSON cannot see. Each one errs in a known direction, so say which.
  const MISSING_DIM_NOTES: Record<string, string> = {
    head_dim:
      "config.json omits head_dim, so it was derived as hidden_size / num_attention_heads. " +
      "Some families set a different value in their model class (gemma-3-12b's is 256 where the " +
      "division gives 240), and a derived value that is too small UNDER-states the KV cache.",
    max_position_embeddings:
      "config.json omits max_position_embeddings, so the model's advertised context is unknown " +
      "and a max_model_len has to be chosen rather than defaulted.",
    layer_types:
      "this model declares a sliding window but no per-layer table, so every layer is priced as " +
      "if it caches the full context. That over-states the KV cache, often by a lot: a 5:1 " +
      "sliding trunk really caches the window on five layers in six.",
  };

  // The Hub's config extract fills the architecture in where the real config was gated.
  const apiConfig = asRecord(info.config) ?? {};
  if (!dims.architecture) {
    const architectures = apiConfig.architectures;
    if (Array.isArray(architectures) && architectures.length) {
      dims.architecture = String(architectures[0]);
    }
  }

  // Scheme, most authoritative first. The config extract sits second because it is a curated subset
  // but is readable when the real config is not, which is the gated-and-quantized case. Header
  // inference is last, per `schemeFromHeaders`. The Python asks its sidecar files between the
  // config and the headers; here the headers come first because they are already fetched and reach
  // the same *width* for every scheme a sidecar names — only the label differs.
  // Per-dtype element counts. The Hub's aggregate is free and correct for the ordinary repo, but it
  // spans every `.safetensors` in the repo, so where a variant set exists the shards' own headers
  // are the only honest source -- see `readShardHeaders`.
  let elementsByDtype: Record<string, number> = {
    ...(info.safetensors?.parameters ?? {}),
  };
  // Which of the two answered, because they do not mean the same thing. A header count is *stored
  // elements* and has to be unpacked; the Hub's aggregate is already a parameter count on the repos
  // that need unpacking most -- it reports DeepSeek V4's fp4 experts as 283.5e9 against 141.7e9
  // bytes on disk, and drops the ue8m0 scales. Unpacking that a second time would double a 291B
  // model, so `expert_dtype` is only handed to `logicalParamCount` on the header path.
  let fromHeaders = false;
  if (hasVariantShards(info)) {
    const fromIndex = [
      ...new Set(Object.values(indexResult.data?.weight_map ?? {})),
    ];
    const shards = fromIndex.length
      ? fromIndex
      : rootShards(info).map((s) => s.rfilename);
    const measured = await readShardHeaders(id, shards, options);
    if (Object.keys(measured.elements).length) {
      elementsByDtype = measured.elements;
      fromHeaders = true;
      recoverDims(measured.shapes);
    } else {
      elementsByDtype = {};
      notes.push(
        "this repo ships a second .safetensors set in a subdirectory, so the Hub's per-dtype " +
          "totals describe more than the served checkpoint, and the shard headers could not be " +
          "read to replace them. The parameter count is derived from the packed bytes instead, " +
          "which over-states what a dequantizing load would cost.",
      );
    }
  } else if (
    config &&
    (!dims.vocabSize || dims.derivedDims.includes("head_dim"))
  ) {
    // The other reason to read a header: the config parsed, but left a field a model class would have
    // filled. One or two shards rather than the set, since `weight_map` says which hold the tensors
    // that answer -- a few KB against a term that is otherwise priced at zero.
    const shards = dimBearingShards(info, indexResult.data);
    if (shards.length) {
      recoverDims((await readShardHeaders(id, shards, options)).shapes);
    }
  }

  // After the recovery above, so a field read off the checkpoint does not also carry a warning that
  // it was guessed.
  for (const field of dims.derivedDims) {
    const note = MISSING_DIM_NOTES[field];
    if (note) notes.push(note);
  }

  let quantMethod = config ? configQuantMethod(config) : "";
  if (!quantMethod) {
    const q = asRecord(apiConfig.quantization_config);
    const method = q?.quant_method ?? q?.quantization;
    if (typeof method === "string" && method)
      quantMethod = method.toLowerCase();
  }
  if (!quantMethod) quantMethod = schemeFromHeaders(elementsByDtype);

  // NVIDIA's ModelOpt exports declare both schemes in a sidecar and leave `config.json` with no
  // `quantization_config` at all, so this is the only place the KV width is stated. One small JSON,
  // fetched only when the weights are quantized -- a dense checkpoint has no such file and asking
  // for it would spend a round trip per model to learn nothing.
  let kvQuantAlgo = "";
  if (quantMethod) {
    const sidecar = await fetchRepoJson<HfQuantConfig>(
      id,
      "hf_quant_config.json",
      options,
    );
    const algo = sidecar.data?.quantization?.kv_cache_quant_algo;
    if (typeof algo === "string" && algo) kvQuantAlgo = algo.toLowerCase();
  }

  const storedDtype = config ? configStoredDtype(config) : "";
  const expertDtype = config
    ? configExpertDtype(config)
    : configExpertDtype(apiConfig as Json);

  // Rung 1: the shard index's own total. Exact, and scoped to this checkpoint rather than the repo.
  let onDiskBytes = Math.trunc(indexResult.data?.metadata?.total_size ?? 0);
  let source: WeightSource = onDiskBytes ? "safetensors-index" : "unknown";

  // Rung 2: the file sizes, which need no token even where the index did.
  if (!onDiskBytes) {
    const shards = rootShards(info);
    onDiskBytes = shards.reduce((sum, shard) => sum + (shard.size ?? 0), 0);
    if (onDiskBytes) {
      source = "file-sizes";
      if (indexResult.status === 401 || indexResult.status === 403) {
        notes.push(
          "the shard index is gated too, so the weight total is the sum of the root shard sizes " +
            "instead — same number, read a different way.",
        );
      }
    }
  }

  // Rung 3: the headers alone. Element counts times their own container widths IS the stored size.
  if (!onDiskBytes && Object.keys(elementsByDtype).length) {
    onDiskBytes = Math.round(
      Object.entries(elementsByDtype).reduce(
        (sum, [tag, count]) =>
          sum +
          count * dtypeBytes(tag, CONTAINER_DTYPES[tag.toUpperCase()] ?? 2.0),
        0,
      ),
    );
    if (onDiskBytes) source = "safetensors-headers";
  }

  let paramCount = Object.keys(elementsByDtype).length
    ? logicalParamCount(
        elementsByDtype,
        quantMethod,
        fromHeaders ? expertDtype : "",
      )
    : 0;

  // Rung 4: no shard metadata at all. Count the dims, which is a lower bound and labelled as one.
  if (!onDiskBytes && config) {
    const counted = configParamCount(config, dims);
    if (counted) {
      paramCount = counted;
      onDiskBytes = Math.round(
        counted * (schemeWidth(quantMethod) ?? dtypeBytes(storedDtype)),
      );
      source = "config-count";
      notes.push(
        "no shard metadata was readable, so the weight figure is counted from the config dims. " +
          "It omits norms, biases and quantization scales, so treat it as a floor.",
      );
    }
  }

  if (!paramCount && onDiskBytes) {
    // Bytes but no headers. Divide by the width one parameter really occupies, which for a packed
    // checkpoint is the scheme's rather than the container's: reading MXFP4 bytes as bf16 elements
    // halves the count, and this number is what prices a dequantizing load.
    paramCount = Math.round(
      onDiskBytes / (schemeWidth(quantMethod) ?? dtypeBytes(storedDtype)),
    );
  }

  if (!onDiskBytes) {
    notes.push(
      `nothing in ${id} reported a weight size, so no memory figure can be trusted.`,
    );
  }

  return {
    modelId: id,
    weights: {
      paramCount,
      onDiskBytes,
      storedDtype,
      quantMethod,
      expertDtype,
      elementsByDtype,
      source,
    },
    ...dims,
    kvQuantAlgo,
    gated: info.gated ?? false,
    trunkDimsKnown: dims.nLayers > 0 && dims.dModel > 0,
    notes,
  };
}

/**
 * The `ModelMemoryFacts` fields a KV figure depends on, and whether any was guessed.
 *
 * A convenience for the one question a sizer has to ask before quoting a context length, since
 * both halves of it are easy to forget: the dims may be absent entirely (gated, no token) or
 * present but partly derived (a config that leans on its model class).
 */
export function kvPrecision(
  facts: ModelMemoryFacts,
): "exact" | "derived" | "unknown" {
  if (!facts.trunkDimsKnown) return "unknown";
  return facts.derivedDims.includes("head_dim") ||
    facts.derivedDims.includes("layer_types")
    ? "derived"
    : "exact";
}

/**
 * KV-cache elements per token per layer, K and V together.
 *
 * `2 * dModel` — the fallback when no head dims are known — is the *pre-GQA* worst case and is
 * wrong by 8x on the models where sizing is tight, which is why it is a fallback and why
 * {@link ModelMemoryFacts.trunkDimsKnown} exists to be checked first.
 */
export function kvCacheWidth(facts: ModelMemoryFacts): number {
  if (facts.nKvHeads && facts.headDim) {
    return facts.nKvHeads * (facts.headDim + (facts.vHeadDim || facts.headDim));
  }
  return 2 * facts.dModel;
}

/** Layers that cache the whole context. Everything else caches a window. */
export function fullAttentionLayers(facts: ModelMemoryFacts): number {
  if (!facts.layerTypes) return facts.nLayers;
  return facts.layerTypes
    .slice(0, facts.nLayers)
    .filter((kind) => !kind.includes("sliding") && !kind.includes("linear"))
    .length;
}
