"use client";

/**
 * The sizer's body: name a model, and see which cards will run it and how.
 *
 * Two different costs, handled two different ways. Resolving a model is a handful of requests to the
 * Hub, so it happens when asked — on Enter or the button — and not on a keystroke. Pricing the result
 * is arithmetic over numbers already in hand, so every control below reprices immediately, which is
 * the reason `lib/size.ts` is a port rather than an API call.
 *
 * Nothing here is a claim this app makes on its own: the catalog and the calibration constants come
 * from `interp_engine.memory` through `data/gpus.generated.ts`, the arithmetic is checked against the
 * Python by `scripts/check-size.ts`, and any row the verification harness has actually run on a real
 * card says so rather than being presented as an estimate.
 */

import {
  AlertTriangle,
  ChevronRight,
  Circle,
  Loader2,
  Search,
} from "lucide-react";
import { Popover } from "radix-ui";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { CopyButton } from "@/components/CopyButton";
import { VramBar } from "@/components/sizer/VramBar";
import { CALIBRATION, GPUS } from "@/data/gpus.generated";
import {
  GIB,
  concurrentSequences,
  estimate,
  fitAcross,
  isVllm,
  jacobianLens,
  offeredStaticPoints,
  reservations,
  resolvedStaticPoints,
  snippet,
  staticPointElements,
  totalBytes,
  totalGib,
  workload,
  type Backend,
  type FitResult,
  type Reservations,
} from "@/lib/size";
import {
  HubError,
  kvPrecision,
  setHubToken,
  type ModelMemoryFacts,
} from "@/lib/hub";
import { cachedModelIds } from "@/lib/models";
import { resolveFacts } from "@/lib/resolve";
import {
  byTier,
  defaultTier,
  shortGpuName,
  tierEvidence,
  tierGpus,
  type Tier,
} from "@/lib/tiers";

/** What each backend costs and gives, from `docs/PERFORMANCE.md` by way of `gpu-sizer/fit.py`. */
const BACKEND_NOTE: Record<Backend, string> = {
  vllm: "default engine, a balance of speed and memory required",
  "vllm-static":
    "adds a fixed tap buffer per site and graph pool for faster decode",
  "vllm-generate": "graphs without the static taps: fast decode, no capture",
  eager: "transformers. quantized checkpoints (eg dsv4) can dequantize on load",
};

/**
 * The three engines this offers, which is not all of `BACKENDS`.
 *
 * `vllm-generate` is left out on purpose: it replays CUDA graphs with no taps at
 * all, so `hooks_available` is False and every capture, steer and lens entry
 * point refuses. It is the fastest of the four and cheaper than `vllm-static`,
 * which is exactly the problem — on two bars it reads as the free win, and what
 * it costs is the reason anyone is on this page. Sizing a machine for it is a
 * question about a completion pod, not about interp.
 */
const OFFERED_BACKENDS: Backend[] = ["vllm", "vllm-static", "eager"];

/**
 * The rough shape of each backend, for the two bars beside its name. Nothing
 * here is computed — the figures in the third column are the computed ones, for
 * this model on this card. This is the standing ranking the engine's own docs
 * publish (`docs-site/docs/loading-models.md`), so that the choice can be made
 * before a model is even resolved.
 *
 * Three rungs. `vllm` is the cheapest on memory — `enforce_eager=True` means no
 * graph pool and no tap buffers — and `vllm-static` buys 4-11x decode with a ~3
 * GiB pool plus a buffer per tap site per row. `eager` is slowest by far and
 * cheapest of all on memory, because it has no pool at all.
 *
 * `risk` is the rest of that bar, drawn rather than left empty. Eager is the
 * bottom rung *most* of the time and the top rung on a quantized checkpoint,
 * and a bar has to pick one — pick the high rung and every unquantized model is
 * libelled, pick the low one and the 3.6x is somewhere nobody looks. So the
 * floor is where it usually sits and the amber is how far it goes, which is the
 * one shape that is not a lie in either direction.
 */
interface Risk {
  /** The track's remainder above the floor rung: `h-9` less the `h-2` under it. */
  height: string;
  title: string;
  body: string;
}

const BACKEND_BARS: Record<
  Backend,
  { speed: number; vram: number; risk?: Risk }
> = {
  vllm: { speed: 2, vram: 1 },
  "vllm-static": { speed: 3, vram: 3 },
  "vllm-generate": { speed: 3, vram: 3 },
  eager: {
    speed: 1,
    vram: 1,
    risk: {
      height: "h-7",
      title: "A quantized checkpoint expands on load",
      body: "transformers reads dtype as the width to materialize the weights in, rather than as an activation dtype, so a packed checkpoint is unpacked on the way in: DeepSeek-V4's fp8 goes 155 → 567 GiB, and gpt-oss-20b's MXFP4 12.8 → 41.2. It also happens silently whenever the fp8 or MXFP4 kernels are missing, whatever dtype was asked for. An unquantized checkpoint, or a quantized one at dtype=auto with its kernels installed, stays at the bottom of the bar.",
    },
  },
};

/** Indexed by rung, one to three. Literals, so the classes survive the scan. */
const BAR_HEIGHTS = ["h-2", "h-5", "h-9"];

/**
 * Three models to start from, so the first thing on the page is not an empty
 * box asking for an id in a format nobody has memorised.
 *
 * The short label is what people call these, and the id is what the Hub calls
 * them — a one-off mapping rather than anything derived, because "llama-70b-it"
 * naming the 3.3 checkpoint is an editorial choice and not a rule.
 *
 * All three are in `data/models.generated.ts`, so a click resolves out of the
 * cache with no Hub round trip and no token. `scripts/build-model-cache.ts`
 * lists them in `EXTRA` to keep it that way — one of them is gated, and a quick
 * pick that asks for a token would be worse than no quick pick.
 */
const QUICK_PICKS = [
  { label: "gemma-2-9b-it", id: "google/gemma-2-9b-it" },
  { label: "qwen3.6-27b", id: "Qwen/Qwen3.6-27B" },
  { label: "llama-70b-it", id: "meta-llama/Llama-3.3-70B-Instruct" },
  { label: "deepseek-v4-flash", id: "deepseek-ai/DeepSeek-V4-Flash" },
];

const DTYPES = ["auto", "bfloat16", "float16", "float32"];

const CONTEXT_MODES = [
  { value: "auto", label: "auto" },
  { value: "manual", label: "manual" },
];

const LENS_MODES = [
  { value: "on", label: "Load J-Lens" },
  { value: "off", label: "No J-Lens" },
];

/**
 * The spinner off a number field. It is a control for walking a value one step
 * at a time, and nobody reaches 32768 tokens in 1024-token clicks — what it
 * really does here is put two hit targets where the digits are, which is also
 * where the unit now sits.
 */
const NO_STEPPER =
  "[appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none";

/** The heading names the model, and the org prefix is the half of an id nobody says out loud. */
const shortModelName = (modelId: string) =>
  modelId.slice(modelId.indexOf("/") + 1);

const gib = (bytes: number) => `${(bytes / GIB).toFixed(1)} GiB`;
const num = (value: number) => Math.round(value).toLocaleString("en-US");

/**
 * The dtype the CLI would pick when asked for none: as stored for a quantized checkpoint, so it is
 * priced as vLLM serves it, and bf16 otherwise. Not the *engine's* default, which for the eager
 * backend is float32 and would double a bf16 checkpoint — a sizer should say that rather than
 * reproduce it silently.
 */
function recommendedDtype(facts: ModelMemoryFacts): string {
  return facts.weights.quantMethod ? "auto" : "bfloat16";
}

export function Sizer({
  initialModel = "",
  onModelChange,
}: {
  /** A model named by the URL, resolved once on mount. */
  initialModel?: string;
  /** The resolved id, or empty when there is none. Must be stable across renders. */
  onModelChange?: (modelId: string) => void;
} = {}) {
  const [query, setQuery] = useState(initialModel);
  const [token, setToken] = useState("");
  const [facts, setFacts] = useState<ModelMemoryFacts | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [backend, setBackend] = useState<Backend>("vllm");
  // Empty means "whatever this checkpoint calls for", resolved at use.
  const [dtypeChoice, setDtypeChoice] = useState("");
  const [context, setContext] = useState(0);
  // Empty means `"auto"`, which the trunk resolves rather than this component: a hyper-connection
  // block has no single `resid_post` to tap, so the default there is the whole stream stack.
  const [staticPoints, setStaticPoints] = useState<string[]>([]);
  const [reserveGib, setReserveGib] = useState(0);
  const [lens, setLens] = useState(false);
  const [selected, setSelected] = useState("");
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const dtype = facts ? dtypeChoice || recommendedDtype(facts) : "bfloat16";

  // Also what warms the cache chunk: the suggestions and the lookup read the same module, so asking
  // for the ids on mount means it has landed before anyone finishes typing an id.
  useEffect(() => {
    let live = true;
    void cachedModelIds().then((ids) => {
      if (live) setSuggestions(ids);
    });
    return () => {
      live = false;
    };
  }, []);

  // Through a ref because `resolve` is rebuilt every render and listing it would re-run the arrival.
  const resolveRef = useRef<(id: string) => void>(() => {});
  useEffect(() => {
    if (initialModel) resolveRef.current(initialModel);
  }, [initialModel]);

  useEffect(() => {
    onModelChange?.(facts?.modelId ?? "");
  }, [facts, onModelChange]);

  function reset() {
    setQuery("");
    setFacts(null);
    setError("");
    setBackend("vllm");
    setDtypeChoice("");
    setContext(0);
    setStaticPoints([]);
    setReserveGib(0);
    setLens(false);
    setSelected("");
  }

  resolveRef.current = (id: string) => void resolve(id);

  async function resolve(id: string) {
    const wanted = id.trim();
    if (!wanted || busy) return;
    setBusy(true);
    setError("");
    setHubToken(token || null);
    try {
      const resolved = await resolveFacts(wanted);
      setFacts(resolved);
      // A model whose advertised context could not be read needs one chosen rather than defaulted,
      // and 8192 is a length worth pricing rather than a guess at the model's limit.
      setContext(
        resolved.derivedDims.includes("max_position_embeddings") ? 8192 : 0,
      );
      setDtypeChoice("");
      setSelected("");
    } catch (cause) {
      setFacts(null);
      setError(
        cause instanceof HubError
          ? cause.message
          : cause instanceof Error
            ? cause.message
            : "could not reach the Hub",
      );
    } finally {
      setBusy(false);
    }
  }

  const res = useMemo(() => {
    if (!facts) return reservations();
    const base = lens ? jacobianLens(facts) : reservations();
    if (!reserveGib) return base;
    return reservations({
      perRankBytes: base.perRankBytes + Math.round(reserveGib * GIB),
      note: [base.note, `${reserveGib} GiB reserved for your own tensors`]
        .filter(Boolean)
        .join(" + "),
    });
  }, [facts, lens, reserveGib]);

  const results = useMemo(
    () =>
      facts
        ? fitAcross(facts, {
            backend,
            dtype,
            maxModelLen: context,
            staticPoints,
            res,
          })
        : [],
    [facts, backend, dtype, context, staticPoints, res],
  );

  const tiers = useMemo(() => byTier(results), [results]);

  const chosen =
    results.find((result) => result.gpu.name === selected) ??
    defaultTier(tiers)?.result;

  return (
    /* `grid-rows` pinned to one `1fr` track rather than left on `auto`: an auto
       row grows to its tallest item, which would push the bottom of both columns
       past the window instead of handing the overflow to the columns to scroll. */
    <div className="grid lg:h-full lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:grid-rows-[minmax(0,1fr)]">
      <div className="thin-scrollbar flex flex-col gap-y-4 bg-white px-4 py-5 lg:min-h-0 lg:overflow-y-auto lg:pr-6">
        <ColumnLabel>1️⃣ Choose a Model</ColumnLabel>

        <ModelField
          query={query}
          onQuery={setQuery}
          busy={busy}
          onSubmit={(value) => resolve(value)}
          onReset={facts ? reset : undefined}
          suggestions={suggestions}
          current={facts?.modelId ?? ""}
        />

        {error && (
          <p className="flex items-start gap-x-2 rounded-md bg-red-50 px-3 py-2 text-[11px] leading-relaxed text-red-700">
            <AlertTriangle className="mt-px h-3.5 w-3.5 shrink-0" />
            {error}
          </p>
        )}

        {facts && <ModelSummary facts={facts} />}

        <TokenOverride token={token} onToken={setToken} />
      </div>

      {/* Kept in the tree before a model resolves, so the headings and the rules
          between the columns say where the answer is going to appear. Held back
          to `lg`, because below it the columns stack and the rules are gone — a
          heading over nothing would be all that was left of the promise.

          Splits again at `xl` into the knobs and what they produce. Not at `lg`:
          three columns in 1024px leaves the results table about 350px, and it
          carries five of its own. Under that width the knobs sit above the
          table instead, and this wrapper does the scrolling for both. */}
      <div
        className={`thin-scrollbar min-w-0 flex-col lg:flex lg:min-h-0 lg:overflow-y-auto lg:border-l lg:border-slate-200 xl:grid xl:grid-cols-[minmax(0,20rem)_minmax(0,1fr)] xl:grid-rows-[minmax(0,1fr)] xl:overflow-hidden ${
          facts ? "flex" : "hidden"
        }`}
      >
        <div className="thin-scrollbar flex flex-col gap-y-4 px-4 pt-5 lg:pl-6 xl:min-h-0 xl:overflow-y-auto xl:pr-6 xl:pb-5">
          <ColumnLabel>2️⃣ Set Speed & Configs</ColumnLabel>

          {!facts && <Waiting />}

          {facts && (
            <Controls
              facts={facts}
              backend={backend}
              onBackend={setBackend}
              dtype={dtype}
              onDtype={setDtypeChoice}
              staticPoints={staticPoints}
              onStaticPoints={setStaticPoints}
              context={context}
              onContext={setContext}
              reserveGib={reserveGib}
              onReserve={setReserveGib}
              lens={lens}
              onLens={setLens}
            />
          )}
        </div>

        <div className="thin-scrollbar flex min-w-0 flex-col gap-y-4 px-4 pt-5 pb-5 lg:pl-6 xl:min-h-0 xl:overflow-y-auto xl:border-l xl:border-slate-200 xl:pl-6">
          <ColumnLabel>
            {facts ? (
              <>
                3️⃣ Results: GPUs that Fit <Token>{shortModelName(facts.modelId)}</Token>{" "}
                on interp-engine <Token>{backend}</Token> backend
              </>
            ) : (
              "3️⃣ Results: GPUs that Fit"
            )}
          </ColumnLabel>

          {!facts && <Waiting />}

          {facts && (
            <>
              <Results
                facts={facts}
                backend={backend}
                dtype={dtype}
                tiers={tiers}
                selected={chosen?.gpu.name ?? ""}
                onSelect={setSelected}
              />

              {chosen ? (
                <Detail
                  facts={facts}
                  result={chosen}
                  backend={backend}
                  res={res}
                />
              ) : (
                facts.trunkDimsKnown && (
                  <NoFit
                    facts={facts}
                    backend={backend}
                    dtype={dtype}
                    context={context}
                    staticPoints={staticPoints}
                    res={res}
                  />
                )
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * What a column says before a model resolves. The headings are already in the
 * tree by then to promise where the answer lands, and this is the other half of
 * that promise: it names the one thing that has to happen for the column to
 * fill, so an empty pane reads as waiting rather than as broken.
 */
function Waiting() {
  return (
    <p className="text-[11px] text-slate-400">Choose Model to Get Started</p>
  );
}

function ColumnLabel({ children }: { children: ReactNode }) {
  return (
    <h3 className="font-heading -mb-1 text-sm font-semibold text-slate-800">
      {children}
    </h3>
  );
}

/**
 * The two things in the results heading that change with the controls: which model, and which
 * engine. Everything either side of them is fixed text, so a heading in one weight makes the reader
 * re-read the whole line to find the half that just moved.
 *
 * Dashed rather than solid, and white on a white column, so it reads as a slot the controls fill
 * rather than as a button. A hair smaller than the heading because mono runs wide at the same size
 * and would otherwise outweigh the words it sits between.
 *
 * The noun each one names — "backend" — stays outside the border, since what the controls change is
 * the value and a slot drawn around both would be promising to fill a word that never moves.
 */
function Token({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-md border border-dashed border-sky-700/50 bg-white px-1.5 py-0.5 font-mono text-[13px] font-bold whitespace-nowrap text-sky-700">
      {children}
    </span>
  );
}

function ModelField({
  query,
  onQuery,
  busy,
  onSubmit,
  onReset,
  suggestions,
  current,
}: {
  query: string;
  onQuery: (value: string) => void;
  busy: boolean;
  onSubmit: (value: string) => void;
  onReset?: () => void;
  suggestions: string[];
  /** The resolved model, for marking whichever quick pick is showing. */
  current: string;
}) {
  return (
    <div>
      <div>
        <div className="mb-2 grid grid-cols-2 gap-1.5">
          {QUICK_PICKS.map((pick) => {
            // Against the resolved id rather than the field's text, so the mark
            // follows what is loaded and not what someone has half-typed over
            // it. Case-insensitive, because the Hub is.
            const active = current.toLowerCase() === pick.id.toLowerCase();
            return (
              <button
                key={pick.id}
                type="button"
                disabled={busy}
                onClick={() => {
                  onQuery(pick.id);
                  onSubmit(pick.id);
                }}
                title={pick.id}
                className={`cursor-pointer truncate rounded-md border px-2 py-2.5 font-mono text-[11px] font-medium transition-colors disabled:pointer-events-none disabled:opacity-40 ${
                  active
                    ? "border-sky-700 bg-sky-200 text-sky-700"
                    : "border-slate-300 bg-white text-slate-600 hover:border-sky-600 hover:text-sky-700"
                }`}
              >
                {pick.label}
              </button>
            );
          })}
        </div>

        <div className="relative min-w-0">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
          <input
            value={query}
            // Clicking a suggestion resolves it, without a second trip to Find.
            // There is no event for picking one, so it is inferred: a pick
            // carries no keystroke behind it -- Firefox calls it
            // `insertReplacementText` and Chrome reports no `inputType` at all --
            // and it lands on an entry exactly. Both halves are needed, since
            // pasting an id that happens to be on the list also arrives whole,
            // and resolving on a paste would take the decision away from you.
            onChange={(event) => {
              const value = event.target.value;
              onQuery(value);
              const how = (event.nativeEvent as InputEvent).inputType;
              if (how && how !== "insertReplacementText") return;
              if (!suggestions.includes(value)) return;
              onSubmit(value);
            }}
            // Deferred by one task, and that deferral is the whole handler rather than a caution.
            // Accepting a datalist suggestion with Enter dispatches this keydown BEFORE the browser
            // writes the suggestion into the field, so there is no value to read yet -- not in
            // `query`, and not off the element either. Submitting what is there resolves the
            // fragment that was typed ("no model called google/gemma-3-12b-"), and because that
            // sets `busy`, the real id arriving a moment later through `onChange` is dropped and the
            // error stays on screen. Worse, `onQuery` with the fragment fights the browser's own
            // commit, since the field is controlled.
            //
            // One turn of the event loop later the field holds what was picked. The `onChange` path
            // above may have submitted it already; that call wins and this one is dropped as busy,
            // which is the same id either way.
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              const field = event.currentTarget;
              setTimeout(() => {
                onQuery(field.value);
                onSubmit(field.value);
              }, 0);
            }}
            placeholder="HF Model ID (google/gemma-2-9b-it)"
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            list="sizer-model-ids"
            aria-label="Hugging Face model id"
            className="w-full rounded-md border border-slate-300 bg-white py-2 pr-3 pl-8 font-mono text-[12px] text-slate-800 placeholder:text-slate-400 focus:border-sky-600 focus:outline-none"
          />
          {/* A native `datalist` rather than a combobox: any id the Hub knows is valid here, so the
              list is a shortcut and never a constraint, and the one thing a custom popup would add
              is a way to get in the way of typing an id that is not on it. */}
          <datalist id="sizer-model-ids">
            {suggestions.map((id) => (
              <option key={id} value={id} />
            ))}
          </datalist>
        </div>

        <div className="mt-2 flex gap-x-2">
          <button
            type="button"
            onClick={() => onSubmit(query)}
            disabled={busy || !query.trim()}
            className="flex flex-1 cursor-pointer items-center justify-center gap-x-1.5 rounded-md bg-sky-700 py-2 text-[12px] font-medium text-white transition-colors hover:bg-sky-800 disabled:pointer-events-none disabled:opacity-40"
          >
            {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {busy ? "Reading" : "Find"}
          </button>
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="flex-1 cursor-pointer rounded-md bg-rose-700 py-2 text-[12px] font-medium text-white transition-colors hover:bg-rose-800"
            >
              Reset
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Somewhere to put your own Hub token, folded away until asked for.
 *
 * Below the summary rather than beside the field, because it is not part of
 * naming a model. Most readers never need it: the sixty models in the cache
 * answer without one, gated ones included, and an unlisted public repo resolves
 * anonymously. What is left is a private repo, or a gated one nobody has
 * pre-resolved — and that reader arrives *after* a refusal, which is where this
 * now sits.
 */
function TokenOverride({
  token,
  onToken,
}: {
  token: string;
  onToken: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((was) => !was)}
        className="w-full cursor-pointer rounded-md border border-slate-300 bg-slate-100 px-3 py-2 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-200 hover:text-slate-800"
      >
        {open ? "Hide HF Token" : "Override HF Token"}
      </button>
      <p className="mt-1 text-center text-[10px] text-slate-400">
        for private/gated models
      </p>

      {open && (
        <div className="mt-2">
          <input
            value={token}
            onChange={(event) => onToken(event.target.value)}
            type="password"
            placeholder="hf_..."
            spellCheck={false}
            aria-label="Hugging Face token"
            className="w-full rounded-md border border-slate-300 bg-white px-2.5 py-1.5 font-mono text-[11px] text-slate-800 placeholder:text-slate-400 focus:border-sky-600 focus:outline-none"
          />
          {/* The commitment `lib/hub.ts` makes, stated where the token is asked for rather than only
              in the module that keeps it. */}
          <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
            Held in this tab only — never stored, and sent to huggingface.co and
            nowhere else, skipping this site&rsquo;s shared token and the server
            that holds it. Only needed for a repo that token cannot see.
          </p>
        </div>
      )}
    </div>
  );
}

/** Safetensors dtype tags as the name someone would type into `--model_dtype`. */
const TAG_NAMES: Record<string, string> = {
  BF16: "bfloat16",
  F16: "float16",
  F32: "float32",
  F8_E4M3: "fp8 e4m3",
  F8_E5M2: "fp8 e5m2",
};

/**
 * Schemes that name a container rather than a precision.
 *
 * `compressed-tensors` holds fp8, int8 or int4 depending on how it was written, so the scheme on its
 * own does not answer "which dtype" — and printing it under a label that says `dtype` is the one
 * answer that is certainly wrong. Here the headers can be trusted, because the narrow tag is the
 * *majority* of the checkpoint. That majority is exactly what makes this safe where the general rule
 * is not: nvfp4 carries an `F8_E4M3` tag too, but as a small minority, since those are block scales
 * rather than weights.
 */
const CONTAINER_SCHEMES = new Set(["compressed-tensors"]);

/**
 * What the weights actually are, which is not what the config says they are.
 *
 * The scheme wins wherever there is a real one, and the reason is that on a packed checkpoint the
 * header tags name *containers* rather than dtypes: DeepSeek V4 ships its 256 routed experts as fp4
 * inside an int8 array, so `I8` covers 283 of its 291 billion parameters. Reading the dominant tag
 * there answers "int8", which is true of the file and false of the model.
 *
 * A mixed-precision checkpoint gets both halves, because the scheme alone names the smaller one:
 * DeepSeek V4's `quant_method` is `fp8` and that covers its attention and dense layers, while the
 * routed experts it spends most of its parameters on are fp4. Which half is which does not fit the
 * cell, so it goes in the hover.
 *
 * Where nothing declares a scheme the tag gets to speak, and then it is the better source —
 * `config.dtype` is absent often enough (gpt2, starcoder2 and gpt-oss-20b all declare none) that the
 * headers recover a dtype the config cannot give.
 */
function weightsDtype(facts: ModelMemoryFacts): {
  text: string;
  title?: string;
} {
  const scheme = facts.weights.quantMethod;
  const experts = facts.weights.expertDtype;
  if (scheme && experts && experts !== scheme)
    return {
      text: `${scheme} + ${experts}`,
      title: `${scheme} attention and dense layers, ${experts} routed experts`,
    };
  if (scheme && !CONTAINER_SCHEMES.has(scheme)) return { text: scheme };

  const tags = Object.entries(facts.weights.elementsByDtype);
  if (!tags.length) return { text: facts.weights.storedDtype || "?" };

  const [tag] = tags.reduce((widest, entry) =>
    entry[1] > widest[1] ? entry : widest,
  );
  return { text: TAG_NAMES[tag] ?? tag.toLowerCase() };
}

function ModelSummary({ facts }: { facts: ModelMemoryFacts }) {
  const precision = kvPrecision(facts);
  const weights = weightsDtype(facts);
  const [showNotes, setShowNotes] = useState(false);
  return (
    <div className="@container rounded-md border border-slate-200 bg-slate-50 px-3 py-2.5">
      {/* No quantization badge: `dtype (weights)` below already reports the scheme, and reports it
          better -- a checkpoint whose routed experts are narrower than the rest of it reads there as
          `fp8 + fp4`, which the badge flattened to `fp8`. */}
      <div className="flex flex-wrap items-baseline justify-center gap-x-2 gap-y-1">
        <span className="font-mono text-[12px] font-semibold text-slate-800">
          {facts.modelId}
        </span>
        {facts.architecture && (
          <span className="text-[10px] text-slate-500">
            {facts.architecture}
          </span>
        )}
      </div>

      {/* A container query, not a breakpoint: this panel is a narrow column beside the results on a
          wide screen and full width when they stack, so what it can fit is its own width rather than
          the viewport's.

          `text-center` here rather than on `Fact`, which the detail panel next to the results also
          uses and where the cells sit in a row that reads better left-aligned. */}
      <dl className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-2.5 text-center @md:grid-cols-4">
        <Fact label="parameters">
          {(facts.weights.paramCount / 1e9).toFixed(2)}B
        </Fact>
        <Fact label="on disk">{gib(facts.weights.onDiskBytes)}</Fact>
        <Fact label="dtype (weights)" title={weights.title}>
          {weights.text}
        </Fact>
        {/* What `dtype="auto"` and vLLM's `--dtype` resolve to, which on a quantized repo is a
            different answer from the row above it: DeepSeek V4 stores fp8 and fp4 weights and
            declares bfloat16. bfloat16 is the residual stream, the norms and every unquantized
            parameter rather than every multiply -- a checkpoint with `activation_scheme: dynamic`
            casts the inputs of its quantized GEMMs back down. */}
        <Fact label="dtype (compute)">{facts.weights.storedDtype || "?"}</Fact>
        <Fact label="layers">{facts.nLayers || "?"}</Fact>
        <Fact label="d_model">{facts.dModel || "?"}</Fact>
        <Fact label="kv heads">
          {facts.nKvHeads || "?"} x {facts.headDim || "?"}
        </Fact>
        <Fact label="vocab">
          {facts.vocabSize ? num(facts.vocabSize) : "?"}
        </Fact>
        <Fact label="advertised ctx">
          {facts.maxPositionEmbeddings
            ? num(facts.maxPositionEmbeddings)
            : "unknown"}
        </Fact>
        <Fact label="kv figures">{precision}</Fact>
      </dl>

      {/* The caveats `lib/hub.ts` attached. Kept verbatim: each one names the direction its
          uncertainty errs in, which is the part that decides whether to trust the row. */}
      {facts.notes.length > 0 && (
        <div className="mt-2 border-t border-slate-200 pt-2">
          <button
            type="button"
            onClick={() => setShowNotes((was) => !was)}
            aria-expanded={showNotes}
            className="flex cursor-pointer items-center gap-x-1 text-[10px] text-slate-500 transition-colors hover:text-slate-800"
          >
            <ChevronRight
              className={`h-3 w-3 shrink-0 transition-transform ${
                showNotes ? "rotate-90" : ""
              }`}
            />
            More Details
          </button>

          {showNotes && (
            <ul className="mt-1.5 flex flex-col gap-y-1">
              {facts.notes.map((note) => (
                <li
                  key={note}
                  className="text-[10px] leading-relaxed text-slate-500"
                >
                  {note}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function Fact({
  label,
  title,
  children,
}: {
  label: string;
  /** Hover text, for a value the cell is too narrow to spell out. */
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <dt className="text-[9px] text-slate-400">{label}</dt>
      <dd
        title={title}
        className="truncate font-mono text-[11px] text-slate-700"
      >
        {children}
      </dd>
    </div>
  );
}

function Controls({
  facts,
  backend,
  onBackend,
  dtype,
  onDtype,
  staticPoints,
  onStaticPoints,
  context,
  onContext,
  reserveGib,
  onReserve,
  lens,
  onLens,
}: {
  facts: ModelMemoryFacts;
  backend: Backend;
  onBackend: (value: Backend) => void;
  dtype: string;
  onDtype: (value: string) => void;
  staticPoints: string[];
  onStaticPoints: (value: string[]) => void;
  context: number;
  onContext: (value: number) => void;
  reserveGib: number;
  onReserve: (value: number) => void;
  lens: boolean;
  onLens: (value: boolean) => void;
}) {
  const blind = facts.derivedDims.includes("max_position_embeddings");
  return (
    <div className="flex flex-col gap-y-5">
      {/* One card per row, each carrying its own note: the choice is between
          three whole engines rather than three settings, and a single note under
          the group only ever described the one already picked. */}
      <Field label="backend">
        <div className="grid grid-cols-1 gap-2">
          {OFFERED_BACKENDS.map((option) => {
            const active = backend === option;
            return (
              <button
                key={option}
                type="button"
                onClick={() => onBackend(option)}
                className={`flex cursor-pointer items-center gap-x-3 rounded-lg border px-3 py-4 text-left transition-colors ${
                  active
                    ? "border-sky-700 bg-sky-200"
                    : "border-slate-300 bg-white hover:bg-slate-50"
                }`}
              >
                <span className="min-w-0 flex-1">
                  <span
                    className={`block font-mono text-[11px] font-bold ${
                      active ? "text-sky-700" : "text-slate-700"
                    }`}
                  >
                    {option}
                  </span>
                  <span
                    className={`mt-0.5 block text-[10px] leading-relaxed ${
                      active ? "text-sky-800" : "text-slate-500"
                    }`}
                  >
                    {BACKEND_NOTE[option]}
                  </span>
                </span>
                <BackendBars backend={option} active={active} />
              </button>
            );
          })}
        </div>
      </Field>

      {/* One per row now that this lives in a column of its own. Side by side,
          `max_model_len` and its hint had about half the width the label needed
          and wrapped to two lines while `dtype` beside it sat on one. */}
      <div className="grid grid-cols-1 gap-5">
        {/* Only on the static backend, because it is the only one where a tap is
            a preallocated buffer. The hooked backends build their tensors as the
            forward runs, so which points are read costs nothing up front. */}
        {backend === "vllm-static" && (
          <Field label="static points" hint="read + write, every layer">
            <MultiSelect
              options={offeredStaticPoints(facts).map((point) => ({
                value: point,
                label: point,
                title: `${num(staticPointElements(point, facts))} elements per token per layer`,
              }))}
              value={resolvedStaticPoints(
                workload({ backend, staticPoints }),
                facts,
              )}
              onChange={onStaticPoints}
            />
          </Field>
        )}

        <Field label="dtype">
          <Segmented
            options={DTYPES.map((option) => ({
              value: option,
              label: option,
            }))}
            value={dtype}
            onChange={onDtype}
          />
        </Field>

        <Field
          label={isVllm(backend) ? "max_model_len" : "prompt tokens"}
          hint={blind ? "required" : ""}
        >
          {/* No auto to offer on a model whose own limit could not be read: the
              length is the thing that has to be chosen, which is what `blind`
              means, so the field is the whole control. */}
          {!blind && (
            <Segmented
              options={CONTEXT_MODES}
              value={context ? "manual" : "auto"}
              onChange={(mode) =>
                onContext(
                  mode === "auto"
                    ? 0
                    : Math.min(facts.maxPositionEmbeddings || 8192, 8192),
                )
              }
            />
          )}
          {(blind || context > 0) && (
            <div className={blind ? "" : "mt-1.5"}>
              <NumberField
                value={context}
                onChange={onContext}
                unit="tokens"
                step={1024}
                placeholder="8192"
              />
            </div>
          )}
        </Field>

        <Field label="reserve VRAM" hint="for custom tensors, SAEs, etc">
          <NumberField
            value={reserveGib}
            onChange={onReserve}
            unit="GiB"
            step={1}
            placeholder="0"
          />
        </Field>

        <Field label="jacobian lens" hint="n_layers x d_model^2">
          <Segmented
            options={LENS_MODES}
            value={lens ? "on" : "off"}
            onChange={(mode) => onLens(mode === "on")}
          />
        </Field>
      </div>
    </div>
  );
}

function BackendBars({
  backend,
  active,
}: {
  backend: Backend;
  active: boolean;
}) {
  const bars = BACKEND_BARS[backend];
  const pairs = [
    { label: "Speed", rung: bars.speed, risk: undefined },
    { label: "VRAM", rung: bars.vram, risk: bars.risk },
  ];
  return (
    <div className="flex shrink-0 items-end gap-x-2">
      {pairs.map(({ label, rung, risk }) => (
        <div key={label} className="flex flex-col items-center gap-y-1">
          {/* Rounded and clipped on the track rather than on the fill, so a bar
              in two colours keeps one outline instead of two stacked pills. */}
          <div
            className={`flex h-9 w-3.5 flex-col justify-end overflow-hidden rounded-sm ${
              active ? "bg-white/70" : "bg-slate-100"
            }`}
          >
            {risk && <RiskCap risk={risk} />}
            <div
              className={`w-full shrink-0 bg-sky-600 ${BAR_HEIGHTS[rung - 1]}`}
            />
          </div>
          <span
            className={`text-[8px] leading-none ${
              active ? "text-sky-700" : "text-slate-400"
            }`}
          >
            {label}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * The amber cap on eager's VRAM bar, and why it is there.
 *
 * Hatched rather than solid, borrowing the over-capacity marking from
 * `VramBar`: the sky below it is a measurement and this is a ceiling the model
 * in front of you may never reach, and a second solid block would read as more
 * of the same quantity rather than as a different kind of claim.
 *
 * The popover is `ModeInfo`'s, for `ModeInfo`'s reasons — a hand-driven
 * `Popover` rather than a `HoverCard`, which never opens on touch, and a `span`
 * with a button's role because the engine option it sits in is already a button
 * and one cannot nest in another. It stops its own click too: reading why the
 * bar goes higher should not select the engine.
 */
function RiskCap({ risk }: { risk: Risk }) {
  const [open, setOpen] = useState(false);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Anchor asChild>
        <span
          role="button"
          tabIndex={0}
          aria-label={risk.title}
          onPointerEnter={(event) => {
            if (event.pointerType === "mouse") setOpen(true);
          }}
          onPointerLeave={(event) => {
            if (event.pointerType === "mouse") setOpen(false);
          }}
          onPointerDown={(event) => {
            if (event.pointerType === "mouse") return;
            event.preventDefault();
            event.stopPropagation();
            setOpen((was) => !was);
          }}
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            event.stopPropagation();
            setOpen((was) => !was);
          }}
          className={`w-full shrink-0 cursor-help bg-[repeating-linear-gradient(45deg,rgb(251_191_36)_0_2px,transparent_2px_5px)] focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 ${risk.height}`}
        />
      </Popover.Anchor>
      <Popover.Portal>
        <Popover.Content
          side="top"
          align="end"
          sideOffset={8}
          collisionPadding={12}
          onOpenAutoFocus={(event) => event.preventDefault()}
          className="animate-in fade-in-0 zoom-in-95 z-[60] w-[min(320px,calc(100vw-24px))] rounded-md border border-slate-200 bg-white p-3 shadow-lg"
        >
          <p className="text-xs font-semibold text-slate-700">{risk.title}</p>
          <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
            {risk.body}
          </p>
          <Popover.Arrow className="fill-white" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <div className="mb-0.5 flex items-baseline justify-between gap-x-2">
        <span className="font-mono text-[9px] text-slate-500">{label}</span>
        {hint && (
          <span className="text-[9px] text-slate-400 italic">{hint}</span>
        )}
      </div>
      {children}
    </div>
  );
}

/**
 * A row of exclusive choices, in the engine selector's colours because it is the
 * same kind of decision one size down.
 *
 * Drawn as one control rather than as buttons that happen to be adjacent: the
 * borders collapse onto each other with `-space-x-px` and only the two ends are
 * rounded, so four options read as one bar with a position in it. An active
 * segment is lifted a layer, or its sky border would be half-covered by the
 * slate one of whichever neighbour React drew last.
 */
/**
 * A number and its unit, sharing one box. The unit is inside the field rather
 * than in the label because it belongs to the value: "8192" and "8192 tokens"
 * are the same claim, and a reader checking what they typed should not have to
 * look above the box to find out what it is counting.
 */
function NumberField({
  value,
  onChange,
  unit,
  step,
  placeholder,
}: {
  value: number;
  onChange: (value: number) => void;
  unit: string;
  step: number;
  placeholder: string;
}) {
  return (
    <div className="flex items-center rounded-md border border-slate-300 bg-white focus-within:border-sky-600">
      <input
        type="number"
        min={0}
        step={step}
        value={value || ""}
        placeholder={placeholder}
        onChange={(event) => onChange(Number(event.target.value) || 0)}
        className={`min-w-0 flex-1 bg-transparent px-2 py-1.5 font-mono text-[11px] text-slate-700 focus:outline-none ${NO_STEPPER}`}
      />
      <span className="shrink-0 pr-2 pl-1 font-mono text-[10px] text-slate-400">
        {unit}
      </span>
    </div>
  );
}

function Segmented({
  options,
  value,
  onChange,
}: {
  options: readonly { value: string; label: string }[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <div className="flex -space-x-px">
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            className={`relative flex-1 cursor-pointer border px-1.5 py-1.5 font-mono text-[10px] whitespace-nowrap transition-colors first:rounded-l-md last:rounded-r-md ${
              active
                ? "z-10 border-sky-700 bg-sky-200 text-sky-700"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/**
 * `Segmented` for a set rather than a choice: same colours, but wrapped chips
 * instead of one bar, because eight point names do not fit a column this narrow
 * and a bar that wraps loses the one thing it was drawn for — reading as a
 * position in a range.
 *
 * The last selected chip is held down. An empty static set is not "tap nothing",
 * it is the generation engine wearing the static engine's name, and the engine
 * refuses it; offering it here would show a VRAM figure no `vllm-static` pod can
 * actually run at.
 */
function MultiSelect({
  options,
  value,
  onChange,
}: {
  options: readonly { value: string; label: string; title?: string }[];
  value: string[];
  onChange: (value: string[]) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((option) => {
        const active = value.includes(option.value);
        const held = active && value.length === 1;
        return (
          <button
            key={option.value}
            type="button"
            title={held ? "a static set cannot be empty" : option.title}
            onClick={() =>
              // Rebuilt from the offered order rather than appended, so the set
              // reads down the forward pass wherever it is printed.
              onChange(
                options
                  .filter((other) =>
                    other.value === option.value
                      ? !active
                      : value.includes(other.value),
                  )
                  .map((other) => other.value),
              )
            }
            className={`rounded-md border px-1.5 py-1.5 font-mono text-[10px] whitespace-nowrap transition-colors ${
              held ? "cursor-default" : "cursor-pointer"
            } ${
              active
                ? "border-sky-700 bg-sky-200 text-sky-700"
                : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

function Results({
  facts,
  backend,
  dtype,
  tiers,
  selected,
  onSelect,
}: {
  facts: ModelMemoryFacts;
  backend: Backend;
  dtype: string;
  tiers: Tier[];
  selected: string;
  onSelect: (name: string) => void;
}) {
  if (!facts.trunkDimsKnown) {
    // "Nothing fits" and "nothing could be computed" look identical in an empty list and mean
    // opposite things: the first sends someone to buy a bigger card, and the second is a missing
    // token. Only the weights above are real here.
    return (
      <Empty>
        This model cannot be sized: its config gave no layer or head dimensions,
        so everything that depends on the KV cache is omitted rather than
        guessed. A gated repo returns file sizes without a token but not its
        config — add one above.
      </Empty>
    );
  }
  if (!tiers.length) {
    return (
      <Empty>
        None of the {GPUS.length} cards in the catalog fits this, even sharded
        across 8 of them.
        {facts.weights.quantMethod && dtype !== "auto"
          ? ` Try dtype "auto": ${facts.weights.quantMethod} served natively is much smaller than the dequantized figure this is pricing.`
          : " A quantized checkpoint of this model would, or a shorter context."}
      </Empty>
    );
  }

  return (
    <div className="overflow-hidden rounded-md border border-slate-200 bg-white">
      <div
        className={`${COLUMNS} border-b border-slate-200 px-3 py-1.5 text-[9px] text-slate-400`}
      >
        <span>GPU Config</span>
        <span className="text-right">utilization</span>
        <span className="text-right">
          {isVllm(backend) ? "context" : "prompt"}
        </span>
        <span className="text-right">
          {isVllm(backend) ? "concurrent" : "peak"}
        </span>
      </div>

      <ul className="divide-y divide-slate-100">
        {tiers.map((tier) => {
          const est = tier.result.estimate;
          const evidence = tierEvidence(facts, tier);
          const active = tierGpus(tier).some((gpu) => gpu.name === selected);
          return (
            <li key={`${tier.gib}/${tier.count}`}>
              <button
                type="button"
                onClick={() => onSelect(tier.result.gpu.name)}
                // Every card on the rung with what it really holds. The gap
                // between the board size and the usable figure is the whole
                // reason two 24GB cards can land on different rows.
                title={tierGpus(tier)
                  .map(
                    (gpu) =>
                      `${gpu.name} — ${totalGib(gpu).toFixed(1)} GiB usable`,
                  )
                  .join("\n")}
                className={`${COLUMNS} w-full cursor-pointer px-3 py-2.5 text-left transition-colors ${
                  active ? "bg-sky-100" : "hover:bg-slate-50"
                }`}
              >
                <span className="flex min-w-0 items-center gap-x-1.5">
                  {/* Picking a tier swaps what the panel below is about rather
                      than opening anything, so the mark is the one a radio uses
                      and not a disclosure arrow. */}
                  <Circle
                    className={`h-3 w-3 shrink-0 ${
                      active ? "fill-current text-sky-700" : "text-slate-300"
                    }`}
                  />
                  <span
                    className={`shrink-0 text-[13px] font-medium ${
                      active ? "text-sky-800" : "text-slate-700"
                    }`}
                  >
                    {tier.count}x {tier.gib}GB
                  </span>
                  <span
                    className={`truncate text-[10px] ${
                      active ? "text-sky-600" : "text-slate-400"
                    }`}
                  >
                    {examples(tier)}
                  </span>
                  {/* Only where hardware has something to say. "estimated" is
                      every other row, so a badge for it was a column of noise —
                      but "known to fail" has to stay visible wherever it lands,
                      since those rows fit on paper and look like any other. */}
                  {evidence.kind !== "estimated" && (
                    <Badge evidence={evidence.kind}>{evidence.label}</Badge>
                  )}
                </span>
                <span
                  className={`text-right font-mono text-[10px] ${
                    active ? "text-sky-800" : "text-slate-600"
                  }`}
                >
                  {isVllm(backend)
                    ? `${Math.round(est.spec.gpuMemoryUtilization * 100)}%`
                    : "-"}
                </span>
                <span
                  className={`text-right font-mono text-[10px] ${
                    active ? "text-sky-800" : "text-slate-600"
                  }`}
                >
                  {compactTokens(
                    isVllm(backend) ? est.spec.maxModelLen : est.spec.seqLen,
                  )}
                </span>
                <span
                  className={`text-right font-mono text-[10px] ${
                    active ? "text-sky-800" : "text-slate-600"
                  }`}
                >
                  {isVllm(backend)
                    ? `${concurrentSequences(est)} seq`
                    : gib(totalBytes(est))}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * The tier table's tracks, shared by the header and every row.
 *
 * Fixed widths and one constant, because the header and the rows are separate
 * grids — a `<div>` and a `<button>` per row, since a row has to be clickable —
 * and `auto` tracks are measured per grid. Each row sized its own columns to its
 * own contents, so "90%" and "16k" and "3 seq" each pulled the numbers to a
 * different offset and none of them sat under the label naming them.
 */
const COLUMNS =
  "grid grid-cols-[minmax(0,1fr)_4rem_3.5rem_5rem] items-center gap-x-3";

/**
 * One card in the results column. White rather than inheriting, because this
 * column is the only one on the slate page background and three bordered cards
 * on slate read as a single grey field with lines through it.
 */
const PANEL =
  "flex flex-col gap-y-3 rounded-md border border-slate-200 bg-white px-3 py-3";

/** 16,384 as `16k`. The exact figure is in the panel below; the column is for scanning. */
function compactTokens(tokens: number): string {
  if (tokens < 1000) return String(tokens);
  if (tokens < 1_000_000) return `${Math.round(tokens / 1000)}k`;
  return `${(tokens / 1_000_000).toFixed(1)}M`;
}

/**
 * Two names, and no count of the rest. Two is what the row has space for without
 * pushing the numbers off the right edge; the full list, with each card's usable
 * capacity, is on the row's `title`. These are an illustration of the rung rather
 * than an inventory of it, and a trailing "+6" invited the row to be read as the
 * second thing.
 *
 * De-duplicated after shortening, which is what makes the shorthand safe: an
 * A100 80GB PCIe and an A100-SXM4-80GB are two catalog entries and one name, and
 * a row that said "A100 80GB, A100 80GB" would read as a bug.
 */
function examples(tier: Tier): string {
  const names: string[] = [];
  for (const gpu of tierGpus(tier)) {
    const short = shortGpuName(gpu.name);
    if (!names.includes(short)) names.push(short);
    if (names.length === 2) break;
  }
  return names.join(", ");
}

function Badge({
  evidence,
  children,
}: {
  evidence: "verified" | "fails" | "estimated";
  children: ReactNode;
}) {
  const tone =
    evidence === "verified"
      ? "bg-emerald-100 text-emerald-700"
      : evidence === "fails"
        ? "bg-red-100 text-red-700"
        : "bg-slate-100 text-slate-500";
  return (
    <span
      className={`rounded-sm px-1.5 py-px text-[9px] font-medium whitespace-nowrap ${tone}`}
    >
      {children}
    </span>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-md bg-amber-50 px-3 py-2.5 text-[11px] leading-relaxed text-amber-800">
      {children}
    </p>
  );
}

/**
 * Why nothing fits, priced on the biggest card in the catalog.
 *
 * Without this the advice `lib/size.ts` writes is unreachable: it exists only for a configuration that
 * does not fit, and the list above only ever holds configurations that do — so the one moment a
 * reader needs to be told which knob to turn is the one moment the tool had nothing to say. The
 * largest card is the right subject because if the answer is still no there, no smaller card changes
 * it, and the shortfall on the biggest is the smallest shortfall there is.
 */
function NoFit({
  facts,
  backend,
  dtype,
  context,
  staticPoints,
  res,
}: {
  facts: ModelMemoryFacts;
  backend: Backend;
  dtype: string;
  context: number;
  staticPoints: string[];
  res: Reservations;
}) {
  const biggest = [...GPUS].sort((a, b) => b.totalBytes - a.totalBytes)[0];
  const est = estimate(
    facts,
    biggest,
    workload({
      backend,
      dtype,
      maxModelLen: context,
      staticPoints,
      // vLLM's own default rather than a derived ceiling: a fit search would have lowered this to buy
      // margin, and the question here is what the configuration costs, not how to squeeze it.
      gpuMemoryUtilization: isVllm(backend) ? CALIBRATION.max_util : 0,
    }),
    res,
  );
  const vllm = isVllm(backend);

  return (
    <div className={PANEL}>
      <p className="text-[11px] leading-relaxed text-slate-600">
        Priced on the largest card in the catalog, a{" "}
        <span className="font-medium">{biggest.name}</span> at{" "}
        {totalGib(biggest).toFixed(1)} GiB, so the shortfall below is the
        smallest one available.
      </p>

      <VramBar
        label={`${biggest.name}, whole card`}
        hint={
          vllm
            ? `pool is card x ${est.spec.gpuMemoryUtilization}`
            : "one process, no pool"
        }
        capacityBytes={biggest.totalBytes}
        poolBytes={est.poolBytes}
        terms={est.terms}
      />

      {est.advice.length > 0 && (
        <ul className="flex flex-col gap-y-1 rounded-sm bg-slate-50 px-2.5 py-2">
          {est.advice.map((line) => (
            <li
              key={line}
              className="text-[10px] leading-relaxed text-slate-600"
            >
              {line}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Detail({
  facts,
  result,
  backend,
  res,
}: {
  facts: ModelMemoryFacts;
  result: FitResult;
  backend: Backend;
  res: Reservations;
}) {
  const { gpu, count, estimate: fitted } = result;
  // Repriced from the fitted spec rather than reused, so the detail is the estimate for exactly the
  // spec the snippet below prints -- the fit search returns the first rung that worked, and reading
  // its terms back is how the two could drift apart.
  const est = estimate(facts, gpu, fitted.spec, res);
  const code = snippet(facts, est.spec, gpu, count);
  const vllm = isVllm(backend);

  return (
    <>
      <div className={PANEL}>
        <VramBar
          label={`${count}x ${gpu.name}`}
          hint={
            vllm
              ? `${est.spec.gpuMemoryUtilization * 100}% of VRAM reserved for vLLM`
              : "one process, no pool: the activation peak sits beside the weights"
          }
          capacityBytes={gpu.totalBytes}
          poolBytes={est.poolBytes}
          terms={est.terms}
        />

        {vllm && (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-200 pt-2.5 sm:grid-cols-4">
            <Fact label="kv cache">{num(est.kvCapacityTokens)} tokens</Fact>
            <Fact label="concurrent">
              {concurrentSequences(est)} x {num(est.spec.maxModelLen)}
            </Fact>
            <Fact label="prefill batch">
              {num(est.spec.maxNumBatchedTokens)}
            </Fact>
            <Fact label="pool free">{gib(est.poolHeadroomBytes)}</Fact>
          </dl>
        )}

        {est.warnings.length > 0 && (
          <ul className="flex flex-col gap-y-1.5">
            {est.warnings.map((warning) => (
              <li
                key={warning}
                className="flex items-start gap-x-2 text-[10px] leading-relaxed text-amber-800"
              >
                <AlertTriangle className="mt-px h-3 w-3 shrink-0 text-amber-500" />
                {warning}
              </li>
            ))}
          </ul>
        )}

        {est.advice.length > 0 && (
          <ul className="flex flex-col gap-y-1 rounded-sm bg-slate-50 px-2.5 py-2">
            {est.advice.map((line) => (
              <li
                key={line}
                className="text-[10px] leading-relaxed text-slate-600"
              >
                {line}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className={PANEL}>
        {/* A flex row for one button, so it is sized by its label rather than
            stretched by the column it sits in. */}
        <div className="flex">
          <CopyButton
            text={code}
            label="Copy Code"
            iconClassName="h-3 w-3"
            className="gap-x-1.5 rounded-full bg-sky-700 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-sky-800 hover:text-white"
          />
        </div>
        <pre className="thin-scrollbar overflow-x-auto rounded-sm bg-slate-50 px-3 py-2 font-mono text-[10px] leading-relaxed text-slate-700">
          {code}
        </pre>
      </div>
    </>
  );
}
