"use client";

/**
 * The README's numbers, with the conditions they were taken under.
 *
 * Shared by the two places that make the speed claim: the **Fast** hover card
 * under the wordmark, and the tour's first slide. One component rather than a
 * card and a slide that both draw a chart from `data/benchmarks.ts`, because the
 * second copy is the one that keeps the old figures after a sweep republishes.
 *
 * The chart draws the concurrent regime. Full figures live in the linked report.
 * The hover card names the machine and links that report; the tour's compact
 * slide does not.
 */

import {
  BENCHMARKS,
  BENCHMARK_CONCURRENCY,
  BENCHMARK_GPU,
  type Benchmark,
  type Comparison,
  type Rate,
  tokensPerSecond,
} from "@/data/benchmarks";
import { BENCHMARK_VIDEO } from "@/lib/assets";
import { cn } from "@/lib/utils";

export function Throughput({
  repoUrl,
  compact = false,
}: {
  repoUrl: string;
  /**
   * The tour's first slide: the two featured models only, without the
   * machine line. The hover card keeps every row from `sm` up, and the same
   * two on a phone.
   */
  compact?: boolean;
}) {
  const rows = compact
    ? BENCHMARKS.filter((row) => FEATURED_MODELS.has(row.model))
    : BENCHMARKS;

  return (
    <>
      <video
        src={BENCHMARK_VIDEO}
        autoPlay
        muted
        loop
        playsInline
        aria-label="Benchmark: interp-engine capturing and generating at serving throughput"
        className="mb-3 w-full rounded-md"
      />
      <div className="flex items-baseline justify-start gap-x-2">
        <span className="text-sm font-semibold text-slate-700">
          Tokens Per Second
        </span>
        <span className="shrink-0 text-[10px] font-medium tracking-wide text-slate-400 max-sm:hidden">
          total throughput while capturing + generating
        </span>
      </div>

      {/* The key and what is being keyed, on one line: both are captions for the
          chart below, and stacking them spent a row saying less than this does. */}
      <div className="mt-1.5 flex items-baseline justify-between gap-x-3 text-[10px]">
        <div className="flex items-center gap-x-3 text-slate-500">
          {SERIES.map(({ field, label, bar }) => (
            <span key={field} className="flex items-center gap-x-1">
              <span className={cn("h-2 w-2 rounded-sm", bar)} aria-hidden />
              {label}
            </span>
          ))}
        </div>

        <div className="flex shrink-0 items-baseline gap-x-1.5">
          <span className="font-mono text-slate-500">@ {CHARTED.label}</span>
          <span className="text-slate-400">{CHARTED.caption}</span>
        </div>
      </div>

      {/* The caption's card is 560px; the tour is wider and drops every model
          but two. Neither needs a min-width floor now that the figure table is
          gone — the chart fills whatever width it is given. */}
      <div className="mt-2">
        <ThroughputChart
          regime={CHARTED.regime}
          rows={rows}
          mark={!compact}
          height={compact ? 100 : CHART_HEIGHT}
        />
      </div>

      {!compact && (
        <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
          1 x {BENCHMARK_GPU}.{" See "}
          <a
            href={`${repoUrl}/blob/main/benchmarks/results-latest.md`}
            target="_blank"
            rel="noreferrer noopener"
            // Unbroken: a filename split across two lines at its own hyphen reads
            // as two words, and the underline makes the break look deliberate.
            className="font-mono whitespace-nowrap text-sky-700 underline underline-offset-2 hover:text-sky-800"
          >
            results-latest.md
          </a>{" "}
          for detailed results.
        </p>
      )}
    </>
  );
}

/**
 * The two models a phone has room for. The tour is these at every width; the
 * hover card drops the rest below `sm`, so the Fast card on a phone is the
 * same pair rather than five groups squeezed into `100vw`.
 */
const FEATURED_MODELS = new Set(["qwen3.8-27b", "deepseek-v4-flash-0731"]);

/**
 * The rows that did not run under the conditions the rest share.
 *
 * A row is here because the sweep declared a per-model override for it, not
 * because anyone listed it: `benchmarks/cells.py` derives `differs`, so the next
 * checkpoint needing an argument of its own arrives marked rather than as one
 * more line of what looks like a like-for-like table.
 */
const FOOTNOTED = BENCHMARKS.filter((row) => row.differs?.length);

/**
 * The mark on such a row, wherever that row's name is printed.
 *
 * An asterisk and not a footnote number, because there is no numbered note to
 * reach: what the mark says is "this row is not quite comparable, and the linked
 * report says how". A `1` promises a note 1 somewhere on the card.
 */
function Marker({ row }: { row: Benchmark }) {
  if (!FOOTNOTED.includes(row)) return null;
  return <sup className="text-slate-400">*</sup>;
}

/**
 * The two rows a model is drawn as. `regime` names the field to read from a
 * `Rate`, so nothing can take its figure from one regime and its multiplier from
 * the other.
 */
const REGIMES = [
  { regime: "single", label: "1x", caption: "one stream at a time" },
  {
    regime: "concurrent",
    label: `${BENCHMARK_CONCURRENCY}x`,
    caption: `${BENCHMARK_CONCURRENCY} requests in flight`,
  },
] as const;

type Regime = (typeof REGIMES)[number]["regime"];

/**
 * The one regime the chart draws. Both are still in the table below it, which is
 * where a reader who wants the single-stream figure goes.
 *
 * Concurrent rather than single because it is the regime the engine is used in
 * and the one the two backends differ most in — eager serializes eight requests
 * where vLLM batches them. Charting both put the same five models on screen
 * twice, and the pair of charts read as one comparison with a fold in it: they
 * are different workloads on axes that are not shared, which nothing about two
 * stacked charts of identical shape conveys.
 */
const CHARTED = REGIMES[1];

/** The multiplier a rate carries for one regime, or null on the baseline column. */
function multiplierOf(rate: Rate | Comparison, regime: Regime): string | null {
  if (!("singleVsEager" in rate)) return null;
  return regime === "single" ? rate.singleVsEager : rate.concurrentVsEager;
}

/**
 * Whether a multiplier is a win, which not all of them are: hooked vLLM decodes
 * one stream of `deepseek-v4-flash-0731` at `0.9x` eager. Green on that figure
 * would tell the reader the opposite of what the figure says, and the column is
 * scanned rather than read, so the colour lands before the number does.
 */
function isGain(multiplier: string): boolean {
  return Number.parseFloat(multiplier) >= 1;
}

/**
 * A bar per backend, in the table's column order and coloured the way its
 * heading is, so the two halves of the card are read with one key.
 */
const SERIES = [
  {
    field: "eager",
    label: "eager",
    bar: "bg-slate-400",
    ink: "text-slate-500",
  },
  {
    field: "vllm",
    label: "IE-vLLM",
    bar: "bg-emerald-500",
    ink: "text-emerald-600",
  },
  {
    field: "static",
    label: "IE-vLLM-static",
    bar: "bg-emerald-700",
    ink: "text-emerald-700",
  },
] as const;

/**
 * Bar area, and the tallest bar in it: the difference is headroom for a
 * multiplier. Derived rather than a second number, because that headroom holds
 * two lines of 10px text and so does not scale with the chart — doubling both
 * would buy nothing but empty pixels above the tallest bar.
 */
const CHART_HEIGHT = 112;
/** Under this a bar cannot hold its own label, so the figure goes above it. */
const LABEL_FITS = 15;

/**
 * One regime's bars, three to a model, each model on an axis of its own.
 *
 * A group's tallest bar is full height, so what a group shows is the *shape* of
 * that model's win — how much of static's throughput hooked vLLM reaches, and how
 * little of it eager does. Heights are therefore comparable within a group and
 * not across groups, which is the trade this scaling makes: one axis over the
 * whole chart flattened every group but the fastest model's, since 87 tok/s
 * beside a 1,536 peak is 4% of the height whatever the win inside that group was.
 * The figure on each bar and the multiplier above it are what carry the
 * cross-model comparison, and the table below prints both in full.
 *
 * The floor at 2px is why a group cannot be read as an exact ratio: an eager bar
 * is a pixel or under at these multipliers and would otherwise vanish.
 *
 * `aria-hidden`, because the table below is the same numbers in a form a screen
 * reader can read in order; a grid of unlabelled `div`s is not.
 */
function ThroughputChart({
  regime,
  rows,
  mark,
  height,
}: {
  regime: Regime;
  rows: Benchmark[];
  /** Asterisk on a row whose conditions differ. Off in the tour, which has no
   *  report link for the mark to point at. */
  mark: boolean;
  height: number;
}) {
  const barMax = height - 22;
  return (
    <div aria-hidden>
      <div className="flex items-end gap-x-2" style={{ height }}>
        {rows.map((row) => {
          const peak = Math.max(
            ...SERIES.map(({ field }) => {
              const rate = row[field];
              return rate ? tokensPerSecond(rate[regime]) : 0;
            }),
          );
          return (
            <div
              key={row.model}
              className={cn(
                "flex h-full flex-1 items-end gap-x-[3px]",
                !FEATURED_MODELS.has(row.model) && "max-sm:hidden",
              )}
            >
              {SERIES.map((series) => (
                <Bar
                  key={series.field}
                  row={row}
                  series={series}
                  regime={regime}
                  peak={peak}
                  barMax={barMax}
                />
              ))}
            </div>
          );
        })}
      </div>

      <div className="mt-1 flex gap-x-2">
        {rows.map((row) => (
          <div
            key={row.model}
            // Wrapping rather than truncating: a checkpoint name is mostly
            // hyphen-separated version, so `deepseek-v4-flash-0731` truncates to
            // a prefix shared with nothing but breaks onto two lines cleanly.
            className={cn(
              "flex-1 text-center font-mono text-[10px] leading-tight text-slate-500",
              !FEATURED_MODELS.has(row.model) && "max-sm:hidden",
            )}
          >
            {row.model}
            {mark && <Marker row={row} />}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * One bar, its tok/s at the top of it and its multiplier above it. A backend the
 * sweep did not run keeps its slot in the group and draws nothing, so the bars
 * under a model name stay in the same order across every group.
 */
function Bar({
  row,
  series,
  regime,
  peak,
  barMax,
}: {
  row: Benchmark;
  series: (typeof SERIES)[number];
  regime: Regime;
  /** This model's fastest backend, so a group's tallest bar is `barMax`. */
  peak: number;
  barMax: number;
}) {
  const rate = row[series.field];
  if (rate === null) return <div className="flex-1" />;

  const figure = rate[regime];
  const height = Math.max(
    2,
    Math.round((tokensPerSecond(figure) / peak) * barMax),
  );
  const vsEager = multiplierOf(rate, regime);
  // A 2px bar with white text in it reads as a smudge, so the short ones print
  // their figure above the bar instead, under the multiplier.
  const inside = height >= LABEL_FITS;

  return (
    <div className="flex flex-1 flex-col items-center justify-end">
      {vsEager && (
        <span
          className={cn(
            "text-[10px] leading-none",
            isGain(vsEager) ? series.ink : "text-slate-400",
          )}
        >
          {vsEager}
        </span>
      )}
      {!inside && (
        <span className="mt-0.5 font-mono text-[10px] leading-none text-slate-500">
          {figure}
        </span>
      )}
      <div
        className={cn("mt-0.5 w-full rounded-t", series.bar)}
        style={{ height }}
      >
        {inside && (
          <span className="block pt-px text-center font-mono text-[10px] leading-none whitespace-nowrap text-white">
            {figure}
          </span>
        )}
      </div>
    </div>
  );
}
