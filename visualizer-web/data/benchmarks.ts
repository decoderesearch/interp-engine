/**
 * The throughput card's types, and the reasoning behind the shape it draws. The
 * numbers themselves are in `./benchmarks.generated`, written by
 * `python -m benchmarks.publish` from `benchmarks/results/*.json` — the same
 * cells `benchmarks/results-latest.md` and the repo README's tables render, so
 * the three cannot disagree about a figure. This module re-exports them, so a
 * component imports `@/data/benchmarks` and never has to know which half of the
 * pair a name came from.
 *
 * Unlike `points.ts`, this file is *not* a hand transcription: it was one, and
 * it drifted. Change a number by re-running the sweep, never by editing either
 * file.
 *
 * The machine travels with the numbers, because these are one box's figures and
 * not a specification. The rest of what a reader needs to hold their own run
 * against them — the dtype, the two lengths, and the settings a given row needed
 * of its own — is a link to `benchmarks/results-latest.md` rather than more lines
 * under the table: that file is generated from the same cells, so it cannot fall
 * behind the card, and it is the only copy that keeps full precision.
 *
 * Three backends against two concurrency regimes is six figures a model, which
 * is more than four columns of a 560px card can hold. So a column is a backend
 * and each model is drawn as two rows, `@ 1x` and `@ 8x`, one figure a cell.
 * That shape is also what makes the multipliers checkable — each is against
 * eager *in its own regime*, and the eager figure to divide by is in the same
 * row. The layout before this one printed one eager number and compared the
 * concurrent column to a baseline that was not shown, which reads as a much
 * bigger win than it was.
 *
 * Most rows differ from each other only in the model. A row that ran under any
 * other condition carries `differs`, and the card marks that row and sends the
 * reader to the report for the detail — `deepseek-v4-flash-0731` reserves more of
 * the GPU than the rest and serves an FP8 KV cache, which is a different
 * comparison rather than a detail of the setup. The card used to drop such a row
 * instead, which cost it the largest static win in the sweep; being unexplained
 * and being unpublishable are not the same problem. What the mark has to do is
 * stop the row being read as one more line of a like-for-like table, which one
 * character does as well as a paragraph.
 *
 * Every tok/s figure is a whole token at 10 and above, as the README's tables
 * are, and `benchmarks/results-latest.md` remains the full-precision copy. A
 * tenth beside a four-digit figure in the next column claims a resolution the
 * reader has no way to use. The multipliers are the *unrounded* ratios, so
 * dividing two printed figures by hand can differ in the last place: this sweep
 * prints `127x` where 402 over 3.2 gives 126. Which rows disagree changes every
 * time the numbers are re-measured, so this is an illustration and not a fact
 * about the table — `tests/test_published_benchmarks.py` checks the rule against
 * the cells rather than against a remembered row.
 */

export {
  BENCHMARKS,
  BENCHMARK_CONCURRENCY,
  BENCHMARK_GPU,
} from "./benchmarks.generated";

/** One backend's tok/s on both regimes, with no comparison of its own. */
export interface Rate {
  /** tok/s, one stream at a time. */
  single: string;
  /** tok/s aggregate, `BENCHMARK_CONCURRENCY` requests in flight. */
  concurrent: string;
}

/**
 * A vLLM configuration, against eager. Each figure compares the two backends on
 * the *same* workload — `single` against eager's single, `concurrent` against
 * eager's concurrent — never across regimes.
 *
 * Always a multiplier, never a percent: this column is scanned top to bottom,
 * and `+20%` beside `27x` makes the reader convert one to compare them. A
 * multiplier throughout costs the small win nothing — `1.2x` next to a printed
 * 31 and 38 is checkable on the spot.
 */
export interface Comparison extends Rate {
  singleVsEager: string;
  concurrentVsEager: string;
}

/**
 * A figure's numeric value, for anything that has to size something by it — the
 * chart's bars.
 *
 * Parsed from the display string rather than carried as a second numeric field:
 * a bar drawn from the number printed on it cannot disagree with that number,
 * and the rounding it inherits is worth well under a pixel at this height. The
 * comma is the only thing standing between the two forms.
 */
export function tokensPerSecond(figure: string): number {
  return Number(figure.replace(/,/g, ""));
}

export interface Benchmark {
  /** Checkpoint, named as the sweep names it: family and size, no owner. */
  model: string;
  eager: Rate;
  /** Hooked vLLM: `enforce_eager=True`, the default. Serves every point. */
  vllm: Comparison;
  /**
   * `static_points="auto"` — CUDA-graph replay with static `copy_` taps, so
   * capture and steering survive the graph. Null where the sweep did not run it:
   * it is restricted to conventional trunks, and `"auto"` refuses one carrying
   * parallel residual streams.
   */
  static: Comparison | null;
  /**
   * Every way this row's conditions depart from the shared ones, one short
   * phrase each, in markdown's inline-code spelling because
   * `benchmarks/cells.py` writes the same phrases into the full report.
   *
   * Absent on a row that ran under exactly the shared conditions, which is most
   * of them, so the presence of the field is what the card's mark keys on. The
   * card shows the mark and not the phrases; they are what the reader finds on
   * the other end of the link, under *Where a row differs*.
   */
  differs?: string[];
}
