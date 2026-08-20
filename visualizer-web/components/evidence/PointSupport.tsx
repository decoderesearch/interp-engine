"use client";

/**
 * Every point against both backends, from `data/points.ts` — the same table the
 * diagram is drawn from, so the two cannot disagree about which backend serves
 * what.
 *
 * Two shapes of the one list, because the two places that make the
 * standardization claim have opposite budgets. The **Standardized** hover card
 * under the wordmark is 340px wide and as tall as the window allows, so there it
 * is a table that scrolls down. The tour's slide is 720px wide inside a dialog
 * that already scrolls vertically, and a second vertical scroll inside it is a
 * scrollbar beside a scrollbar — so there the same rows run down five columns
 * and scroll sideways instead.
 *
 * The eager column is a constant, and printed anyway: "eager captures all of
 * them" is exactly the claim, and a single vLLM column would leave the reader
 * guessing what a missing check meant.
 *
 * Two marks, though `data/points.ts` has three states. It separates a point vLLM
 * serves from its worker hooks from one it rebuilds off-kernel from captured
 * q/k, and the point's own card on the diagram says which and quotes the
 * engine's reason — but that is a fact about how the engine gets there, not about
 * whether you can have the tensor. The question these two ask is what the
 * backend serves, and a recompute is served. A third mark spends a legend entry
 * and an amber glyph putting an asterisk on a claim that does not have one.
 */

import { Check, X } from "lucide-react";

import { ALL_POINTS } from "@/data/points";
import type { VllmSupport } from "@/lib/types";
import { cn } from "@/lib/utils";

/** In legend order, which is best case first. */
const SUPPORT = [
  { icon: Check, className: "text-emerald-600", label: "served" },
  { icon: X, className: "text-slate-300", label: "not served" },
] as const;

const [SERVED, NOT_SERVED] = SUPPORT;

export function PointSupport() {
  return (
    <>
      <Heading />

      <div className="thin-scrollbar mt-2 max-h-[min(46dvh,300px)] overflow-y-auto overscroll-contain">
        <table className="w-full border-collapse text-left text-[10px]">
          {/* Sticky, because the two columns are only telling you apart while
              their headings are on screen, and this table is 34 rows in 300px. */}
          <thead className="sticky top-0 bg-white">
            <tr className="text-[9px] tracking-wide text-slate-400">
              <th className="pb-1 pr-2 font-medium uppercase">point</th>
              <th className="px-2 pb-1 text-center font-medium uppercase">
                eager
              </th>
              <th className="pb-1 pl-2 text-center font-medium">vLLM</th>
            </tr>
          </thead>
          <tbody>
            {ALL_POINTS.map((spec) => (
              <tr key={spec.name} className="border-t border-slate-100">
                <td className="py-1 pr-2 font-mono text-slate-600">
                  {spec.name}
                </td>
                <td className="px-2 py-1">
                  <Served value="hooks" />
                </td>
                <td className="py-1 pl-2">
                  <Served value={spec.vllm} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SupportLegend />
    </>
  );
}

/**
 * Five columns, filled down then across. 34 names at seven rows is one empty
 * cell; six rows was six columns, which the narrower tour dialog cannot hold
 * without a long sideways scroll.
 */
const GRID_COLUMNS = 5;
const GRID_ROWS = Math.ceil(ALL_POINTS.length / GRID_COLUMNS);

/**
 * The same points laid out to be read across rather than scrolled through.
 *
 * Names only. The tour's claim is that the list exists and is one list, not
 * which backend serves which row — that detail still lives on the hover card
 * under the wordmark, and on every point's own card on the diagram.
 */
export function PointGrid() {
  return (
    <div>
      <div className="text-xs font-bold text-slate-700">Supported Points</div>
      <div className="thin-scrollbar mt-2 overflow-x-auto overscroll-contain pb-2">
        {/* `w-max` is what makes the row overflow rather than wrap: a grid with
            `grid-flow-col` inside a block that is only as wide as its parent still
            shrinks its columns to fit, and the names are monospace and unbreakable. */}
        <div
          className="grid w-max grid-flow-col gap-x-6 gap-y-2.5"
          style={{ gridTemplateRows: `repeat(${GRID_ROWS}, min-content)` }}
        >
          {ALL_POINTS.map((spec) => (
            <span
              key={spec.name}
              className="flex items-center gap-x-1.5 font-mono text-[10.5px] whitespace-nowrap text-slate-600"
            >
              <Check
                className="h-2.5 w-2.5 shrink-0 text-emerald-600"
                aria-hidden
              />
              {spec.name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

function Heading() {
  return (
    <div className="flex items-baseline justify-between gap-x-2">
      <span className="text-xs font-semibold text-slate-700">
        {ALL_POINTS.length} standardized points
      </span>
      <span className="shrink-0 text-[9px] font-medium tracking-wide text-slate-400 uppercase">
        one name per tensor
      </span>
    </div>
  );
}

function SupportLegend() {
  return (
    <div className="mt-2 flex items-center gap-x-2.5 border-t border-slate-100 pt-1.5 text-[9px] text-slate-400">
      {SUPPORT.map(({ icon: Icon, className, label }) => (
        <span key={label} className="flex items-center gap-x-1">
          <Icon className={cn("h-2.5 w-2.5", className)} aria-hidden />
          {label}
        </span>
      ))}
    </div>
  );
}

function Served({ value }: { value: VllmSupport }) {
  const { icon: Icon, className, label } =
    value === "none" ? NOT_SERVED : SERVED;
  return (
    <span className="flex shrink-0 justify-center">
      <Icon className={cn("h-3 w-3", className)} aria-hidden />
      <span className="sr-only">{label}</span>
    </span>
  );
}
