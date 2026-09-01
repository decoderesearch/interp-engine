"use client";

/**
 * One card, spent term by term, with the overflow drawn rather than described.
 *
 * The card rather than the budget, which is a deliberate change from what this drew first. vLLM
 * divides the card at a utilization line — the pool holds the weights, the context and the cache,
 * and the slice outside it absorbs warmup overshoot and fragmentation — and the two shortfalls are
 * fixed by turning the same knob opposite ways. Drawing them as separate bars said that, and cost
 * more than it was worth: two scales, two legends and two "x of y" figures for one card, so the
 * question everyone actually arrives with — how much of this card is left — took arithmetic. One
 * bar answers it and keeps the division: the pool's terms, the pool's remainder, then the terms
 * charged past the line and what is left beyond them, which is `estimate`'s own arithmetic laid
 * end to end and adds back up to the card exactly. The bracket above marks where the line falls.
 *
 * The scale is the larger of the card and what the terms need, so an overflowing bar stays in
 * proportion: at 1.5x the card the red is half the bar, not a token sliver at the end. The capacity
 * marker is where the card ran out.
 */

import { useState } from "react";

import { GIB, type MemoryTerm } from "@/lib/size";

/**
 * A colour per term, and stable across every bar in the panel so a segment means the same thing
 * wherever it appears. Grouped by what the term *is*: the model in sky, the cache in emerald, the
 * things a caller chose in violet, and vLLM's own unavoidable overheads in slate, so a glance at a
 * bar says whether it is the model or the settings taking the room.
 */
const COLOR: Record<string, string> = {
  weights: "bg-sky-600",
  kv_cache_floor: "bg-emerald-500",
  cuda_context: "bg-slate-400",
  vllm_overshoot: "bg-slate-300",
  fragmentation: "bg-slate-200",
  static_buffers: "bg-violet-500",
  graph_pool: "bg-fuchsia-500",
  reserved: "bg-amber-500",
  // Eager's prompt-driven terms, which are the ones that OOM it.
  logits: "bg-amber-500",
  attention: "bg-rose-500",
  hidden_states: "bg-emerald-500",
  mlp_intermediate: "bg-teal-500",
  capture_buffers: "bg-violet-500",
  workspace: "bg-slate-300",
};

const gib = (bytes: number) => `${(bytes / GIB).toFixed(1)} GiB`;

/** What the model asked for before what the engine spends around it. Eager has one side and sorts flat. */
const SIDE_ORDER: Record<string, number> = { pool: 0, eager: 0, outside: 1 };

export function VramBar({
  label,
  hint,
  capacityBytes,
  poolBytes = 0,
  terms,
}: {
  label: string;
  /** What the card is being asked to hold, since a bare capacity means nothing on its own. */
  hint: string;
  capacityBytes: number;
  /**
   * How far vLLM's pool reaches. `0` on eager, which has no pool and draws as one flat run.
   *
   * A budget rather than a total of anything below it, and it positions the layout as well as the
   * bracket: the terms outside the pool are charged past this line, so they start here rather than
   * against the last pool term. Where it lands is the whole question — a pool the weights nearly
   * fill is one `gpu_memory_utilization` away from fitting.
   */
  poolBytes?: number;
  terms: MemoryTerm[];
}) {
  const [hovered, setHovered] = useState<number | null>(null);

  const ordered = [...terms].sort(
    (a, b) => (SIDE_ORDER[a.side] ?? 0) - (SIDE_ORDER[b.side] ?? 0),
  );
  const needed = ordered.reduce((sum, term) => sum + term.bytes, 0);

  // The two budgets, drawn apart rather than packed together. `poolBytes` is a line across the card
  // that the outside terms are charged *past*, so laying every term end to end put fragmentation and
  // overshoot inside the pool -- the one region they are the definition of not being in. Each side
  // gets its own unused remainder, and the four add back up to the card exactly, which is the same
  // arithmetic `estimate` does to decide `fits`.
  const split = poolBytes > 0;
  const used = (side: string) =>
    ordered
      .filter((term) => term.side === side)
      .reduce((sum, term) => sum + term.bytes, 0);
  const poolFree = split ? Math.max(poolBytes - used("pool"), 0) : 0;
  const outsideFree = split
    ? Math.max(capacityBytes - poolBytes - used("outside"), 0)
    : Math.max(capacityBytes - needed, 0);

  const scale =
    Math.max(capacityBytes, needed + poolFree + outsideFree) || 1;
  const over = needed - capacityBytes;
  const pct = (bytes: number) => `${(bytes / scale) * 100}%`;

  // One list drives the bar, the legend and the hover card, so a segment cannot end up named one
  // thing in the key and another under the cursor. `fill` is what the bar paints and `swatch` is
  // what the key and the card draw: they differ only for a gap, which is transparent in the bar --
  // the track underneath already reads as room left -- but needs a visible chip on white.
  const term = (source: MemoryTerm) => ({
    name: source.name,
    bytes: source.bytes,
    fill: COLOR[source.name] ?? "bg-slate-400",
    swatch: COLOR[source.name] ?? "bg-slate-400",
    note: source.note,
  });
  const gap = (name: string, bytes: number) => ({
    name,
    bytes,
    fill: "",
    swatch: "bg-slate-100 ring-1 ring-slate-200 ring-inset",
    note: undefined as string | undefined,
  });

  let cursor = 0;
  const blocks = [
    ...ordered.filter((item) => item.side !== "outside").map(term),
    ...(poolFree > 0 ? [gap("pool free", poolFree)] : []),
    ...ordered.filter((item) => item.side === "outside").map(term),
    ...(outsideFree > 0 ? [gap("free", outsideFree)] : []),
  ]
    .filter((block) => block.bytes > 0)
    // Where each block starts, kept as bytes so the hover card can be centred on the segment in the
    // same units everything else here is measured in.
    .map((block) => {
      const start = cursor;
      cursor += block.bytes;
      return { ...block, start };
    });

  const active = hovered === null ? null : blocks[hovered];

  return (
    <div>
      <div className="flex items-baseline justify-between gap-x-2">
        <div className="min-w-0">
          <span className="text-[11px] font-medium text-slate-700">
            {label}
          </span>
          <span className="ml-1.5 text-[10px] text-slate-400">{hint}</span>
        </div>
        <span
          className={`shrink-0 font-mono text-[10px] ${
            over > 0 ? "font-semibold text-red-600" : "text-slate-500"
          }`}
        >
          {gib(needed)} / {gib(capacityBytes)}
          {over > 0 ? ` (over by ${gib(over)})` : ""}
        </span>
      </div>

      {/* Three borders on a short box: the top rule spans the pool and the two sides fall as notches
          at its ends, which is a bracket without a second element or an SVG. Rose because it is the
          one mark here that is not a quantity — every colour in the bar below is a term, and this is
          a line drawn across them, and the caption is rose to be read as part of the same mark. */}
      {poolBytes > 0 && (
        <>
          <div className="relative mt-1 h-[9px]">
            <span
              style={{ left: pct(poolBytes / 2) }}
              className="absolute -translate-x-1/2 text-[8px] leading-none font-semibold tracking-wider text-rose-400 uppercase"
            >
              vllm util - {Math.round((poolBytes / capacityBytes) * 100)}%
            </span>
          </div>
          <div className="relative h-[5px]">
            <div
              style={{ width: pct(poolBytes) }}
              className="absolute inset-y-0 left-0 border-x border-t border-rose-400"
            />
          </div>
        </>
      )}

      {/* A hover card rather than the `title` each segment used to carry: a thin segment is the one
          that most needs naming and the one a native tooltip is slowest to name, and the outline is
          the part that answers "which of these am I reading". Wrapped in its own positioned box
          because the track clips its overflow, which is what keeps the bar's corners rounded. */}
      <div className={`relative ${poolBytes > 0 ? "mt-0.5" : "mt-1"}`}>
        <div
          onMouseLeave={() => setHovered(null)}
          className="relative flex h-5 w-full overflow-hidden rounded-sm bg-slate-100"
        >
          {blocks.map((block, index) => (
            <div
              key={`${block.name}-${index}`}
              onMouseEnter={() => setHovered(index)}
              style={{ width: pct(block.bytes) }}
              className={`h-full ${block.fill} ${
                hovered === index ? "ring-1 ring-slate-700 ring-inset" : ""
              }`}
            />
          ))}

          {/* Where the budget ran out. Only drawn when something is past it: on a bar that fits, the
              marker would sit at the far right and read as an edge rather than a limit. */}
          {over > 0 && (
            <div
              style={{ left: pct(capacityBytes) }}
              className="pointer-events-none absolute inset-y-0 w-0.5 -translate-x-px bg-red-600"
            >
              <div className="absolute inset-y-0 left-0.5 w-[3px] bg-[repeating-linear-gradient(45deg,rgb(220_38_38/0.9)_0_2px,transparent_2px_4px)]" />
            </div>
          )}
        </div>

        {active && (
          <div
            // Clamped off the ends so a segment at either edge does not push the card outside the
            // panel. The outline says which segment this is, so the card does not have to point.
            style={{
              left: `${Math.min(
                Math.max(((active.start + active.bytes / 2) / scale) * 100, 12),
                88,
              )}%`,
            }}
            className="pointer-events-none absolute bottom-full z-20 mb-1 -translate-x-1/2 rounded-md border border-slate-300 bg-white px-2 py-1 shadow-lg"
          >
            <span className="flex items-baseline gap-x-1 text-[10px] whitespace-nowrap text-slate-600">
              <span
                className={`h-2 w-2 shrink-0 self-center rounded-[2px] ${active.swatch}`}
              />
              {active.name.replace(/_/g, " ")}
              <span className="font-mono text-slate-400">
                {gib(active.bytes)}
              </span>
            </span>
            {active.note && (
              <p className="mt-0.5 max-w-[15rem] text-[9px] leading-snug text-slate-400">
                {active.note}
              </p>
            )}
          </div>
        )}
      </div>

      {/* The key reads off `blocks`, so it lists the two remainders the bar draws and lists them in
          the order the bar draws them. The gaps earn their entry: they are the segments nobody
          painted and the ones most people are here to size, and only the first is room the cache
          can grow into — the difference between raising `max_model_len` and `gpu_memory_utilization`. */}
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5">
        {blocks.map((block, index) => (
          <span
            key={`${block.name}-${index}`}
            className="flex items-baseline gap-x-1 text-[9px] whitespace-nowrap text-slate-500"
          >
            {/* Baseline, not centre: the label is sans and the figure is mono, and the two font
                families give their line boxes different heights, so centring the boxes leaves the
                glyphs sitting at visibly different levels. The swatch is an empty block with no
                baseline of its own — flexbox would align its bottom edge — so it centres itself. */}
            <span
              className={`h-2 w-2 shrink-0 self-center rounded-[2px] ${block.swatch}`}
            />
            {block.name.replace(/_/g, " ")}
            <span className="font-mono text-slate-400">{gib(block.bytes)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
