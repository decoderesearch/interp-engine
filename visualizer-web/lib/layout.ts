/**
 * Geometry: glyph sizes, row placement and path shapes.
 *
 * Rows are integers around the residual spine — 0 is the spine, negative rows
 * are the attention branch above it, positive rows the MLP branch below. Row
 * *positions* are derived from the glyphs that land on them rather than from a
 * constant, because a 300-neuron MLP draws a mark per neuron and a fixed row
 * pitch would have it overlap its neighbours.
 */

export const COL_W = 58;
export const PAD_X = 56;

/**
 * The label's own box: a point's name is drawn on a pill, so it has a width and a
 * height rather than just a baseline, and both are in the clearance arithmetic
 * below. Monospace is what makes the width knowable without measuring in the DOM —
 * every glyph in Geist Mono is 0.6em wide, so a character count is a width, and the
 * padding absorbs the small differences in the system fallbacks.
 *
 * Its height comes from `Spacing`, because vertical padding is the same scarce
 * resource as every other number in that table and is the one thing here that a
 * phone in compare mode cannot afford at full size.
 */
export const LABEL_FONT = 9.5;
const LABEL_CHAR_W = 0.6;
const LABEL_PAD_X = 6;

export interface LabelBox {
  width: number;
  /** Above the text's baseline: the ink reaches ~7.2px of it at this size. */
  top: number;
  height: number;
}

export function labelBox(chars: number, compact = false): LabelBox {
  const gaps = spacing(compact);
  return {
    width: round(chars * LABEL_FONT * LABEL_CHAR_W + LABEL_PAD_X * 2),
    top: gaps.pillTop,
    height: gaps.pillHeight,
  };
}

/**
 * Vertical geometry, in two sizes.
 *
 * Every number here is spent on the same thing: room for a point's label. The
 * clearances are edge-to-edge between glyphs, and a label sits above its glyph
 * by `labelAbove` plus, on every other column, `stagger` — so a row clearance
 * that does not cover `labelAbove + stagger + labelBox().top` puts one column's
 * label through the row above it. Change one and check the other.
 *
 * `stagger` has a second job now that labels are drawn on pills: it is what keeps
 * two neighbouring columns' pills from overlapping, so it must be at least
 * `pillHeight`. That is the whole reason `compact` gives the pill less vertical
 * padding — a phone can afford neither the 2px inside the pill nor the 2px of row
 * clearance that a stagger clearing it would then need.
 *
 * `compact` is for two diagrams sharing the height of a phone, where a row of
 * empty pixels costs a row of points. It is the same drawing with the label room
 * cut to what a label actually occupies rather than to what reads comfortably.
 */
export interface Spacing {
  /** Above the topmost glyph and below the bottom one; holds the top labels. */
  padY: number;
  /**
   * Extra room at the top only, for the band's own `Layer N`.
   *
   * Without it the two share a strip. A top label's pill reaches
   * `padY - labelAbove - stagger - pillTop` from the canvas edge, which at the
   * normal sizes is 24.8 -- and the band label's ink runs 26.8 to 34, so they
   * occupy the same 13px and collide whenever the leftmost column of a layer
   * happens to be a staggered one. Zero when compact, where the smaller
   * `padY` already puts those pills clear above the band label rather than
   * across it.
   */
  bandLabel: number;
  /** Between two rows' glyphs. */
  row: number;
  /** Between the spine and the first row of either branch. */
  branch: number;
  /** Glyph edge to label baseline, above the glyph. */
  labelAbove: number;
  /** Glyph edge to label baseline, below the glyph — a descender further out. */
  labelBelow: number;
  /** Every other column's label steps this far out, so neighbours clear. */
  stagger: number;
  /** Extra lift for a stream label, which has the stream index under it. */
  streamLift: number;
  /** The label pill's reach above the text's baseline. */
  pillTop: number;
  /** The label pill's height, which `stagger` has to clear. */
  pillHeight: number;
}

const NORMAL: Spacing = {
  padY: 56,
  bandLabel: 16,
  row: 34,
  branch: 62,
  labelAbove: 9,
  labelBelow: 14,
  stagger: 13,
  streamLift: 8,
  pillTop: 9.2,
  pillHeight: 13,
};

const COMPACT: Spacing = {
  padY: 36,
  bandLabel: 0,
  row: 27,
  branch: 40,
  labelAbove: 7,
  labelBelow: 11,
  stagger: 11,
  streamLift: 6,
  pillTop: 8.2,
  pillHeight: 11,
};

export function spacing(compact = false): Spacing {
  return compact ? COMPACT : NORMAL;
}

/** Above this fanout a glyph switches from one mark per unit to a dense band. */
export const FANOUT_DOT_LIMIT = 48;

export type GlyphKind = "dot" | "stack" | "band";

export interface Glyph {
  kind: GlyphKind;
  /** Half the glyph's vertical extent, which is what row placement needs. */
  half: number;
  spacing: number;
  count: number;
}

export function glyphFor(fanout: number): Glyph {
  if (fanout <= 1) return { kind: "dot", half: 6, spacing: 0, count: 1 };
  if (fanout <= FANOUT_DOT_LIMIT) {
    const spacing = fanout <= 10 ? 6 : fanout <= 20 ? 4.5 : 3.4;
    return {
      kind: "stack",
      half: ((fanout - 1) * spacing) / 2 + 2.5,
      spacing,
      count: fanout,
    };
  }
  // Past the point where marks are distinguishable, keep growing but slowly, so
  // 512 reads as visibly more than 64 without taking the whole canvas.
  const height = Math.min(196, 84 + 30 * Math.log2(fanout / FANOUT_DOT_LIMIT));
  return { kind: "band", half: height / 2, spacing: 0, count: fanout };
}

/** Vertical separation between parallel residual streams. */
export function streamGap(streams: number): number {
  return streams <= 3 ? 26 : streams <= 5 ? 20 : 15;
}

export interface RowMetrics {
  /** Row index to its y offset from the spine. */
  offsets: Map<number, number>;
  above: number;
  below: number;
}

/**
 * Place each row far enough from its neighbour to clear both glyphs.
 * `maxFanout` is the widest tensor that lands on each row.
 */
export function rowMetrics(
  maxFanout: Map<number, number>,
  compact = false,
): RowMetrics {
  const offsets = new Map<number, number>([[0, 0]]);
  const gaps = spacing(compact);

  const walk = (rows: number[], direction: 1 | -1) => {
    let edge = 0;
    for (const row of rows) {
      const half = glyphFor(maxFanout.get(row) ?? 1).half;
      const clearance = edge === 0 ? gaps.branch : gaps.row;
      const center = edge + clearance + half;
      offsets.set(row, direction * center);
      edge = center + half;
    }
    return edge;
  };

  const aboveRows = [...maxFanout.keys()]
    .filter((r) => r < 0)
    .sort((a, b) => b - a);
  const belowRows = [...maxFanout.keys()]
    .filter((r) => r > 0)
    .sort((a, b) => a - b);

  const above = walk(aboveRows, -1);
  const below = walk(belowRows, 1);
  return { offsets, above, below };
}

/** A horizontal run along the spine, or between two nodes on the same row. */
export function linePath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): string {
  return `M ${round(x1)} ${round(y1)} L ${round(x2)} ${round(y2)}`;
}

/**
 * An S-curve between rows. The control points are pulled horizontally so the
 * curve leaves and arrives flat, which keeps a fan of edges into one node from
 * crossing each other near the endpoint.
 */
export function curvePath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): string {
  if (Math.abs(y1 - y2) < 0.5) return linePath(x1, y1, x2, y2);
  const dx = Math.max(28, Math.abs(x2 - x1) * 0.55);
  return `M ${round(x1)} ${round(y1)} C ${round(x1 + dx)} ${round(y1)}, ${round(
    x2 - dx,
  )} ${round(y2)}, ${round(x2)} ${round(y2)}`;
}

function round(n: number): number {
  return Math.round(n * 100) / 100;
}
