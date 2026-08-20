"use client";

/**
 * One hook point, in two halves: `HookPoint` is the glyph, `HookLabel` the name
 * beside it. The glyph encodes how wide the tensor is — a single mark for a d_model
 * vector, one mark per unit for a fanned tensor, and a band with a count once there
 * are too many units to tell apart.
 *
 * Two components because they are drawn in two passes, all the glyphs and then all
 * the names. Both halves answer to the pointer, and a glyph never overlaps another
 * point's glyph — the row clearances see to that — but a name is wider than the
 * column it is centred in and does overlap what is beside it, including the next
 * point's touch target. Painted in one pass they would trade: the tip of a long name
 * would reach whichever point was drawn later. In two, a name always wins over a
 * glyph, and the only thing left to arbitrate is one name against another, which the
 * pane does by drawing the point already being pointed at last.
 */

import type { ReactNode } from "react";

import { engine as getEngine } from "@/data/engines";
import { COL_W, LABEL_FONT, glyphFor, labelBox, spacing } from "@/lib/layout";
import type { EngineId, GraphNode, Role, TraitId } from "@/lib/types";

const FILL: Record<Role, string> = {
  resid: "fill-role-resid",
  attn: "fill-role-attn",
  mlp: "fill-role-mlp",
  route: "fill-role-route",
  global: "fill-role-global",
};

const STROKE: Record<Role, string> = {
  resid: "stroke-role-resid",
  attn: "stroke-role-attn",
  mlp: "stroke-role-mlp",
  route: "stroke-role-route",
  global: "stroke-role-global",
};

const SOFT_FILL: Record<Role, string> = {
  resid: "fill-role-resid-soft",
  attn: "fill-role-attn-soft",
  mlp: "fill-role-mlp-soft",
  route: "fill-role-route-soft",
  global: "fill-role-global-soft",
};

interface Shared {
  node: GraphNode;
  engineId: EngineId;
  traits: Set<TraitId>;
  /** Under the pointer or holding the pin: both halves light together. */
  active: boolean;
  /** Trimmed, lowercased search term, or "" when the field is empty. */
  search: string;
  /** In compare mode, whether the other architecture disagrees about this point. */
  differs: boolean;
  /** Steps back while one architecture difference is being pointed at. */
  faded: boolean;
  onHover: (node: GraphNode | null) => void;
  onSelect: (node: GraphNode) => void;
}

interface PointProps extends Shared {
  /**
   * Non-null when the last trait change reworked this point's derivation. The
   * value is the change's token, which remounts the ripple so re-toggling the
   * same trait plays it again.
   */
  flashToken: number | null;
  /** Pinned by a click: the card stays on it and the pointer stops moving it. */
  locked: boolean;
}

interface LabelProps extends Shared {
  /**
   * The tighter spacing the graph was laid out with. It has to be the same value
   * `rowMetrics` used, since the row clearances are sized for exactly the label
   * offsets here — read one from the other's table rather than tuning them apart.
   */
  compact: boolean;
}

/**
 * What this naming stack calls the point, and whether the search is asking about
 * it. Both halves need both: the glyph rings a hit and fades a miss, the label
 * prints the name.
 */
function describe(
  node: GraphNode,
  engineId: EngineId,
  traits: Set<TraitId>,
  search: string,
) {
  const { text } = getEngine(engineId).format({
    point: node.point,
    layer: node.layer,
    stream: node.stream,
    traits,
  });

  return {
    // The layer is already written on the band, so a `blocks.5.` prefix on every
    // label would cost half the column width to say it again. The search still
    // reads the whole name, so `blocks.5` finds one layer.
    label: text?.replace(/^blocks\.\d+\./, "") ?? null,
    matched: search ? Boolean(text?.toLowerCase().includes(search)) : null,
  };
}

/**
 * The wrapper both halves share. Written once because a handler on one and not the
 * other is a point that answers to its glyph and not to its name, which is exactly
 * the bug this file is arranged to avoid.
 */
function Target({
  node,
  opacity,
  onHover,
  onSelect,
  children,
}: {
  node: GraphNode;
  opacity: number;
  onHover: (node: GraphNode | null) => void;
  onSelect: (node: GraphNode) => void;
  children: ReactNode;
}) {
  return (
    <g
      // Read by the pane's outside-click handler to tell a click that moves the
      // lock from one that releases it.
      data-hook-point
      className="cursor-pointer"
      opacity={opacity}
      onPointerEnter={() => onHover(node)}
      onPointerLeave={() => onHover(null)}
      onClick={() => onSelect(node)}
    >
      {children}
    </g>
  );
}

export function HookPoint({
  node,
  engineId,
  traits,
  active,
  flashToken,
  search,
  differs,
  faded,
  locked,
  onHover,
  onSelect,
}: PointProps) {
  const glyph = glyphFor(node.fanout);
  const refused = Boolean(node.refusal);
  const { matched } = describe(node, engineId, traits, search);

  return (
    <Target
      node={node}
      onHover={onHover}
      onSelect={onSelect}
      opacity={
        (refused ? 0.34 : 1) *
        (matched === false ? 0.25 : 1) *
        (faded ? 0.2 : 1)
      }
    >
      {flashToken !== null && (
        <g
          key={flashToken}
          style={
            {
              "--ripple-from": (glyph.half + 4) / (glyph.half + 26),
            } as React.CSSProperties
          }
        >
          {["hook-ripple", "hook-ripple hook-ripple-late"].map((className) => (
            <circle
              key={className}
              className={`${className} ${STROKE[node.role]}`}
              cx={node.x}
              cy={node.y}
              r={glyph.half + 26}
              fill="none"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </g>
      )}

      {/* A diff mark outranks a search hit: in compare mode the red is the
          thing being looked for. */}
      {(differs || matched) && (
        <rect
          x={node.x - 11}
          y={node.y - glyph.half - 5}
          width={22}
          height={glyph.half * 2 + 10}
          rx={11}
          className={
            differs
              ? "fill-red-100 stroke-red-500"
              : "fill-sky-100 stroke-sky-500"
          }
          strokeWidth={1.25}
        />
      )}

      {active && (
        <circle
          cx={node.x}
          cy={node.y}
          r={glyph.half + 8}
          // A marked point already has a backdrop; a second opaque one would
          // paint over it.
          className={`${STROKE[node.role]} ${differs || matched ? "fill-none" : "fill-white"}`}
          strokeWidth={locked ? 2 : 1.5}
          opacity={locked ? 1 : 0.9}
        />
      )}

      {/* A pinned point has to look different from one merely under the pointer,
          or the card staying put reads as the diagram having stuck. */}
      {locked && (
        <circle
          cx={node.x}
          cy={node.y}
          r={glyph.half + 13}
          className={STROKE[node.role]}
          fill="none"
          strokeWidth={1}
          opacity={0.4}
        />
      )}

      {glyph.kind === "dot" && (
        <circle
          cx={node.x}
          cy={node.y}
          r={5.5}
          className={`${FILL[node.role]} stroke-white`}
          strokeWidth={1.75}
          strokeDasharray={refused ? "2 2" : undefined}
        />
      )}

      {glyph.kind === "stack" &&
        Array.from({ length: glyph.count }, (_, i) => (
          <circle
            key={i}
            cx={node.x}
            cy={node.y + (i - (glyph.count - 1) / 2) * glyph.spacing}
            r={Math.min(2.4, glyph.spacing / 2.4)}
            className={FILL[node.role]}
          />
        ))}

      {glyph.kind === "band" && (
        <>
          <rect
            x={node.x - 5.5}
            y={node.y - glyph.half}
            width={11}
            height={glyph.half * 2}
            rx={5.5}
            className={`${SOFT_FILL[node.role]} ${STROKE[node.role]}`}
            strokeWidth={1.25}
          />
          {/* Ticks rather than a flat fill, so the band still reads as a
              countable set of units rather than one solid tensor. */}
          {tickOffsets(glyph.half).map((dy) => (
            <line
              key={dy}
              x1={node.x - 3}
              x2={node.x + 3}
              y1={node.y + dy}
              y2={node.y + dy}
              className={STROKE[node.role]}
              strokeWidth={1}
              opacity={0.5}
            />
          ))}
        </>
      )}

      {/* A touch target that does not depend on how small the glyph is. */}
      <rect
        x={node.x - 22}
        y={node.y - Math.max(22, glyph.half)}
        width={44}
        height={Math.max(44, glyph.half * 2)}
        fill="transparent"
      />
    </Target>
  );
}

/**
 * The point's name, drawn beside its glyph and hoverable in its own right — it is
 * the larger of the two targets, and on a point drawn as a single 11px dot it is
 * most of what there is to aim at.
 */
export function HookLabel({
  node,
  engineId,
  traits,
  active,
  search,
  differs,
  faded,
  compact,
  onHover,
  onSelect,
}: LabelProps) {
  // Only the top stream carries the label, or a hyper-connection trunk would
  // print the same name eight times in a column.
  if (node.stream !== null && node.stream !== 0) return null;

  const gaps = spacing(compact);
  const glyph = glyphFor(node.fanout);
  const refused = Boolean(node.refusal);
  const { label, matched } = describe(node, engineId, traits, search);

  // The count a band prints after its name is inside the same `<text>`, so it is
  // part of what the pill has to cover.
  const suffix = glyph.kind === "band" && label ? ` ×${glyph.count}` : "";
  const pill = labelBox((label?.length ?? 0) + suffix.length, compact);

  const above = node.role !== "mlp" && node.role !== "route";
  const stagger = Math.round(node.x / COL_W) % 2 === 0 ? 0 : gaps.stagger;
  const labelY = above
    ? node.y -
      glyph.half -
      gaps.labelAbove -
      stagger -
      (node.stream !== null ? gaps.streamLift : 0)
    : node.y + glyph.half + gaps.labelBelow + stagger;

  return (
    <Target
      node={node}
      onHover={onHover}
      onSelect={onSelect}
      opacity={
        (refused ? 0.34 : 1) *
        (matched === false ? 0.25 : 1) *
        (faded ? 0.2 : 1)
      }
    >
      {/* The name sits on a pill because the canvas underneath it is not blank
              — it is edges, layer bands and the odd neighbouring glyph, and small
              grey type crossing a curve is what was hard to read. Sized from a
              character count rather than a measured box: see `labelBox`.

          `slate-200`, not `slate-100`, which is what the alternating layer
          bands are filled with at half opacity — a pill in that colour would be
          visible on odd layers and invisible on even ones.

          Its outline does two jobs, one per state. At rest it is the canvas'
          own white and shows only where two pills touch, which they do: names
          run to sixteen characters in a 58px column, `expert_weights.1` next to
          `router_logits.1` overlapped as bare text too, and without it the two
          merge into one shape with a word buried in it. Under the pointer it
          becomes a `sky-600` border, which is what carries the highlight at a
          glance — `sky-100` against `slate-200` is a hue change of about the
          same lightness, easy to miss from across a diagram this wide.

              Unpainted under the em dash, which is the deliberately faint "this
              stack has no name for it" mark — a badge there would give the point
              with nothing to say more presence than the ones that do. Still drawn,
              though, and a transparent fill still takes a pointer: the dash is that
              point's label, and it should answer to a click like any other. */}
      <rect
        x={node.x - pill.width / 2}
        y={labelY - pill.top}
        width={pill.width}
        height={pill.height}
        rx={pill.height / 2}
        className={
          label === null
            ? "fill-transparent"
            : active
              ? "fill-sky-100 stroke-sky-600"
              : "fill-slate-200 stroke-white"
        }
        strokeWidth={active ? 1 : 0.75}
      />
      <text
        x={node.x}
        y={labelY}
        textAnchor="middle"
        fontSize={LABEL_FONT}
        className={`font-mono ${
          differs
            ? "fill-red-600"
            : matched
              ? "fill-sky-700"
              : label === null
                ? "fill-slate-300"
                : active
                  ? "fill-sky-900"
                  : "fill-slate-600"
        } ${active || matched || differs ? "font-semibold" : ""}`}
      >
        {label ?? "—"}
        {suffix ? (
          <tspan className={active ? "fill-sky-700" : "fill-slate-500"}>
            {suffix}
          </tspan>
        ) : null}
      </text>
    </Target>
  );
}

function tickOffsets(half: number): number[] {
  const step = 3.5;
  const count = Math.floor((half * 2 - 8) / step);
  const start = -((count - 1) * step) / 2;
  return Array.from(
    { length: count },
    (_, i) => Math.round((start + i * step) * 10) / 10,
  );
}
