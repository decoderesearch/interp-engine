/**
 * What changed between two architectures, point by point.
 *
 * A point counts as different when one side has it and the other does not, or
 * when both have it but arrive at it differently. "Differently" is read off the
 * things that decide what the tensor actually is: its derivation, whether the
 * engine can reach it here, how wide it is, and whether it has been merged with
 * a neighbouring point because this architecture has nothing between them.
 *
 * Deliberately *not* included: position on the canvas. Two graphs drawn against
 * a shared alignment mostly agree on geometry, but a column that only one side
 * fills shifts nothing and means nothing on its own.
 */

import { formulaChanges, formulaFor } from "@/data/formulas";
import { ALL_POINTS } from "@/data/points";
import { TRAITS, resolveTraits } from "@/data/traits";
import type { Graph, GraphNode, PointName, Trait, TraitId } from "@/lib/types";

export interface Side {
  graph: Graph;
  traits: Set<TraitId>;
}

/** Node ids that differ, across both sides. Ids are unique per point/layer/stream. */
export function diffGraphs(left: Side, right: Side): Set<string> {
  const differing = new Set<string>();
  const here = new Map(left.graph.nodes.map((node) => [node.id, node]));
  const there = new Map(right.graph.nodes.map((node) => [node.id, node]));

  for (const [id, node] of here) {
    const other = there.get(id);
    if (!other || signature(node, left) !== signature(other, right)) {
      differing.add(id);
    }
  }
  for (const id of there.keys()) {
    if (!here.has(id)) differing.add(id);
  }

  return differing;
}

export interface TraitDelta {
  trait: Trait;
  /** True when this side has the trait and the other one does not. */
  has: boolean;
}

/**
 * The same comparison one level up: what this architecture is, said relative to
 * the other one. Ordered by `TRAITS`, which groups attention before norms
 * before the MLP, so the list reads the way the diagram does.
 */
export function traitDiff(
  mine: Set<TraitId>,
  theirs: Set<TraitId>,
): TraitDelta[] {
  return TRAITS.filter((t) => mine.has(t.id) !== theirs.has(t.id)).map((t) => ({
    trait: t,
    has: mine.has(t.id),
  }));
}

export interface PointEffects {
  adds: PointName[];
  removes: PointName[];
  /** Rendered as `merged / distinct`, so these are labels rather than names. */
  splits: string[];
  reworks: PointName[];
  /** Every point named above, for callers that only need "did this trait touch it". */
  touched: Set<PointName>;
}

/**
 * What toggling a trait does to the point set, and to the arithmetic behind the
 * points that stay.
 *
 * The structural three come off the point table; the fourth is measured against
 * the trait set given, because that is what toggling it there would actually
 * do. Several traits have only that fourth effect — fused QKV changes where
 * `value` comes from without moving anything — and without it their hover card
 * would claim they do nothing.
 *
 * Used twice over: to write the trait's hover card, and to decide which of the
 * red rings in compare mode belong to it. Sharing the answer is the point —
 * a card that says it rewrites `mlp_out` should light up `mlp_out`.
 */
export function pointEffects(id: TraitId, traits: Set<TraitId>): PointEffects {
  const adds: PointName[] = [];
  const removes: PointName[] = [];
  const splits: string[] = [];
  const touched = new Set<PointName>();

  for (const spec of ALL_POINTS) {
    if ((spec.requires ?? []).includes(id)) {
      adds.push(spec.name);
      touched.add(spec.name);
    }
    if (spec.refusedBy?.[id]) {
      removes.push(spec.name);
      touched.add(spec.name);
    }
    if (spec.distinctUnder === id) {
      splits.push(`${spec.mergedWith} / ${spec.name}`);
      touched.add(spec.name);
      if (spec.mergedWith) touched.add(spec.mergedWith);
    }
    if (spec.mergedUnder === id) {
      removes.push(spec.name);
      touched.add(spec.name);
    }
  }

  const toggled = new Set(traits);
  if (toggled.has(id)) toggled.delete(id);
  else toggled.add(id);
  const structural = new Set([...adds, ...removes]);
  const reworks = [...formulaChanges(traits, resolveTraits(toggled))].filter(
    (point) => !structural.has(point),
  );
  for (const point of reworks) touched.add(point);

  return { adds, removes, splits, reworks, touched };
}

function signature(node: GraphNode, side: Side): string {
  const isMoe =
    side.graph.layers.find((band) => band.index === node.layer)?.isMoe ?? false;
  const formula = formulaFor(node.point, {
    traits: side.traits,
    layer: node.layer,
    stream: node.stream,
    isMoe,
  });

  return [
    formula?.expr ?? "",
    formula?.note ?? "",
    node.refusal ?? "",
    node.fanout,
    [...node.alsoKnownAs].sort().join(","),
  ].join("\u0000");
}
