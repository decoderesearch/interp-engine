/**
 * Turns a dimension set and a trait set into the diagram's geometry.
 *
 * Pure, and the only place that knows how a transformer is wired. Two rules
 * keep it honest about what interp-engine would actually do:
 *
 *   - A point the architecture does not have is not drawn at all (no
 *     `resid_mid` on a parallel block, no routing points on a dense model).
 *   - A point the architecture has but the engine refuses on *this* layer is
 *     drawn dimmed, carrying the reason (attention points on the
 *     linear-attention layers of a hybrid stack).
 *
 * Edges come from a full successor relation over point names rather than from
 * "whatever was in the previous column". When a point is absent the walk passes
 * straight through it, so a model without QK-norm gets `attn_in -> attn_scores`
 * without anyone having to enumerate that case.
 */

import { ALL_POINTS, pointSpec } from "@/data/points";
import type {
  Dimensions,
  Graph,
  GraphEdge,
  GraphNode,
  LayerBand,
  LayerKind,
  PointName,
  PointSpec,
  Role,
  TraitId,
} from "@/lib/types";
import {
  COL_W,
  PAD_X,
  curvePath,
  linePath,
  rowMetrics,
  spacing,
  streamGap,
} from "@/lib/layout";
import type { RowMetrics } from "@/lib/layout";

/**
 * One architecture as the builder needs to know it, which is its traits and
 * nothing else. The class name is deliberately not here: every point's existence
 * is a consequence of how the model is wired, including the hyper-connection
 * ones, which `points_for` keys on the trunk it found rather than on a family.
 */
export interface BuildSide {
  traits: Set<TraitId>;
}

export interface BuildOptions extends BuildSide {
  dims: Dimensions;
  /**
   * Draw with the tighter vertical spacing — for two diagrams sharing the height
   * of a phone. It moves rows, so both sides of a comparison must be built with
   * the same value or the pair stops lining up.
   */
  compact?: boolean;
  /**
   * Geometry to share with another graph. Two architectures drawn one above
   * the other are only comparable if the same point lands at the same place in
   * both, so the layout is planned for the union of the two trait sets and each
   * side then draws the subset it actually has. Omit it and a graph is simply
   * aligned with itself, which is the ordinary single-diagram case.
   */
  align?: Alignment;
}

/** A column and row plan two graphs can be built against. */
export interface Alignment {
  /** Union of both sides' traits. Sets the rows and the stream spacing. */
  traits: Set<TraitId>;
  /** Only a block that is parallel on *both* sides can drop its resid_mid column. */
  parallel: boolean;
  /**
   * Each side as it decides existence for itself. A column is reserved when
   * either side has the point — which is not the same as asking whether the
   * union has it, since one side's `refusedBy` would then delete the other
   * side's column.
   */
  sides: (BuildSide & { moe: boolean[] })[];
}

export function alignmentFor(
  dims: Dimensions,
  a: BuildSide,
  b: BuildSide,
): Alignment {
  return {
    traits: new Set([...a.traits, ...b.traits]),
    parallel:
      a.traits.has("parallel_attn_mlp") && b.traits.has("parallel_attn_mlp"),
    sides: [
      { ...a, moe: moeLayers(dims, a.traits) },
      { ...b, moe: moeLayers(dims, b.traits) },
    ],
  };
}

const LINEAR_LAYER_REFUSAL =
  "This layer runs a linear-attention mixer, so no softmax attention tensor exists here.";

/** Three linear-attention layers per softmax one, as on Qwen3-Next. */
const HYBRID_PERIOD = 4;

/** Points that only exist where the layer runs real softmax attention. */
const SOFTMAX_ONLY = new Set([
  "q_norm_in",
  "q_norm_out",
  "k_norm_in",
  "k_norm_out",
  "value",
  "attn_scores",
  "attn_probs",
  "z",
]);

// --- the wiring -------------------------------------------------------------

/**
 * The complete forward relation between points inside one layer, including the
 * residual carries. Two variants, because a parallel block is not a rewiring of
 * the sequential one — it genuinely has no tensor between the sublayers.
 */
function successorsFor(parallel: boolean): Record<PointName, PointName[]> {
  const attn: Record<PointName, PointName[]> = {
    attn_stream_mix: ["attn_stream_collapse"],
    attn_stream_collapse: ["attn_in"],
    attn_in: ["q_norm_in", "k_norm_in", "value"],
    q_norm_in: ["q_norm_out"],
    q_norm_out: ["attn_scores"],
    k_norm_in: ["k_norm_out"],
    k_norm_out: ["attn_scores"],
    value: ["z"],
    attn_scores: ["attn_probs"],
    attn_probs: ["z"],
    z: ["attn_gate"],
    attn_gate: ["attn_out"],
    attn_out: ["attn_out_post"],
    attn_out_post: ["attn_stream_write"],
  };
  const mlp: Record<PointName, PointName[]> = {
    mlp_stream_mix: ["mlp_stream_collapse"],
    mlp_stream_collapse: ["mlp_in"],
    mlp_in: ["mlp_pre", "mlp_pre_linear", "router_logits"],
    mlp_pre: ["mlp_act"],
    mlp_pre_linear: ["mlp_act"],
    mlp_act: ["mlp_out"],
    router_logits: ["expert_weights", "expert_indices"],
    expert_weights: ["mlp_out"],
    expert_indices: ["mlp_out"],
    mlp_out: ["mlp_out_post"],
    mlp_out_post: ["mlp_stream_write"],
  };

  // `resid_streams` is the stack the per-stream trunk nodes are slices of, and each
  // sublayer's mixing matrix is read off the stack it collapses -- entry for
  // attention, mid-block for the MLP, which is the one the drawing has to hand
  // there. A single-stream trunk has none of the three, and the walk in
  // `resolveLinks` passes straight through them.
  if (parallel) {
    return {
      resid_pre: ["resid_streams", "resid_post"],
      resid_streams: ["attn_stream_mix", "mlp_stream_mix"],
      ...attn,
      attn_stream_write: ["resid_post"],
      ...mlp,
      mlp_stream_write: ["resid_post"],
      resid_post: [],
    };
  }
  return {
    resid_pre: ["resid_streams", "resid_mid"],
    resid_streams: ["attn_stream_mix"],
    ...attn,
    attn_stream_write: ["resid_mid"],
    resid_mid: ["mlp_stream_mix", "resid_post"],
    ...mlp,
    mlp_stream_write: ["resid_post"],
    resid_post: [],
  };
}

/**
 * Point-level edges over the subset of points that actually exist here.
 *
 * From every present point, walk forward; the first present point on each path
 * becomes an edge and the walk stops there. Absent points are transparent.
 */
function resolveLinks(
  present: Set<PointName>,
  successors: Record<PointName, PointName[]>,
): [PointName, PointName][] {
  const links: [PointName, PointName][] = [];
  for (const from of present) {
    const seen = new Set<PointName>();
    const queue = [...(successors[from] ?? [])];
    while (queue.length) {
      const next = queue.shift()!;
      if (seen.has(next)) continue;
      seen.add(next);
      if (present.has(next)) links.push([from, next]);
      else queue.push(...(successors[next] ?? []));
    }
  }
  return links;
}

// --- entry point ------------------------------------------------------------

export function buildGraph({
  dims,
  traits,
  align,
  compact = false,
}: BuildOptions): Graph {
  const side = { traits };
  const plan = align ?? alignmentFor(dims, side, side);
  const kinds = layerKinds(dims, traits);
  const moe = moeLayers(dims, traits);

  const rows = rowMetrics(rowFanouts(dims, plan.traits), compact);
  const pads = spacing(compact);
  const padY = pads.padY;
  // Asymmetric: the extra strip is added above the spine and not below, so it
  // buys clearance for the band's `Layer N` without also padding the bottom
  // edge, where there is nothing to clear.
  const spineY = padY + pads.bandLabel + rows.above;
  const height = spineY + rows.below + padY;

  // The trunk fans out once per stream this side actually has, but the streams
  // are spaced for the union — a single-stream model then sits on the centre
  // line of the band its multi-stream counterpart occupies.
  const streams = traits.has("multi_residual_streams")
    ? Math.max(2, dims.streams)
    : 1;
  const spread = plan.traits.has("multi_residual_streams")
    ? Math.max(2, dims.streams)
    : 1;
  const gap = streamGap(spread);
  const spineYs =
    streams === 1
      ? [spineY]
      : Array.from(
          { length: streams },
          (_, i) => spineY + (i - (streams - 1) / 2) * gap,
        );

  const ctx: Ctx = {
    dims,
    traits,
    sides: plan.sides,
    spineY,
    spineYs,
    streams,
    rows,
    aliases: aliasIndex(traits),
    successors: successorsFor(traits.has("parallel_attn_mlp")),
  };

  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const bands: LayerBand[] = [];

  let x = PAD_X;

  const embed = makeNode(ctx, "embeddings", null, x, 0, null);
  nodes.push(embed);
  x += COL_W;

  let previousExits: GraphNode[] = [embed];

  for (let layer = 0; layer < dims.layers; layer += 1) {
    const built = buildLayer(ctx, {
      layer,
      kind: kinds[layer],
      isMoe: moe[layer],
      parallel: plan.parallel,
      x0: x,
    });

    nodes.push(...built.nodes);
    edges.push(...built.edges);
    edges.push(...joinLayers(previousExits, built.entries));

    bands.push({
      index: layer,
      kind: kinds[layer],
      isMoe: moe[layer],
      x,
      width: built.width,
    });
    x += built.width;
    previousExits = built.exits;
  }

  const finalNorm = makeNode(ctx, "final_norm", null, x, 0, null);
  nodes.push(finalNorm);
  edges.push(...joinLayers(previousExits, [finalNorm]));
  x += COL_W;

  const head = makeNode(ctx, "lm_head", null, x, 0, null);
  nodes.push(head);
  edges.push(edge(finalNorm, head, "spine"));
  x += COL_W;

  return {
    nodes,
    edges,
    layers: bands,
    width: x + PAD_X,
    height,
    spineYs,
  };
}

/** Stream-matched where both ends sit on the trunk, all-to-all otherwise. */
function joinLayers(from: GraphNode[], to: GraphNode[]): GraphEdge[] {
  const out: GraphEdge[] = [];
  for (const source of from) {
    for (const target of to) {
      if (
        source.stream !== null &&
        target.stream !== null &&
        source.stream !== target.stream
      ) {
        continue;
      }
      out.push(edge(source, target, "spine"));
    }
  }
  return out;
}

// --- layer construction -----------------------------------------------------

interface Ctx {
  dims: Dimensions;
  traits: Set<TraitId>;
  /** Both sides of the comparison, for deciding which columns to reserve. */
  sides: Alignment["sides"];
  spineY: number;
  spineYs: number[];
  streams: number;
  rows: RowMetrics;
  aliases: Map<PointName, PointName[]>;
  successors: Record<PointName, PointName[]>;
}

interface BuiltLayer {
  nodes: GraphNode[];
  edges: GraphEdge[];
  entries: GraphNode[];
  exits: GraphNode[];
  width: number;
}

/** One column of the layer: the points on it and the row each sits on. */
type Stage = { point: PointName; row: number }[];

interface LayerPlan {
  layer: number;
  kind: LayerKind;
  /** Whether *this* side runs a sparse block here. */
  isMoe: boolean;
  parallel: boolean;
  x0: number;
}

function buildLayer(ctx: Ctx, plan: LayerPlan): BuiltLayer {
  const { layer, isMoe, parallel, x0 } = plan;
  const linear = plan.kind === "linear_attention";

  // Every point that could sit in this block is listed, and `reserve` keeps
  // the ones some side actually has. The sparse and dense MLP wirings share
  // their two columns, so a dense block and a sparse one stay in step.
  const attnStages = reserve(ctx, layer, [
    // The stack sits on the spine, inside the band its own streams are drawn in:
    // it is the same tensor they are slices of, one column along so that the
    // gather is a shape the eye can follow rather than a node on top of a node.
    [{ point: "resid_streams", row: 0 }],
    [{ point: "attn_stream_mix", row: -2 }],
    [{ point: "attn_stream_collapse", row: -1 }],
    [{ point: "attn_in", row: -1 }],
    [
      { point: "q_norm_in", row: -4 },
      { point: "k_norm_in", row: -3 },
      { point: "value", row: -2 },
    ],
    [
      { point: "q_norm_out", row: -4 },
      { point: "k_norm_out", row: -3 },
    ],
    [{ point: "attn_scores", row: -1 }],
    [{ point: "attn_probs", row: -1 }],
    [{ point: "z", row: -1 }],
    [{ point: "attn_gate", row: -1 }],
    [{ point: "attn_out", row: -1 }],
    [{ point: "attn_out_post", row: -1 }],
    [{ point: "attn_stream_write", row: -1 }],
  ]);

  const mlpStages = reserve(ctx, layer, [
    [{ point: "mlp_stream_mix", row: 2 }],
    [{ point: "mlp_stream_collapse", row: 1 }],
    [{ point: "mlp_in", row: 1 }],
    [
      { point: "router_logits", row: 2 },
      { point: "mlp_pre", row: 2 },
      { point: "mlp_pre_linear", row: 3 },
    ],
    [
      { point: "expert_weights", row: 2 },
      { point: "expert_indices", row: 3 },
      { point: "mlp_act", row: 1 },
    ],
    [{ point: "mlp_out", row: 1 }],
    [{ point: "mlp_out_post", row: 1 }],
    [{ point: "mlp_stream_write", row: 1 }],
  ]);

  // Column plan. Sequential blocks run the attention stages, then the mid
  // residual, then the MLP stages; parallel blocks overlay the two.
  const placed = new Map<PointName, { col: number; row: number }>();
  let col = 1; // column 0 is resid_pre

  if (parallel) {
    attnStages.forEach((stage, i) =>
      stage.forEach((s) => placed.set(s.point, { col: col + i, row: s.row })),
    );
    mlpStages.forEach((stage, i) =>
      stage.forEach((s) => placed.set(s.point, { col: col + i, row: s.row })),
    );
    col += Math.max(attnStages.length, mlpStages.length);
  } else {
    attnStages.forEach((stage, i) =>
      stage.forEach((s) => placed.set(s.point, { col: col + i, row: s.row })),
    );
    col += attnStages.length;
    placed.set("resid_mid", { col, row: 0 });
    col += 1;
    mlpStages.forEach((stage, i) =>
      stage.forEach((s) => placed.set(s.point, { col: col + i, row: s.row })),
    );
    col += mlpStages.length;
  }
  placed.set("resid_pre", { col: 0, row: 0 });
  placed.set("resid_post", { col, row: 0 });
  const totalCols = col + 1;

  // Materialize whatever *this* side has of the reserved plan; a point it
  // lacks leaves its column empty. Trunk points become one node per stream.
  const nodes: GraphNode[] = [];
  const byPoint = new Map<PointName, GraphNode[]>();
  for (const [point, { col: c, row }] of placed) {
    if (!exists(point, ctx, isMoe)) continue;
    const refusal =
      linear && SOFTMAX_ONLY.has(point) ? LINEAR_LAYER_REFUSAL : null;
    const x = x0 + c * COL_W;
    const made =
      isTrunk(point) && ctx.streams > 1
        ? ctx.spineYs.map((_, i) =>
            makeNode(ctx, point, layer, x, row, refusal, i),
          )
        : [makeNode(ctx, point, layer, x, row, refusal)];
    nodes.push(...made);
    byPoint.set(point, made);
  }

  const edges: GraphEdge[] = [];
  for (const [from, to] of resolveLinks(
    new Set(byPoint.keys()),
    ctx.successors,
  )) {
    const kindOf: GraphEdge["kind"] =
      isTrunk(from) && isTrunk(to) ? "spine" : isTrunk(to) ? "merge" : "branch";
    for (const source of byPoint.get(from) ?? []) {
      for (const target of byPoint.get(to) ?? []) {
        if (
          source.stream !== null &&
          target.stream !== null &&
          source.stream !== target.stream
        ) {
          continue;
        }
        edges.push(edge(source, target, kindOf));
      }
    }
  }

  return {
    nodes,
    edges,
    entries: byPoint.get("resid_pre") ?? [],
    exits: byPoint.get("resid_post") ?? [],
    width: totalCols * COL_W,
  };
}

const TRUNK = new Set(["resid_pre", "resid_mid", "resid_post"]);

function isTrunk(point: PointName): boolean {
  return TRUNK.has(point);
}

/**
 * Which stages get a column. A point survives if either side of the comparison
 * has it, so the side that lacks it leaves a hole rather than shifting
 * everything downstream of it left. With one architecture on screen this is
 * just "the points this architecture has", which is what it always was.
 */
function reserve(ctx: Ctx, layer: number, stages: Stage[]): Stage[] {
  return stages
    .map((stage) =>
      stage.filter((s) =>
        ctx.sides.some((side) =>
          exists(s.point, side, side.moe[layer] ?? false),
        ),
      ),
    )
    .filter((stage) => stage.length > 0);
}

// --- nodes and edges --------------------------------------------------------

function makeNode(
  ctx: Ctx,
  point: PointName,
  layer: number | null,
  x: number,
  row: number,
  refusal: string | null,
  stream: number | null = null,
): GraphNode {
  const spec = pointSpec(point);
  if (!spec) throw new Error(`unknown point: ${point}`);

  return {
    id: nodeId(point, layer, stream),
    point,
    alsoKnownAs: ctx.aliases.get(point) ?? [],
    layer,
    stream,
    role: spec.role as Role,
    width: spec.width,
    fanout: fanoutFor(spec, ctx),
    refusal,
    x,
    y:
      row === 0 && stream !== null
        ? ctx.spineYs[stream]
        : ctx.spineY + (ctx.rows.offsets.get(row) ?? 0),
  };
}

function nodeId(
  point: PointName,
  layer: number | null,
  stream: number | null,
): string {
  const base = layer === null ? point : `${point}.${layer}`;
  return stream === null ? base : `${base}.stream-${stream}`;
}

function edge(
  from: GraphNode,
  to: GraphNode,
  kind: GraphEdge["kind"],
): GraphEdge {
  const flat = Math.abs(from.y - to.y) < 0.5;
  return {
    id: `${from.id}->${to.id}`,
    from: from.id,
    to: to.id,
    kind,
    role: kind === "spine" ? "resid" : to.role,
    path: flat
      ? linePath(from.x, from.y, to.x, to.y)
      : curvePath(from.x, from.y, to.x, to.y),
    dimmed: Boolean(from.refusal || to.refusal),
  };
}

// --- trait gating -----------------------------------------------------------

function exists(point: PointName, side: BuildSide, isMoe: boolean): boolean {
  const spec = pointSpec(point);
  if (!spec) return false;
  const { traits } = side;

  // `moe` is the one trait settled per layer rather than per model: a sparse
  // stack can still keep dense blocks, and those have a neuron basis and no
  // router. Every other trait is a property of the whole architecture.
  const active = (id: TraitId) =>
    id === "moe" ? traits.has(id) && isMoe : traits.has(id);

  for (const required of spec.requires ?? []) {
    if (!active(required)) return false;
  }
  if (mergeTarget(spec, traits)) return false;
  for (const traitId of Object.keys(spec.refusedBy ?? {}) as TraitId[]) {
    if (!active(traitId)) continue;
    // Every `moe` refusal here says the same thing -- that the neuron basis is not a single tensor
    // because the expert bank is fused -- and under `dense_mlp_beside_experts` that reason is simply
    // untrue: the dense MLP is still there, beside the experts, with its neurons intact. So the
    // refusal is waived rather than the points being listed twice with opposite gating.
    if (traitId === "moe" && traits.has("dense_mlp_beside_experts")) continue;
    return false;
  }
  return true;
}

function mergeTarget(spec: PointSpec, traits: Set<TraitId>): PointName | null {
  if (!spec.mergedWith) return null;
  if (spec.distinctUnder)
    return traits.has(spec.distinctUnder) ? null : spec.mergedWith;
  if (spec.mergedUnder)
    return traits.has(spec.mergedUnder) ? spec.mergedWith : null;
  return spec.mergedWith;
}

/** For each surviving point, the names it also stands for under these traits. */
function aliasIndex(traits: Set<TraitId>): Map<PointName, PointName[]> {
  const out = new Map<PointName, PointName[]>();
  for (const spec of ALL_POINTS) {
    const target = mergeTarget(spec, traits);
    if (!target) continue;
    out.set(target, [...(out.get(target) ?? []), spec.name]);
  }
  return out;
}

/**
 * The widest tensor that lands on each row, which is what decides how far apart
 * the rows have to sit. Rows are shared across layers, so a stack with one
 * dense block among many sparse ones still reserves room for the neuron basis.
 */
function rowFanouts(
  dims: Dimensions,
  traits: Set<TraitId>,
): Map<number, number> {
  const out = new Map<number, number>();
  const kv = traits.has("gqa")
    ? Math.min(dims.kvHeads, dims.heads)
    : dims.heads;
  const streams = traits.has("multi_residual_streams") ? dims.streams : 1;

  out.set(-1, Math.max(dims.heads, streams));
  // Row -2 carries `value`, and on a hyper-connection trunk `attn_stream_mix` as well.
  // Both can be absent: latent attention has no value tensor, and every family with
  // one has latent attention so far, which is why the row is claimed by the mix here
  // rather than assumed to exist -- an unclaimed row has no offset, and a glyph placed
  // on it lands on the spine.
  const mix = traits.has("multi_residual_streams") ? streams : 0;
  const valueRow = Math.max(traits.has("mla") ? 0 : kv, mix);
  if (valueRow > 0) out.set(-2, valueRow);
  if (traits.has("qk_norm")) {
    out.set(-3, kv);
    out.set(-4, dims.heads);
  }

  const sparse = traits.has("moe");
  // Whether the neuron basis is drawn anywhere in the stack, which is what reserves the width for it.
  // Not the same question as whether a *dense block* exists: under `dense_mlp_beside_experts` there
  // is no dense block at all and the basis is on every layer, because the MLP survives beside the
  // experts. Getting this wrong collapses the row to one glyph and the neurons land on the spine.
  const hasNeuronBasis =
    !sparse || traits.has("dense_mlp_beside_experts") || dims.layers >= 3;

  out.set(1, Math.max(hasNeuronBasis ? dims.neurons : 1, streams));
  out.set(
    2,
    Math.max(hasNeuronBasis ? dims.neurons : 1, sparse ? dims.experts : 1, mix),
  );
  const row3 = Math.max(
    hasNeuronBasis && traits.has("gated_mlp") ? dims.neurons : 0,
    sparse ? dims.activeExperts : 0,
  );
  if (row3 > 0) out.set(3, row3);

  return out;
}

function fanoutFor(spec: PointSpec, ctx: Ctx): number {
  const { dims, traits } = ctx;
  switch (spec.width) {
    case "heads":
    case "scores":
      return dims.heads;
    case "kv_heads":
      return traits.has("gqa")
        ? Math.min(dims.kvHeads, dims.heads)
        : dims.heads;
    case "neurons":
      return dims.neurons;
    case "routing":
      return spec.name === "router_logits" ? dims.experts : dims.activeExperts;
    case "streams":
      return traits.has("multi_residual_streams") ? dims.streams : 1;
    default:
      return 1;
  }
}

// --- layer sequencing -------------------------------------------------------

function layerKinds(dims: Dimensions, traits: Set<TraitId>): LayerKind[] {
  return Array.from({ length: dims.layers }, (_, i) => {
    if (traits.has("hybrid_linear_attn") && (i + 1) % HYBRID_PERIOD !== 0) {
      return "linear_attention";
    }
    if (
      traits.has("sliding_window") &&
      (i + 1) % (dims.windowRatio + 1) !== 0
    ) {
      return "sliding_attention";
    }
    return "full_attention";
  });
}

/**
 * Which blocks are sparse. Per layer, because a sparse stack can still hold dense blocks.
 *
 * The diagram has no checkpoint to read — the layer count is a slider — so the pattern is a
 * convention rather than a fact. Keeping the first block dense is the convention because several
 * families really do (DeepSeek-V3 keeps three) and the contrast between a dense block and a sparse
 * one is worth a column.
 *
 * `dense_mlp_beside_experts` is the case where that convention has nothing to say: there the experts
 * are added to the MLP rather than replacing it, so every layer is sparse *and* every layer keeps its
 * neuron basis. Drawing a dense first block would invent a contrast the family does not have, and
 * hide the routing points on the one layer a reader looks at first.
 */
function moeLayers(dims: Dimensions, traits: Set<TraitId>): boolean[] {
  if (!traits.has("moe")) return Array(dims.layers).fill(false);
  if (traits.has("dense_mlp_beside_experts"))
    return Array(dims.layers).fill(true);
  return Array.from(
    { length: dims.layers },
    (_, i) => dims.layers < 3 || i > 0,
  );
}

/**
 * Every layer is drawn whole, at every depth the sliders reach.
 *
 * There used to be a visibility pass here that expanded one focused layer and
 * elided the rest into labelled gaps, on the grounds that a deep stack cannot
 * be drawn in full. The premise was about a 128-layer model; `dims.layers` caps
 * at 16, which measures ~13,300px — wide, but the scrubber exists to move
 * around exactly that, and a slider nudge at that depth re-renders in ~175ms.
 * What it cost was the thing the diagram is for: at the default depth of 4 the
 * collapsing never triggered, so the control that chose the expanded layer
 * looked inert, and past 4 every layer but one turned into two featureless
 * boxes. If a much deeper stack is ever wanted, reach for per-layer
 * virtualisation over the horizontal scroll rather than bringing back a mode
 * where most of the drawing is missing.
 */
