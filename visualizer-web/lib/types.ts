/**
 * Shared vocabulary for the whole app.
 *
 * `data/` describes what a transformer is; `lib/` turns that into geometry;
 * `components/` draws it. These types are the only thing all three share.
 */

/** A canonical interp-engine point name, e.g. `resid_post`. */
export type PointName = string;

/** Whether a point exists once per forward pass or once per layer. */
export type Scope = "global" | "layer";

/** The axis a captured tensor is wide along. Drives how many dots we draw. */
export type Width =
  | "d_model"
  | "heads"
  | "kv_heads"
  | "neurons"
  | "routing"
  | "scores"
  | "vocab"
  | "streams";

/** Structural role, which is also the point's colour. */
export type Role = "resid" | "attn" | "mlp" | "route" | "global";

/** Whether interp-engine can serve the point under vLLM, from `points.py`. */
export type VllmSupport = "hooks" | "recompute" | "none";

export type EngineId = "interp-engine" | "transformerlens" | "nnterp";

export type TraitId =
  | "gqa"
  | "mla"
  | "qk_norm"
  | "gated_attn_out"
  | "attn_sinks"
  | "sliding_window"
  | "hybrid_linear_attn"
  | "fused_qkv"
  | "sandwich_norms"
  | "no_pre_attn_norm"
  | "parallel_attn_mlp"
  | "gated_mlp"
  | "moe"
  | "shared_experts"
  | "multi_residual_streams"
  | "logit_softcapping"
  | "residual_multipliers";

export type TraitGroup = "attention" | "norms" | "mlp" | "residual";

export interface PointSpec {
  name: PointName;
  scope: Scope;
  width: Width;
  role: Role;
  /** One line, plain language, shown in the popover. */
  description: string;
  vllm: VllmSupport;
  /**
   * `note` from the same `points.py` row: why the vLLM path is limited the way
   * it is, in the engine's own words — including the `see <other>`
   * indirections, which `vllmReason` in `data/points.ts` follows the way the
   * engine's `reason()` does. Absent where the point is served: the field is
   * here to explain a limit, and nothing displays it for a point that has none.
   */
  vllmNote?: string;
  /** The point only exists when every listed trait is on. */
  requires?: TraitId[];
  /** interp-engine refuses the point when any listed trait is on. */
  refusedBy?: Partial<Record<TraitId, string>>;
  /**
   * The point can collapse onto `mergedWith`, becoming one node carrying both
   * names. `distinctUnder` separates them when the trait is on (the
   * `attn_out` / `attn_out_post` case: one tensor on Llama, two on Gemma);
   * `mergedUnder` joins them when the trait is on (`attn_in` is `resid_pre`
   * on a family with no pre-attention norm).
   */
  mergedWith?: PointName;
  distinctUnder?: TraitId;
  mergedUnder?: TraitId;
}

export interface Trait {
  id: TraitId;
  /** Pill label. Kept short; the popover carries the detail. */
  label: string;
  group: TraitGroup;
  description: string;
  /** Full HF ids. The UI strips the owner prefix for display. */
  exampleModels: string[];
  /** Turning the trait on raises the layer slider's floor to this. */
  minLayers?: number;
  implies?: TraitId[];
  conflicts?: TraitId[];
}

export interface Architecture {
  /** HF architecture class, as in the validator's `sweep_architectures.json`. */
  id: string;
  label: string;
  /**
   * `YYYY-MM`, when the architecture class first shipped rather than when the
   * newest checkpoint using it did. Approximate by design: it orders a
   * timeline, it does not date a release.
   */
  released: string;
  /**
   * Two sentences, opening with the family's own name: what it changed, and
   * what that bought. Prose about the model rather than about the diagram —
   * `note` is where a remark about the drawing belongs.
   */
  significance: string;
  traits: TraitId[];
  exampleModels: string[];
  note?: string;
}

export interface Dimensions {
  layers: number;
  heads: number;
  kvHeads: number;
  neurons: number;
  experts: number;
  activeExperts: number;
  streams: number;
  /** Local attention layers per one global layer, under `sliding_window`. */
  windowRatio: number;
}

/** What kind of sequence mixer a given layer runs. */
export type LayerKind =
  "full_attention" | "sliding_attention" | "linear_attention";

export interface GraphNode {
  id: string;
  point: PointName;
  /**
   * Canonical names this node also stands for, because a trait that would
   * separate them is off. Both appear in the popover.
   */
  alsoKnownAs: PointName[];
  layer: number | null;
  stream: number | null;
  role: Role;
  width: Width;
  /** How many sub-units (heads, neurons, experts) the glyph should render. */
  fanout: number;
  /** Set when interp-engine would refuse this point here; carries the reason. */
  refusal: string | null;
  x: number;
  y: number;
}

export type EdgeKind = "spine" | "branch" | "merge";

export interface GraphEdge {
  id: string;
  from: string;
  to: string;
  kind: EdgeKind;
  role: Role;
  path: string;
  /** Muted, because one of its endpoints is refused on this layer. */
  dimmed: boolean;
}

export interface LayerBand {
  index: number;
  kind: LayerKind;
  isMoe: boolean;
  x: number;
  width: number;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** Every layer, always drawn in full: none is elided or condensed. */
  layers: LayerBand[];
  width: number;
  height: number;
  /** Y of the residual spine, or of each stream under hyper-connections. */
  spineYs: number[];
}
