/**
 * The adjustable dimensions, and the traits that make each one meaningful.
 *
 * These are schematic counts, not a checkpoint's real config: the diagram caps
 * them low enough that every head, neuron and expert can be drawn as its own
 * mark, which is the point of showing them at all.
 */

import type { Dimensions, TraitId } from "@/lib/types";

export interface DimensionSpec {
  key: keyof Dimensions;
  label: string;
  min: number;
  max: number;
  step: number;
  /** Only shown when this trait is on. */
  requires?: TraitId;
  /** Cannot exceed the current value of this other dimension. */
  boundedBy?: keyof Dimensions;
  /** Rendered next to the value in the slider thumb. */
  format?: (value: number) => string;
  help: string;
}

export const DIMENSIONS: DimensionSpec[] = [
  {
    key: "layers",
    label: "Layers",
    min: 2,
    max: 16,
    step: 1,
    help: "Blocks in the stack. Every one is drawn in full; use the strip under the diagram to move along it.",
  },
  {
    key: "heads",
    label: "Attn heads",
    min: 1,
    max: 16,
    step: 1,
    help: "Query heads. Sets the width of z, the attention pattern and the QK-norm points.",
  },
  {
    key: "kvHeads",
    label: "KV heads",
    min: 1,
    max: 16,
    step: 1,
    requires: "gqa",
    boundedBy: "heads",
    help: "Key/value heads shared across the query heads. Sets the width of value and the k-norm points.",
  },
  {
    key: "neurons",
    label: "MLP neurons",
    min: 3,
    max: 64,
    step: 1,
    help: "Width of the neuron basis: mlp_pre, mlp_pre_linear and mlp_act.",
  },
  {
    key: "experts",
    label: "Experts",
    min: 2,
    max: 32,
    step: 1,
    requires: "moe",
    help: "Routed experts per layer. Sets the width of router_logits.",
  },
  {
    key: "activeExperts",
    label: "Top-k",
    min: 1,
    max: 32,
    step: 1,
    requires: "moe",
    boundedBy: "experts",
    help: "Experts each token is routed to. Sets the width of expert_weights and expert_indices.",
  },
  {
    key: "streams",
    label: "Streams",
    min: 2,
    max: 4,
    step: 1,
    requires: "multi_residual_streams",
    help: "Parallel residual streams on the trunk. Every resid point needs a stream coordinate to name one.",
  },
  {
    key: "windowRatio",
    label: "Local : global",
    min: 1,
    max: 8,
    step: 1,
    requires: "sliding_window",
    format: (v) => `${v}:1`,
    help: "Sliding-window layers between each full-context layer. Gemma 3 runs 5:1.",
  },
];

export const DEFAULT_DIMENSIONS: Dimensions = {
  layers: 4,
  heads: 4,
  kvHeads: 2,
  // Six rather than eight: the neuron stack is the tallest glyph in the drawing,
  // and every row below the spine is placed to clear it. Two fewer marks still
  // reads as "one mark per neuron" — which is the whole job of the default — and
  // takes ~12px off the height of every MLP column.
  neurons: 6,
  experts: 8,
  activeExperts: 2,
  streams: 2,
  windowRatio: 5,
};
