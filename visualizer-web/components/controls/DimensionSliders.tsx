"use client";

import { AnimatePresence, motion } from "motion/react";

import { DIMENSIONS } from "@/data/dimensions";
import { minLayersFor } from "@/data/traits";
import { ValueSlider } from "@/components/controls/ValueSlider";
import type { VisualizerState } from "@/lib/state";

interface Props {
  state: VisualizerState;
}

/**
 * Only the dimensions that mean something under the current traits are shown.
 * An expert-count slider on a dense model would be a control with no effect,
 * which is worse than a control that is not there. While comparing, a slider
 * that either architecture needs is shown, since both are drawn at one size.
 */
export function DimensionSliders({ state }: Props) {
  const { dims, activeTraits, setDimension } = state;
  const layerFloor = minLayersFor(activeTraits, dims.windowRatio);

  const visible = DIMENSIONS.filter(
    (spec) => !spec.requires || activeTraits.has(spec.requires),
  );

  return (
    <div className="flex flex-col gap-y-1.5">
      <AnimatePresence initial={false}>
        {visible.map((spec) => {
          const min = spec.key === "layers" ? layerFloor : spec.min;
          const max = spec.boundedBy
            ? Math.min(spec.max, dims[spec.boundedBy])
            : spec.max;
          return (
            <motion.div
              key={spec.key}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
              className="overflow-hidden"
            >
              <ValueSlider
                label={spec.label}
                value={dims[spec.key]}
                min={min}
                max={Math.max(max, min)}
                step={spec.step}
                format={spec.format}
                onChange={(value) => setDimension(spec.key, value)}
              />
            </motion.div>
          );
        })}
      </AnimatePresence>
      {layerFloor > 2 && (
        <div className="pl-[102px] text-[9px] leading-tight text-slate-400">
          The active traits need at least {layerFloor} layers to show their
          pattern.
        </div>
      )}
    </div>
  );
}
