"use client";

import { ENGINES } from "@/data/engines";
import type { VisualizerState } from "@/lib/state";

interface Props {
  state: VisualizerState;
}

/**
 * Which stack's names to print on the diagram, and which vocabulary the hover
 * cards speak: under interp-engine each one also carries the two calls that
 * read the point, which are this engine's API and belong under no other name.
 */
export function NamingToggle({ state }: Props) {
  const { engineId, setEngineId } = state;

  return (
    <div
      role="radiogroup"
      aria-label="Naming scheme"
      className="flex w-full rounded-md border border-slate-300 bg-white p-0.5"
    >
      {ENGINES.map((engine) => {
        const active = engine.id === engineId;
        return (
          <button
            key={engine.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setEngineId(engine.id)}
            title={engine.label}
            className={`flex-1 cursor-pointer rounded-sm px-2.5 py-1.5 font-mono text-[10px] font-medium whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 ${
              active
                ? "bg-sky-100 text-sky-700"
                : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
            }`}
          >
            {engine.label}
          </button>
        );
      })}
    </div>
  );
}
