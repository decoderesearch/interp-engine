"use client";

/**
 * Everything that is not the diagram.
 *
 * The controls used to ring the canvas — traits along the top, sliders bottom
 * left, legend bottom right — which cost the diagram its corners and left no
 * room to grow. One column on the right keeps the drawing rectangular and gives
 * each control a full width to lay itself out in.
 *
 * The same component fills the sheet on narrow screens, so there is one list of
 * controls rather than a desktop set and a mobile set that drift apart.
 */

import { X } from "lucide-react";
import type { ReactNode } from "react";

import { DimensionSliders } from "@/components/controls/DimensionSliders";
import { NamingToggle } from "@/components/controls/NamingToggle";
import { SearchField } from "@/components/controls/SearchField";
import { TraitBar, TraitList } from "@/components/controls/TraitBar";
import { Legend } from "@/components/diagram/Legend";
import { displayModel } from "@/data/architectures";
import type { VisualizerState } from "@/lib/state";

interface Props {
  state: VisualizerState;
  note?: string;
  examples: string[];
  /** Big tap targets and no hover cards, for the sheet. */
  touch?: boolean;
}

export function Sidebar({ state, note, examples, touch = false }: Props) {
  return (
    <div className="flex flex-col gap-y-5 p-4">
      <Section title="Legend">
        <Legend comparing={state.comparing} />
      </Section>

      <Section title="Dimensions">
        <DimensionSliders state={state} />
      </Section>

      <Section title="Traits">
        {state.comparing ? (
          <CompareNotice onExit={() => state.setCompareId(null)} />
        ) : touch ? (
          <TraitList state={state} />
        ) : (
          <TraitBar state={state} />
        )}
      </Section>

      {/* Reads as the answer to the section above it: these toggles, together,
          describe these models. Nothing to answer with while comparing, where
          each pane names its own architecture. */}
      {!state.comparing && (
        <Section title="Models matching these traits">
          {examples.length > 0 ? (
            <ul className="flex flex-col gap-y-0.5">
              {examples.map((model) => (
                <li
                  key={model}
                  className="font-mono text-[10px] text-slate-600"
                >
                  {displayModel(model)}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[10px] leading-relaxed text-slate-400">
              No named family has exactly this combination.
            </p>
          )}
          {note && (
            <p className="text-[10px] leading-relaxed text-slate-400">{note}</p>
          )}
        </Section>
      )}

      <Section title="Search">
        <SearchField state={state} />
      </Section>

      <Section title="Names">
        <NamingToggle state={state} />
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    // The trait groups carry small-caps labels of their own, so the section
    // headings need the rule to stay a level above them.
    <section className="flex flex-col gap-y-2 border-t border-slate-200 pt-4 first:border-0 first:pt-0">
      <h2 className="text-[10px] font-semibold tracking-wide text-slate-600 uppercase">
        {title}
      </h2>
      {children}
    </section>
  );
}

function CompareNotice({ onExit }: { onExit: () => void }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <p className="text-[11px] leading-relaxed text-slate-500">
        Hidden while comparing. Both diagrams show a named architecture, and
        hand-editing the traits of one would leave the comparison describing
        something with no name.
      </p>
      <button
        type="button"
        onClick={onExit}
        className="mt-2 flex cursor-pointer items-center gap-x-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-[10px] font-medium text-slate-600 transition-colors hover:bg-slate-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40"
      >
        <X className="h-3 w-3" />
        Back to single mode
      </button>
    </div>
  );
}
