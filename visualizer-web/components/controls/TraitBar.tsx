"use client";

import { HoverCard } from "radix-ui";
import { type ReactNode, useMemo } from "react";

import { displayModel } from "@/data/architectures";
import { TRAITS, TRAIT_GROUPS } from "@/data/traits";
import { pointEffects } from "@/lib/diff";
import type { VisualizerState } from "@/lib/state";
import type { Trait, TraitId } from "@/lib/types";

interface Props {
  state: VisualizerState;
}

/** The pills, grouped and stacked for the sidebar column. */
export function TraitBar({ state }: Props) {
  const { traits, toggleTrait } = state;

  return (
    <div className="flex flex-col gap-y-2.5">
      {TRAIT_GROUPS.map((group) => {
        const members = TRAITS.filter((t) => t.group === group.id);
        if (!members.length) return null;
        return (
          <div key={group.id} className="flex flex-col gap-y-1">
            <div className="text-[9px] font-medium tracking-wide text-slate-400 uppercase">
              {group.label}
            </div>
            <div className="flex flex-wrap gap-1">
              {members.map((trait) => (
                <TraitPill
                  key={trait.id}
                  trait={trait}
                  traits={traits}
                  active={traits.has(trait.id)}
                  onToggle={() => toggleTrait(trait.id)}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TraitPill({
  trait,
  traits,
  active,
  onToggle,
}: {
  trait: Trait;
  traits: Set<TraitId>;
  active: boolean;
  onToggle: () => void;
}) {
  return (
    <HoverCard.Root openDelay={120} closeDelay={60}>
      <HoverCard.Trigger asChild>
        <button
          type="button"
          onClick={onToggle}
          aria-pressed={active}
          className={`shrink-0 cursor-pointer rounded-full border px-2 py-1 text-[10px] font-medium whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 ${
            active
              ? "border-sky-600 bg-sky-100 text-sky-700"
              : "border-slate-200 bg-white text-slate-500 hover:bg-slate-100 hover:text-slate-700"
          }`}
        >
          {trait.label}
        </button>
      </HoverCard.Trigger>
      <TraitCard trait={trait} traits={traits} />
    </HoverCard.Root>
  );
}

/**
 * The hover card body, shared by the trait pills and by the per-pane diff pills
 * in compare mode. `lead` is where the caller says why this trait is being
 * pointed at — the comparison pills open with which side has it.
 *
 * Expects to be inside a `HoverCard.Root` whose trigger the caller owns.
 */
export function TraitCard({
  trait,
  traits,
  lead,
}: {
  trait: Trait;
  traits: Set<TraitId>;
  lead?: ReactNode;
}) {
  return (
    <HoverCard.Portal>
      <HoverCard.Content
        sideOffset={8}
        collisionPadding={12}
        className="animate-in fade-in-0 zoom-in-95 z-50 w-[292px] rounded-md border border-slate-200 bg-white p-3 shadow-lg"
      >
        {lead}
        <div className="text-xs font-semibold text-slate-700">
          {trait.label}
        </div>
        <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
          {trait.description}
        </p>
        <PointEffects id={trait.id} traits={traits} />
        <div className="mt-2.5 border-t border-slate-100 pt-2">
          <div className="text-[9px] font-medium tracking-wide text-slate-400 uppercase">
            Example models
          </div>
          <ul className="mt-1 space-y-0.5">
            {trait.exampleModels.map((model) => (
              <li key={model} className="font-mono text-[10px] text-slate-600">
                {displayModel(model)}
              </li>
            ))}
          </ul>
        </div>
        {trait.minLayers ? (
          <div className="mt-2 text-[10px] text-amber-700">
            Needs at least {trait.minLayers} layers to show its pattern.
          </div>
        ) : null}
        <HoverCard.Arrow className="fill-white" />
      </HoverCard.Content>
    </HoverCard.Portal>
  );
}

function PointEffects({ id, traits }: { id: TraitId; traits: Set<TraitId> }) {
  const { adds, removes, splits, reworks } = useMemo(
    () => pointEffects(id, traits),
    [id, traits],
  );
  if (!adds.length && !removes.length && !splits.length && !reworks.length)
    return null;
  return (
    <div className="mt-2 space-y-0.5">
      {adds.length > 0 && (
        <EffectRow verb="Adds" points={adds} className="text-emerald-700" />
      )}
      {splits.length > 0 && (
        <EffectRow verb="Splits" points={splits} className="text-sky-700" />
      )}
      {removes.length > 0 && (
        <EffectRow verb="Removes" points={removes} className="text-amber-700" />
      )}
      {reworks.length > 0 && (
        <EffectRow
          verb="Rewrites"
          points={reworks}
          className="text-violet-700"
        />
      )}
    </div>
  );
}

function EffectRow({
  verb,
  points,
  className,
}: {
  verb: string;
  points: string[];
  className: string;
}) {
  return (
    <div className="flex gap-x-1.5 text-[10px] leading-relaxed">
      <span className={`shrink-0 font-medium ${className}`}>{verb}</span>
      <span className="font-mono text-slate-500">{points.join(", ")}</span>
    </div>
  );
}

/** The same traits as scrollable rows, for the mobile sheet where hover is out. */
export function TraitList({ state }: Props) {
  const { traits, toggleTrait } = state;
  return (
    <div className="flex flex-col gap-y-4">
      {TRAIT_GROUPS.map((group) => (
        <div key={group.id}>
          <div className="mb-1.5 text-[10px] font-medium tracking-wide text-slate-400 uppercase">
            {group.label}
          </div>
          <div className="flex flex-col gap-y-1">
            {TRAITS.filter((t) => t.group === group.id).map((trait) => {
              const active = traits.has(trait.id);
              return (
                <button
                  key={trait.id}
                  type="button"
                  onClick={() => toggleTrait(trait.id)}
                  aria-pressed={active}
                  className={`rounded-md border p-2.5 text-left transition-colors ${
                    active
                      ? "border-sky-600 bg-sky-50"
                      : "border-slate-200 bg-white hover:bg-slate-50"
                  }`}
                >
                  <div
                    className={`text-xs font-semibold ${active ? "text-sky-700" : "text-slate-600"}`}
                  >
                    {trait.label}
                  </div>
                  <p className="mt-0.5 text-[11px] leading-relaxed text-slate-500">
                    {trait.description}
                  </p>
                  <PointEffects id={trait.id} traits={traits} />
                  <div className="mt-1 font-mono text-[10px] text-slate-400">
                    {trait.exampleModels.map(displayModel).join(" · ")}
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
