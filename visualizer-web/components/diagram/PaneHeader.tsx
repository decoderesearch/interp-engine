"use client";

/**
 * What sits above a diagram: the architecture it is showing, as a picker and a
 * timeline, and — while two are being compared — how that architecture differs
 * from the one in the other pane.
 *
 * The same band in both modes, so the picker is where the thing it names is
 * rather than in the header in one mode and over the canvas in the other.
 *
 * Three ways to name the same choice, and the band holds all three because they
 * answer different questions: the list answers "which one", the timeline answers
 * "what came after what", and the search field — which stands in for the list
 * while it is open — answers "I already know the name". The circled i beside them
 * says why the one on screen mattered.
 *
 * The difference is stated from this pane's point of view — "has QK-norm", "no
 * MoE" — so each pane reads as a description of itself rather than as half of
 * a comparison you have to assemble in your head.
 *
 * It is a band above the canvas rather than an overlay on it. Floating it saved
 * height but put the pills on top of the top row of hook points, which is where
 * the embeddings and the first attention block live.
 *
 * Pointing at a pill narrows the red rings below to the points that difference
 * is responsible for, which is the only way to read a hundred-ring diff: not
 * "where do these disagree" but "where does *this* disagreement land".
 */

import { Search } from "lucide-react";
import { HoverCard } from "radix-ui";
import { useEffect, useRef, useState } from "react";

import { ArchitectureInfo } from "@/components/controls/ArchitectureInfo";
import { ArchitecturePicker } from "@/components/controls/ArchitecturePicker";
import { ArchitectureSearch } from "@/components/controls/ArchitectureSearch";
import { ReleaseSlider } from "@/components/controls/ReleaseSlider";
import { TraitCard } from "@/components/controls/TraitBar";
import { architecture, formatRelease } from "@/data/architectures";
import type { TraitDelta } from "@/lib/diff";
import type { TraitId } from "@/lib/types";

/** Everything the band shows only while a second diagram is on screen. */
interface Comparison {
  selfLabel: string;
  otherLabel: string;
  traits: Set<TraitId>;
  deltas: TraitDelta[];
  /** Narrows the red rings while a difference is being pointed at. */
  onFocus: (id: TraitId | null) => void;
}

interface Props {
  value: string | null;
  placeholder: string;
  onChange: (id: string | null) => void;
  /** The other pane's architecture: left out of this picker, and named below. */
  exclude?: string | null;
  compare?: Comparison;
}

export function PaneHeader({
  value,
  placeholder,
  onChange,
  exclude = null,
  compare,
}: Props) {
  const selected = value === null ? null : architecture(value);
  const [searching, setSearching] = useState(false);
  const searchScope = useRef<HTMLDivElement>(null);

  // The toggle is inside the scope on purpose. Closing on the field's own blur
  // instead would fire before the toggle's click, which would then read a closed
  // search and open it straight back up — and on touch it would take the list
  // down before a tap on a row could land.
  useEffect(() => {
    if (!searching) return;
    const onDown = (event: PointerEvent) => {
      if (searchScope.current?.contains(event.target as Node)) return;
      setSearching(false);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [searching]);

  return (
    <div
      // Tight against the canvas on purpose. The graph carries `PAD_Y` of its own
      // above the first glyph — that is where the layer band's outline and the
      // `L0` labels sit — so padding here adds to that rather than being the whole
      // of the separation, and in compare mode it is paid twice.
      //
      // A column below `sm` while comparing, so the diff pills get the full width
      // under the picker instead of a third of it beside one. In single mode there
      // is nothing to move: the prose that would sit there is already hidden until
      // `lg`.
      className={`flex shrink-0 px-3 pt-2.5 pb-1 sm:pb-2 ${
        compare
          ? "flex-col gap-y-1.5 sm:flex-row sm:items-start sm:gap-x-3 sm:gap-y-0"
          : "items-start gap-x-3"
      }`}
    >
      {/* Where the timeline goes. Beside the picker at every width in single mode,
          and on a phone it takes whatever the row has left. Comparing, it is not
          monotonic in width, because what it competes with is not either: beside
          the picker on a phone, where the pills have moved out from under it and
          the picker has given back its caption width — then *under* it from `sm`,
          where the pills are back alongside and 228px will not fit next to them —
          then beside it again at `xl`, where everything fits. */}
      {/* Single mode takes the whole row until `lg`, where the prose appears beside
          it and it goes back to its own width. Sized rather than left to content
          because the timeline inside it is flexible: content-sized, the group would
          ask for the timeline's full 228px next to the picker and overflow a phone.
          Comparing, the band is a column at that width and stretches it for us. */}
      <div
        className={`flex items-center gap-x-2 sm:gap-x-3 ${
          compare
            ? "shrink-0 sm:flex-col sm:items-start sm:gap-y-1.5 xl:flex-row xl:items-center"
            : "min-w-0 flex-1 lg:flex-none"
        }`}
      >
        <div
          ref={searchScope}
          className="flex h-9 shrink-0 items-center gap-x-2 sm:h-11"
        >
          <button
            type="button"
            // Must not take focus off the field: see the effect above.
            onPointerDown={(event) => event.preventDefault()}
            onClick={() => setSearching((was) => !was)}
            aria-label="Search architectures"
            aria-pressed={searching}
            className={`flex h-9 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md border shadow-sm transition-colors sm:h-11 sm:w-9 ${
              searching
                ? "border-sky-300 bg-sky-50 text-sky-600"
                : "border-slate-300 bg-white text-slate-400 hover:bg-slate-50 hover:text-slate-600"
            }`}
          >
            <Search className="h-4 w-4" />
          </button>

          {searching ? (
            <ArchitectureSearch
              value={value}
              onChange={onChange}
              onClose={() => setSearching(false)}
              exclude={exclude}
              className="w-[132px] sm:w-[208px]"
            />
          ) : (
            <ArchitecturePicker
              label="Architecture"
              caption={
                selected ? `Released ${formatRelease(selected)}` : undefined
              }
              value={value}
              placeholder={placeholder}
              onChange={onChange}
              exclude={exclude}
              className="shadow-sm"
            />
          )}

          {/* Outside the picker, not inside its trigger: nested there, a tap
              opened the list instead of the card.

              Gone at `lg` in single mode, which is exactly where the same two
              sentences are printed beside the timeline — an icon whose whole
              job is to open a card repeating the paragraph next to it. It stays
              wherever that prose does not fit: below `lg`, and at every width in
              compare mode, where the diff pills own that space. */}
          {selected && (
            <ArchitectureInfo
              arch={selected}
              className={compare ? undefined : "lg:hidden"}
            />
          )}
        </div>

        <ReleaseSlider
          value={value}
          onChange={onChange}
          exclude={exclude}
          // Takes what is left of the row when it is beside the picker on a phone,
          // which is about 150px — enough to drag, and the thumb's own label names
          // the family, so the track is a position rather than the only readout.
          className="min-w-0 flex-1 sm:w-[228px] sm:flex-none"
        />
      </div>

      {compare ? (
        <Diff {...compare} />
      ) : (
        selected && (
          <div className="hidden min-w-0 flex-1 lg:block">
            {/* Not clamped: an ellipsis here hid the second sentence, which is the
                half that says what the change bought. The width cap is what keeps
                the band from growing instead. */}
            <p className="max-w-[760px] pt-0.5 text-[11px] leading-relaxed text-slate-500">
              {selected.significance}
            </p>
          </div>
        )
      )}
    </div>
  );
}

function Diff({ selfLabel, otherLabel, traits, deltas, onFocus }: Comparison) {
  return (
    <>
      <div className="flex min-w-0 flex-1 flex-col items-center gap-y-1">
        <div className="max-w-full truncate text-[9px] font-medium tracking-wide text-slate-400 uppercase">
          Architecture diff vs {otherLabel}
        </div>
        {/* Six pills stacked would cover half a phone-sized pane, so below `sm`
            they stay on one line and scroll instead.

            `overflow-y-hidden` because a box that scrolls on one axis and is
            `visible` on the other does not stay visible — CSS promotes it to
            `auto`, so this scrolled a pill's own height vertically as well as
            sideways. And no scrollbar rather than a thin one: a thin one is drawn
            inside a box exactly as tall as one pill, which is what left nothing
            for the pill. The cut-off pill at the edge is the affordance. */}
        <div className="no-scrollbar flex max-w-full min-w-0 flex-nowrap gap-1 overflow-x-auto overflow-y-hidden sm:max-w-[620px] sm:flex-wrap sm:justify-center sm:overflow-visible">
          {deltas.length === 0 ? (
            <span className="rounded-full border border-slate-200 bg-white px-2 py-0.5 text-[10px] text-slate-400">
              Same traits — only the derivations differ
            </span>
          ) : (
            deltas.map(({ trait, has }) => (
              <HoverCard.Root
                key={trait.id}
                openDelay={120}
                closeDelay={60}
                onOpenChange={(open) => onFocus(open ? trait.id : null)}
              >
                <HoverCard.Trigger asChild>
                  <span
                    tabIndex={0}
                    className={`shrink-0 cursor-help rounded-full border px-2 py-0.5 text-[10px] font-medium whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 ${
                      has
                        ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                        : "border-slate-300 bg-slate-100 text-slate-500"
                    }`}
                  >
                    <span className="opacity-60">{has ? "has" : "no"}</span>{" "}
                    {trait.label}
                  </span>
                </HoverCard.Trigger>
                <TraitCard
                  trait={trait}
                  traits={traits}
                  lead={
                    <div className="mb-2 space-y-0.5 border-b border-slate-100 pb-2">
                      <Sides
                        model={selfLabel}
                        label={trait.label}
                        has={has}
                        emphasis
                      />
                      <Sides
                        model={otherLabel}
                        label={trait.label}
                        has={!has}
                      />
                    </div>
                  }
                />
              </HoverCard.Root>
            ))
          )}
        </div>
      </div>

      {/* Balances the picker so the diff list is centred on the pane, not on
          whatever is left over beside it. Narrow screens need the width more
          than they need the symmetry. */}
      <div className="hidden w-[208px] shrink-0 sm:block" aria-hidden />
    </>
  );
}

function Sides({
  model,
  label,
  has,
  emphasis = false,
}: {
  model: string;
  label: string;
  has: boolean;
  emphasis?: boolean;
}) {
  return (
    <div className="text-[11px] leading-snug">
      <span
        className={`font-mono ${emphasis ? "font-semibold text-slate-700" : "text-slate-500"}`}
      >
        {model}
      </span>{" "}
      <span className={has ? "text-emerald-700" : "text-slate-400"}>
        {has ? "has" : "does not have"}
      </span>{" "}
      <span className="text-slate-500">{label}</span>
    </div>
  );
}
