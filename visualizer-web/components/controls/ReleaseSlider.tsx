"use client";

/**
 * The same choice as the architecture dropdown, laid out as a timeline.
 *
 * The list answers "which family", this answers "what came after what", and
 * the two are one control: picking from either moves both.
 *
 * The stops are evenly spaced rather than positioned by date. A true time axis
 * is the more honest picture and the worse control: three quarters of these
 * families shipped in an eighteen-month window, so their stops landed a few
 * pixels apart while GPT-2 sat alone at the far left with four empty years
 * beside it. Even spacing makes every family the same size target and the
 * order — which is what the timeline is for — still reads left to right. The
 * dates are on the label and in the picker's caption, where they can be read
 * exactly rather than estimated from a position.
 */

import { Slider } from "radix-ui";
import { useState } from "react";

import { BY_RELEASE, formatRelease, releaseYear } from "@/data/architectures";
import { cn } from "@/lib/utils";

const FIRST_YEAR = releaseYear(BY_RELEASE[0]);
const LAST_YEAR = releaseYear(BY_RELEASE[BY_RELEASE.length - 1]);

/** The thumb's width, which the label above it has to be positioned from. */
const THUMB = 8;

/**
 * A stop's position along the track, inset by half a thumb at each end.
 *
 * Radix keeps the thumb inside the track rather than letting it hang off the
 * ends, so its centre travels `track - thumb`, not `track`. The label placed on
 * the plain percentage drifts from the thumb by up to half a thumb width.
 */
function offset(index: number, count: number): string {
  const fraction = count < 2 ? 0 : index / (count - 1);
  return `calc(${THUMB / 2}px + (100% - ${THUMB}px) * ${fraction})`;
}

interface Props {
  /** Architecture id, or null when the traits match no named family. */
  value: string | null;
  onChange: (id: string) => void;
  /** The other pane's choice, which this one may not land on. */
  exclude?: string | null;
  className?: string;
}

export function ReleaseSlider({
  value,
  onChange,
  exclude = null,
  className,
}: Props) {
  const options = BY_RELEASE.filter((arch) => arch.id !== exclude);
  const last = options.length - 1;
  const current = options.find((arch) => arch.id === value) ?? null;

  // Hand-editing traits unselects the architecture. Dropping the thumb back to
  // the far left for that would read as a change of model, so it stays where
  // it was.
  const [held, setHeld] = useState(() =>
    current ? options.indexOf(current) : 0,
  );
  if (current && options.indexOf(current) !== held) {
    setHeld(options.indexOf(current));
  }

  const index = current ? options.indexOf(current) : held;
  const at = last === 0 ? 0 : (index / last) * 100;

  return (
    <div
      // Shorter below `sm`, where the years at the ends are not drawn: with two
      // rows in it instead of three it matches the picker beside it.
      className={cn("flex h-9 flex-col justify-center sm:h-11", className)}
    >
      <div className="relative h-[13px]">
        <span
          className="absolute top-0 text-[9px] whitespace-nowrap"
          // At 0% this reads as left-aligned and at 100% as right-aligned,
          // which keeps the label on the track without unpinning it from the
          // thumb in between.
          style={{
            left: offset(index, options.length),
            transform: `translateX(-${at}%)`,
          }}
        >
          <span
            className={
              current ? "font-semibold text-slate-600" : "text-slate-400"
            }
          >
            {current?.label ?? "Custom"}
          </span>
          {current && (
            <span className="ml-1 text-slate-400">
              {formatRelease(current)}
            </span>
          )}
        </span>
      </div>

      <Slider.Root
        // As tall as the thumb, which stands taller than the track: sized to the
        // track instead, the thumb would hang a pixel out of its own control.
        className="relative flex h-[18px] touch-none items-center select-none"
        value={[index]}
        min={0}
        max={Math.max(last, 0)}
        step={1}
        onValueChange={([next]) => {
          const pick = options[next];
          if (pick && pick.id !== value) onChange(pick.id);
        }}
      >
        {/* No tick per stop. There is one every ~12px at full width and every
            ~7px on a phone, which at that pitch reads as a hatched track rather
            than as countable stops — and the thing they were there to say, that
            the control is discrete, the thumb says by snapping. */}
        <Slider.Track className="relative h-[5px] grow rounded-full border border-slate-300 bg-white">
          <Slider.Range className="absolute h-full rounded-full bg-sky-100" />
        </Slider.Track>
        <Slider.Thumb
          aria-label="Release timeline"
          // Upright and standing well clear of the track, like a scrubber's
          // handle: it reads as something to drag along a line, where a dot the
          // height of the track read as one more stop on it. Fully rounded, so at
          // this width the two ends are semicircles and the thing is a stadium.
          className="block h-[18px] w-2 cursor-grab rounded-full border-2 border-sky-600 bg-white shadow transition-colors hover:bg-sky-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/50 active:cursor-grabbing"
        />
      </Slider.Root>

      {/* The ends of the timeline, not of `options`: excluding the other pane's
          choice must not appear to move the range.

          Gone below `sm`. They are the least of what the control says — the thumb
          carries the family and its exact date — and on a phone the track is about
          150px, where two more numbers under it read as clutter. */}
      <div className="hidden justify-between text-[8px] tabular-nums text-slate-400 sm:flex">
        <span>{FIRST_YEAR}</span>
        <span>{LAST_YEAR}</span>
      </div>
    </div>
  );
}
