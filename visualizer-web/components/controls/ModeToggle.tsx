"use client";

/**
 * The sizer, one diagram, or two. A radio group rather than buttons that swap
 * for other buttons: the mode is visible whichever one you are in, and leaving
 * a mode is the same control that entered it.
 *
 * All three are always shown, so the header's left edge keeps its width and the
 * wordmark beside it does not shift when the mode changes.
 *
 * Two shapes, because it appears in two places that mean different things. In
 * the header it is a **folder tab**: square along the bottom, seated on the
 * header's lower edge, with its own bottom border painted the body's colour so
 * that it cuts the sky divider and the group reads as opening into the pane
 * below rather than sitting on top of it. On a phone the header cannot hold it
 * and it becomes a floating pill in the bottom centre, where a tab attached to
 * nothing would be the wrong metaphor — so that one keeps its rounding and its
 * shadow.
 *
 * The three tracks are equal width rather than text width. `1fr` in a
 * shrink-to-fit grid sizes every column to the widest of them, so "Compare"
 * gets the same tab as "Model Viz + Points" without a hard-coded number that
 * would have to be re-guessed each time a label changes.
 */

import {
  CircleQuestionMark,
  Columns2,
  Cpu,
  SquareActivity,
} from "lucide-react";
import { Popover } from "radix-ui";
import { useState, type ReactNode } from "react";

export type Mode = "single" | "compare" | "sizer";

interface Entry {
  mode: Mode;
  icon: ReactNode;
  label: string;
  blurb: string;
}

const MODES: Entry[] = [
  {
    mode: "sizer",
    icon: <Cpu className="h-3.5 w-3.5" />,
    label: "GPU Sizer",
    blurb:
      "What GPU do you need to run interp-engine on a model? Get exact GPU requirements and configs to optimize speed, context length, parallelism and VRAM. Returns exact code examples.",
  },
  {
    mode: "single",
    icon: <SquareActivity className="h-3.5 w-3.5" />,
    label: "Model Viz + Points",
    blurb:
      "A ‘cheat sheet’ for every ‘point’/Address on model architectures from GPT-2 to Gemma 4, visualized on an easy-to-understand map.",
  },
  {
    mode: "compare",
    // Turned on its side, which is also the way the mode splits: the second
    // diagram appears below the first, not beside it.
    icon: <Columns2 className="h-3.5 w-3.5 rotate-90" />,
    label: "Compare",
    blurb:
      "Like Model Viz, but shows two architectures simultaneously and highlights differences between them.",
  },
];

interface Props {
  mode: Mode;
  onMode: (mode: Mode) => void;
  /** The phone's bottom-centre pill. Header tabs are the default. */
  floating?: boolean;
}

export function ModeToggle({ mode, onMode, floating = false }: Props) {
  return (
    <div
      role="radiogroup"
      aria-label="View mode"
      className={
        floating
          ? "grid shrink-0 grid-cols-3 gap-x-1 rounded-md border border-sky-700 bg-white p-1"
          : // The band under the pill is the gap the active tab opens onto the
            // pane through — without it the tab reads as a button jammed against
            // the content, and much more than this and it stops reading as
            // attached to it at all.
            "grid shrink-0 grid-cols-3 gap-x-1.5 rounded-t-md border border-sky-700 border-b-slate-50 bg-slate-50 p-1.5"
      }
    >
      {MODES.map((entry) => (
        <Option
          key={entry.mode}
          entry={entry}
          active={mode === entry.mode}
          onClick={() => onMode(entry.mode)}
        />
      ))}
    </div>
  );
}

function Option({
  entry,
  active,
  onClick,
}: {
  entry: Entry;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      title={entry.label}
      className={`flex cursor-pointer items-center justify-center gap-x-1.5 rounded-md px-2.5 py-2 text-[13px] font-medium whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 lg:px-3.5 ${
        active
          ? "bg-sky-700 text-white"
          : "text-slate-500 hover:bg-slate-200/70 hover:text-slate-700"
      }`}
    >
      <span className={active ? "text-white" : "text-slate-400"}>
        {entry.icon}
      </span>
      {/* Icon-only until `lg`, where the header also carries the wordmark, the
          Controls trigger, Riz and the icon cluster. Two labels fit from `md`
          and three do not, so the whole group waits for the wider breakpoint
          rather than showing two names and one glyph. `sr-only` rather than
          `hidden`, because the text is this radio's accessible name. */}
      <span className="sr-only lg:not-sr-only">{entry.label}</span>
      <ModeInfo entry={entry} active={active} />
    </button>
  );
}

/**
 * The circled question mark on a tab, and what the mode is for.
 *
 * A `Popover` driven by hand rather than a `HoverCard`, matching the circled i
 * beside the architecture picker: a HoverCard never opens on touch, and mouse
 * and pen are told apart by `pointerType` rather than by a media query, since a
 * hybrid laptop is both and a viewport width does not say which is in your hand.
 *
 * A `span` with a button's role, because the tab it sits in is already a button
 * and one cannot be nested in another. It also stops its own click: reaching for
 * the description should not change the mode out from under you.
 *
 * Held back to `lg` with the labels. Below that a tab is a single glyph, and a
 * second glyph beside it would only raise the question of which one is the mode.
 */
function ModeInfo({ entry, active }: { entry: Entry; active: boolean }) {
  const [open, setOpen] = useState(false);

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Anchor asChild>
        <span
          role="button"
          tabIndex={0}
          aria-label={`About ${entry.label}`}
          onPointerEnter={(event) => {
            if (event.pointerType === "mouse") setOpen(true);
          }}
          onPointerLeave={(event) => {
            if (event.pointerType === "mouse") setOpen(false);
          }}
          onPointerDown={(event) => {
            if (event.pointerType === "mouse") return;
            event.preventDefault();
            event.stopPropagation();
            setOpen((was) => !was);
          }}
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            event.stopPropagation();
            setOpen((was) => !was);
          }}
          className={`hidden shrink-0 cursor-help rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 lg:inline-flex ${
            active
              ? "text-sky-300 hover:text-white"
              : "text-slate-300 hover:text-sky-600"
          }`}
        >
          <CircleQuestionMark className="h-3.5 w-3.5" />
        </span>
      </Popover.Anchor>
      <Popover.Portal>
        <Popover.Content
          side="bottom"
          align="center"
          // Clears the tab's own padding and the divider beneath it, so the card
          // hangs off the folder rather than overlapping its edge.
          sideOffset={16}
          collisionPadding={12}
          // Focus that moved here would leave the tab the reader was on, and on
          // touch it would summon the keyboard.
          onOpenAutoFocus={(event) => event.preventDefault()}
          className="animate-in fade-in-0 zoom-in-95 z-[60] w-[min(320px,calc(100vw-24px))] rounded-md border border-slate-200 bg-white p-3 shadow-lg"
        >
          <p className="text-xs font-semibold text-slate-700">{entry.label}</p>
          <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
            {entry.blurb}
          </p>
          <Popover.Arrow className="fill-white" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
