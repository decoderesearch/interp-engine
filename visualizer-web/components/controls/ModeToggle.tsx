"use client";

/**
 * One diagram or two. A two-state toggle rather than a button that swaps for a
 * different button: the mode is visible whichever one you are in, and leaving
 * compare mode is the same control that entered it.
 *
 * Both halves are always shown, so the header's left edge keeps its width and
 * the wordmark beside it does not shift when the mode changes.
 */

import { Columns2, Square } from "lucide-react";

interface Props {
  comparing: boolean;
  onSingle: () => void;
  onCompare: () => void;
}

export function ModeToggle({ comparing, onSingle, onCompare }: Props) {
  return (
    <div
      role="radiogroup"
      aria-label="View mode"
      className="flex shrink-0 rounded-md border border-slate-300 bg-white p-0.5"
    >
      <Option
        active={!comparing}
        onClick={onSingle}
        icon={<Square className="h-3.5 w-3.5" />}
        label="Single Mode"
      />
      <Option
        active={comparing}
        onClick={onCompare}
        icon={<Columns2 className="h-3.5 w-3.5" />}
        label="Compare Mode"
      />
    </div>
  );
}

function Option({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      onClick={onClick}
      title={label}
      className={`flex cursor-pointer items-center gap-x-1.5 rounded-sm px-2 py-1 text-xs font-medium whitespace-nowrap transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 md:px-3 ${
        active
          ? "bg-sky-100 text-sky-700"
          : "text-slate-500 hover:bg-slate-100 hover:text-slate-700"
      }`}
    >
      <span className={active ? "text-sky-500" : "text-slate-400"}>{icon}</span>
      {/* Icon-only until `md`, where the header also carries the wordmark, the
          Controls trigger, Riz and the icon cluster. A breakpoint later than
          those two: at `sm` all three labels appear at once and the row is
          ~15px over, and this is the pair that survives losing its text —
          a filled square against two columns. `sr-only` rather than `hidden`,
          because the text is this radio's accessible name. */}
      <span className="sr-only md:not-sr-only">{label}</span>
    </button>
  );
}
