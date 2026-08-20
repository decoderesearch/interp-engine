"use client";

/**
 * Why this family exists, on the small circled i beside the picker — and on
 * every row of the open list.
 *
 * Two sides, for two situations. Beside the closed picker the card opens
 * *below*: the picker sits in the band at the top of the window with the
 * diagram underneath, so any other side is off-screen or covering the thing it
 * is explaining. On a row of the open list it opens to the *right*, where there
 * is nothing beside the list — which is also why the icon sits at the right edge
 * of a row. The card anchors on the icon, not on the list, so an icon on the
 * left opens the card over the list it is meant to help you read.
 *
 * A `Popover` driven by hand rather than a `HoverCard`, because a HoverCard
 * never opens on touch and this icon is the only route to the note and the exact
 * release date. Mouse keeps hover; every other pointer gets a tap. The two are
 * told apart by `pointerType` rather than by a media query, since a hybrid
 * laptop is both and a viewport width does not say which one is in your hand.
 */

import { Info } from "lucide-react";
import { Popover } from "radix-ui";
import { useState } from "react";

import { formatRelease } from "@/data/architectures";
import type { Architecture } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  arch: Architecture;
  side?: "bottom" | "right";
  /**
   * Rendered inside a `Select.Item`, which is itself a button in a roving-focus
   * list. A nested button is invalid there, and a tap that reaches the row would
   * pick that architecture instead of explaining it.
   */
  nested?: boolean;
  /** For hiding the icon at the widths where the same prose is already on screen. */
  className?: string;
}

export function ArchitectureInfo({
  arch,
  side = "bottom",
  nested,
  className,
}: Props) {
  const [open, setOpen] = useState(false);

  const handlers = {
    onPointerEnter: (event: React.PointerEvent) => {
      if (event.pointerType === "mouse") setOpen(true);
    },
    onPointerLeave: (event: React.PointerEvent) => {
      if (event.pointerType === "mouse") setOpen(false);
    },
    onPointerDown: (event: React.PointerEvent) => {
      if (event.pointerType === "mouse") return;
      // Both, and in this order: `preventDefault` stops the row's own selection,
      // `stopPropagation` stops the diagram behind it from starting a pan.
      event.preventDefault();
      event.stopPropagation();
      setOpen((was) => !was);
    },
    onKeyDown: (event: React.KeyboardEvent) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      event.stopPropagation();
      setOpen((was) => !was);
    },
    className: cn(
      "shrink-0 cursor-help rounded-full text-slate-300 transition-colors hover:text-sky-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40",
      open && "text-sky-600",
      className,
    ),
    "aria-label": `About ${arch.label}`,
  };

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Anchor asChild>
        {nested ? (
          <span tabIndex={0} role="button" {...handlers}>
            <Info className="h-4 w-4" />
          </span>
        ) : (
          <button type="button" {...handlers}>
            <Info className="h-4 w-4" />
          </button>
        )}
      </Popover.Anchor>
      <Popover.Portal>
        <Popover.Content
          side={side}
          align="start"
          sideOffset={8}
          collisionPadding={12}
          // A hover card that stole focus would move the caret out of whatever
          // the reader was doing, and on touch it would summon the keyboard.
          onOpenAutoFocus={(event) => event.preventDefault()}
          // Above the select's own portal, which it is opened from and would
          // otherwise be drawn behind.
          className="animate-in fade-in-0 zoom-in-95 z-[60] w-[min(320px,calc(100vw-24px))] rounded-md border border-slate-200 bg-white p-3 shadow-lg"
        >
          <div className="flex items-baseline justify-between gap-x-2">
            <span className="text-xs font-semibold text-slate-700">
              {arch.label}
            </span>
            <span className="shrink-0 text-[9px] font-medium tracking-wide text-slate-400 uppercase">
              {formatRelease(arch)}
            </span>
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-slate-500">
            {arch.significance}
          </p>
          {arch.note && (
            <p className="mt-2 border-t border-slate-100 pt-2 text-[11px] leading-relaxed text-slate-500">
              {arch.note}
            </p>
          )}
          <Popover.Arrow className="fill-white" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
