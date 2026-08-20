"use client";

/**
 * The caption under the wordmark, where two of its three claims carry the
 * evidence for themselves: **Fast** opens the throughput table, **Standardized**
 * opens the point-by-point backend support. Both are things a reader is entitled
 * to be sceptical about, and both have a number behind them that was previously
 * only in the repo.
 *
 * Neither card is written here. They live in `components/evidence/`, because the
 * welcome tour makes the same two claims at a width this caption does not have —
 * and the version of a claim that goes stale is the second one.
 *
 * A `HoverCard` rather than the hand-driven `Popover` that `ArchitectureInfo`
 * uses, for the reason `TraitBar` is one too: on a phone the caption is on
 * screen but nothing there can hover, so the cards stay a desktop extra, and
 * the support table has to be *scrolled* — which needs the card to stay open
 * while the pointer crosses into it, and is what a HoverCard's safe area is for.
 *
 * The claims are not links. Each opens on hover and on keyboard focus, and the
 * card is the whole of what it has to say; a claim that also navigated would
 * take the reader off the diagram to a document that says the same thing at
 * length.
 */

import { HoverCard } from "radix-ui";
import type { ReactNode } from "react";

import { PointSupport } from "@/components/evidence/PointSupport";
import { Throughput } from "@/components/evidence/Throughput";
import { cn } from "@/lib/utils";

interface Props {
  repoUrl: string;
  /** The breakpoint the caption appears at, which the header owns. */
  className?: string;
}

export function Tagline({ repoUrl, className }: Props) {
  return (
    <div
      className={cn(
        "text-[11px] text-slate-500 max-sm:leading-snug sm:truncate sm:whitespace-nowrap",
        className,
      )}
    >
      A{" "}
      <Claim
        label="Fast"
        className="font-bold text-emerald-600 italic decoration-emerald-300 hover:text-emerald-700"
        // Wide enough that a model group keeps the bar width it had at four
        // models: the figure printed inside a bar is what sets the floor, and
        // `1,536` in 10px mono does not fit a bar much under 30px.
        width="w-[min(560px,calc(100vw-24px))]"
      >
        <Throughput repoUrl={repoUrl} />
      </Claim>{" "}
      and{" "}
      <Claim
        label="Standardized"
        className="font-bold text-emerald-600 decoration-emerald-300 hover:text-emerald-700"
        width="w-[min(340px,calc(100vw-24px))]"
      >
        <PointSupport />
      </Claim>{" "}
      Interpretability Engine
    </div>
  );
}

/**
 * One word of the caption, and the card it opens below itself. Below because the
 * caption is at the top of the window with the diagram under it, so it is the
 * only side that is neither off-screen nor covering the header — the same reason
 * `ArchitectureInfo` opens downwards from the band beside it.
 */
function Claim({
  label,
  className,
  width,
  children,
}: {
  label: string;
  className: string;
  /** Each card is as wide as its own table, so the caller sets it. */
  width: string;
  children: ReactNode;
}) {
  return (
    <HoverCard.Root openDelay={120} closeDelay={80}>
      <HoverCard.Trigger asChild>
        {/* Focusable, so the card is reachable by keyboard: Radix opens a hover
            card on focus as well as on hover, and a `span` gets neither without
            a tabstop. Not a button — nothing is submitted or toggled here. */}
        <span
          tabIndex={0}
          className={cn(
            "cursor-help underline decoration-dotted underline-offset-2 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40",
            className,
          )}
        >
          {label}
        </span>
      </HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="bottom"
          align="start"
          sideOffset={8}
          collisionPadding={12}
          className={cn(
            // Capped and scrollable because the throughput card is two charts tall
            // now: collision handling shifts a card that does not fit, but it never
            // shrinks one, so on a short window the last rows would sit off-screen.
            "animate-in fade-in-0 zoom-in-95 thin-scrollbar z-50 max-h-[calc(100dvh-24px)] overflow-y-auto rounded-md border border-slate-200 bg-white p-3 shadow-lg",
            width,
          )}
        >
          {children}
          <HoverCard.Arrow className="fill-white" />
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  );
}
