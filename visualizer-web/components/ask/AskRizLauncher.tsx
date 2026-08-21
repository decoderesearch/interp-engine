"use client";

/**
 * The button, and the panel it owns.
 *
 * One component rather than two mounted side by side, because open/closed is
 * the only state either needs and splitting it would push that state up into
 * `page.tsx`, which has no other reason to know a chatbot exists.
 *
 * Two placements, one state. From `sm` up it sits in the header row, where a
 * persistent control belongs on a wide screen and where the diagram's
 * drag-to-pan surface is not underneath it. On a phone the header cannot
 * afford it — it is already carrying the wordmark, the Controls trigger and
 * the icon hamburger — so it becomes the intercom convention instead: a pill
 * pinned over the bottom right corner, carrying its own name because the frog
 * alone says nothing about what tapping it does. The mode toggle gets the same
 * treatment in the bottom centre.
 *
 * The floating one is portaled to `document.body` rather than positioned where
 * it sits in the tree. The header sets `backdrop-blur`, and a `backdrop-filter`
 * makes an element the containing block for its `position: fixed` descendants,
 * so a fixed button left inside the header would pin itself to the header's
 * corner rather than the viewport's.
 */

import Image from "next/image";
import { useCallback, useState } from "react";
import { createPortal } from "react-dom";

import { AskRizPanel } from "@/components/ask/AskRizPanel";
import { useFirstVisit, useJustVisited } from "@/lib/firstVisit";
import type { VisualizerState } from "@/lib/state";
import { useHydrated } from "@/lib/useHydrated";
import { NARROW, useMediaQuery } from "@/lib/useMediaQuery";

/** Shared so the button's `aria-controls` names the panel that opens. */
const PANEL_ID = "ask-riz-panel";

export function AskRizLauncher({ state }: { state: VisualizerState }) {
  // Open on arrival. The panel is the one part of this page nobody thinks to
  // look for -- a diagram invites clicking, a chat box does not announce that
  // it knows the docs -- so it introduces itself once and closes for good with
  // Escape or either button. Nothing is sent until the reader asks something,
  // so an ignored panel costs nothing but the space it sits in.
  //
  // Except on a first visit, where the welcome tour is the introduction and two
  // of them at once is neither. The frog waits behind it and opens as the tour
  // closes, since closing the tour is what marks the visit -- so the panel still
  // introduces itself exactly once, one beat later.
  //
  // Escape is the exception, and it is the one worth having. React flushes a
  // discrete event's updates before the keydown finishes propagating, so the
  // press that dismissed the tour reaches the listener below as it is attached,
  // and closes the panel it just opened. One press clears both, which is what
  // Escape means -- but it is event ordering doing it, not a decision here.
  //
  // A phone gets the introduction and nothing after it. The panel there covers
  // most of what the reader came for, so opening on every later visit is a
  // dialog to dismiss rather than an offer -- `useJustVisited` is true only on
  // the load whose tour they closed, so the frog introduces itself that once
  // and afterwards waits in the corner where its label names it.
  const first = useFirstVisit();
  const justVisited = useJustVisited();
  const narrow = useMediaQuery(NARROW);
  // `null` while that is still deciding. Any press outranks it: `open` is what
  // the reader last asked for, or the arrival default if they have not asked.
  const [override, setOverride] = useState<boolean | null>(null);
  const open = override ?? (narrow ? justVisited : !first);
  const close = useCallback(() => setOverride(false), []);
  const toggle = useCallback(() => setOverride(!open), [open]);
  const hydrated = useHydrated();

  return (
    <>
      <button
        {...buttonProps(open, toggle)}
        className={`hidden shrink-0 cursor-pointer items-center gap-x-2 rounded-full border p-1 transition-colors sm:flex sm:pr-4 sm:pl-2 ${
          open
            ? "border-emerald-500 bg-emerald-100 hover:bg-emerald-200"
            : "border-emerald-400 bg-emerald-50 hover:bg-emerald-100"
        }`}
      >
        <Face className="h-6 w-6" />
        <span
          className={`text-[11px] font-bold whitespace-nowrap ${
            open ? "text-emerald-900" : "text-emerald-700"
          }`}
        >
          Ask Riz
        </span>
      </button>

      {hydrated &&
        createPortal(
          <button
            {...buttonProps(open, toggle)}
            // Above the panel's own `z-40`, so it stays tappable as the thing
            // that closes what it opened. Sits over the scrubber strip, which
            // is the price of the corner and the reason it is only here.
            className={`fixed right-4 bottom-4 z-50 flex h-12 cursor-pointer items-center gap-x-2 rounded-full border pr-1.5 pl-3.5 shadow-lg shadow-slate-900/20 transition-colors sm:hidden ${
              open
                ? "border-emerald-500 bg-emerald-100"
                : "border-emerald-400 bg-emerald-50"
            }`}
          >
            {/* Left of the frog rather than right, so the pill grows inward
                from the corner it is pinned to and the face stays where a
                circular button would have put it. */}
            <span
              className={`text-xs font-bold whitespace-nowrap ${
                open ? "text-emerald-900" : "text-emerald-700"
              }`}
            >
              Ask Riz
            </span>
            <Face className="h-9 w-9" />
          </button>,
          document.body,
        )}

      {/* Only a press asks for the caret. `override` is null for every panel
          that opened by itself, which on a phone is the one that opens as the
          tour closes -- and a keyboard rising over the diagram is not what
          finishing the introduction should look like. */}
      <AskRizPanel
        id={PANEL_ID}
        open={open}
        onClose={close}
        state={state}
        focusOnOpen={override === true}
      />
    </>
  );
}

/** Identical semantics in both placements; only the box around them differs. */
function buttonProps(open: boolean, onToggle: () => void) {
  return {
    type: "button" as const,
    onClick: onToggle,
    "aria-expanded": open,
    "aria-controls": PANEL_ID,
    "aria-label": open ? "Close Ask Riz" : "Ask Riz",
  };
}

/**
 * The source is a frog on a lily pad that runs to all four edges of a square,
 * so a plain circle crop clips the pad and centres on nothing in particular.
 * Scaling up inside the clip puts the frog in the circle and pushes the pad's
 * edges out of it.
 */
function Face({ className }: { className: string }) {
  return (
    <span
      className={`relative block shrink-0 overflow-hidden rounded-full bg-emerald-50 ${className}`}
    >
      <Image
        src="/riz.png"
        alt=""
        width={818}
        height={818}
        // Loaded eagerly now that the panel is open on arrival: the same frog
        // is in the panel header, above the fold, from the first paint.
        priority
        className="absolute inset-0 h-full w-full scale-[1.35] object-cover object-[50%_38%]"
      />
    </span>
  );
}
