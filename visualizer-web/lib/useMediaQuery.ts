"use client";

/**
 * A media query as a boolean, for the handful of decisions CSS cannot make.
 *
 * Almost everything responsive here is a Tailwind breakpoint, which is the right
 * tool and needs no JavaScript. This is for the two places where a width has to
 * change a *number* rather than a class: the row spacing the graph is laid out
 * with, which is arithmetic in `buildGraph`, and which side of a point its card
 * opens on, which is arithmetic in `FlowDiagram`.
 *
 * `useSyncExternalStore` rather than an effect that sets state, so the value is
 * read during render and there is no first paint at the wrong size. The server
 * snapshot is `false`: the markup is prerendered once for every viewport, so the
 * only honest answer before hydration is the one that does not claim to know.
 */

import { useCallback, useSyncExternalStore } from "react";

export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    [query],
  );

  return useSyncExternalStore(
    subscribe,
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** Below Tailwind's `sm`, which is where the band and the cards change shape. */
export const NARROW = "(max-width: 639px)";
