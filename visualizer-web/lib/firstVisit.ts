"use client";

/**
 * Whether this reader has been here before, from `localStorage`.
 *
 * One of the two pieces of state in this app that are neither in the URL nor in
 * a component — `lib/askDismissed.ts` is the other, and the same shape. The tour
 * has to open by itself exactly once, and a query parameter would put "has read
 * the introduction" in a link people send each other.
 *
 * Shaped as a store rather than a hook with an effect in it. Reading storage
 * during a render is a read of something the prerender does not have, and the
 * obvious repair — `useState(false)` plus an effect that corrects it — is the
 * setState-in-an-effect that `lib/useHydrated.ts` documents this app as writing
 * around. `useSyncExternalStore` says it in one render instead: the server
 * snapshot is "not a first visit", so the prerendered HTML has nothing open in
 * it, and the client snapshot is the real answer from the moment React takes
 * over.
 *
 * `markVisited` notifies, so the flip is not a value changing under React
 * without it being told. That matters for one thing beyond the tour: Ask Riz
 * opens on arrival for a returning reader and stays shut behind the tour for a
 * new one, so closing the tour is what hands the page over to the frog.
 *
 * `useJustVisited` is the second half of that, and it is what a phone reads
 * instead. It answers a narrower question — did the tour close during *this*
 * page load — which is the difference between the panel introducing itself once
 * and the panel being in the way on every visit afterwards.
 */

import { useSyncExternalStore } from "react";

const KEY = "interp-engine:visited";

/**
 * `null` until the first read. Cached because a `getSnapshot` that returns a
 * fresh answer per call is one React is entitled to loop on, and a storage read
 * per render is not free either.
 */
let visited: boolean | null = null;

/**
 * Whether the flip above happened here, on this page load, rather than on some
 * earlier one. A returning reader arrives visited, so nothing sets this and it
 * stays false for the life of the page.
 */
let visitedHere = false;

const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/**
 * Treats a refusal as a visit. Storage throws in a private window and under a
 * blocked-cookies setting, and the failure mode worth avoiding is a modal that
 * opens on every load with no way to make it stop — a reader who never sees the
 * tour at all can still open it from the header.
 */
function firstVisit(): boolean {
  if (visited === null) {
    try {
      visited = window.localStorage.getItem(KEY) !== null;
    } catch {
      visited = true;
    }
  }
  return !visited;
}

function justVisited(): boolean {
  return visitedHere;
}

/** Nothing is a first visit during the prerender, and nothing closes a tour. */
const neverFirst = () => false;

export function useFirstVisit(): boolean {
  return useSyncExternalStore(subscribe, firstVisit, neverFirst);
}

export function useJustVisited(): boolean {
  return useSyncExternalStore(subscribe, justVisited, neverFirst);
}

export function markVisited(): void {
  try {
    window.localStorage.setItem(KEY, "1");
  } catch {
    // Silent, and deliberately so: the reader asked to close a dialog, not to
    // be told about the storage it could not be remembered in.
  }
  if (visited === true) return;
  visited = true;
  visitedHere = true;
  for (const listener of [...listeners]) listener();
}
