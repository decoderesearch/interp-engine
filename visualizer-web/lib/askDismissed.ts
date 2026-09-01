"use client";

/**
 * Whether this reader has closed Ask Riz before, from `localStorage`.
 *
 * The panel opens by itself on arrival because it is the one part of this page
 * nobody thinks to look for — a diagram invites clicking, a chat box does not
 * announce that it knows the docs. That is worth doing once. Doing it again to
 * somebody who has already closed it turns an offer into a dialog, and the only
 * thing they learn from the second one is where the X is.
 *
 * Same shape as `lib/firstVisit.ts`, and for the same reason: reading storage
 * during a render is a read the prerender cannot make, and the obvious repair —
 * `useState(false)` plus an effect that corrects it — is the setState-in-an-
 * effect that `lib/useHydrated.ts` documents this app as writing around. A store
 * with a server snapshot says it in one render.
 *
 * A storage failure reads as **dismissed**, where `firstVisit`'s reads as
 * visited. Different words, same rule: pick the answer that cannot strand a
 * reader with something that reopens on every load and no way to stop it. The
 * frog is still in the corner either way.
 */

import { useSyncExternalStore } from "react";

const KEY = "interp-engine:ask-dismissed";

/** `null` until the first read, so `getSnapshot` is stable across renders. */
let dismissed: boolean | null = null;

const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function askDismissed(): boolean {
  if (dismissed === null) {
    try {
      dismissed = window.localStorage.getItem(KEY) !== null;
    } catch {
      dismissed = true;
    }
  }
  return dismissed;
}

/** Nothing has been dismissed during the prerender. */
const neverDismissed = () => false;

export function useAskDismissed(): boolean {
  return useSyncExternalStore(subscribe, askDismissed, neverDismissed);
}

export function markAskDismissed(): void {
  try {
    window.localStorage.setItem(KEY, "1");
  } catch {
    // Silent, and deliberately so: the reader asked to close a panel, not to be
    // told about the storage it could not be remembered in.
  }
  if (dismissed === true) return;
  dismissed = true;
  for (const listener of [...listeners]) listener();
}
