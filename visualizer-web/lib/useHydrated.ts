"use client";

/**
 * Whether this render is running in a browser that has finished hydrating.
 *
 * `createPortal` needs a `document`, which the prerender does not have, so
 * anything portalled has to render nothing until the client takes over. The
 * obvious spelling of that -- `useState(false)` plus an effect that sets it
 * true -- is a setState inside an effect body, which cascades a second render
 * and which the React Compiler lint rules reject.
 *
 * `useSyncExternalStore` says the same thing without the extra render: a store
 * that never changes, whose client snapshot is `true` and whose server
 * snapshot is `false`. React switches between them at exactly the hydration
 * boundary. Same shape as `useMediaQuery` next door, and for the same reason.
 */

import { useSyncExternalStore } from "react";

/** Never fires. The value it guards flips at hydration, not at runtime. */
const subscribe = () => () => {};

export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
