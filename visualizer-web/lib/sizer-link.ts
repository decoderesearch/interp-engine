"use client";

/**
 * The sizer's identity in the address bar: `/sizer/<org>/<model>`.
 *
 * A **path** rather than a query, unlike everything in `lib/link.ts`, because this one names a
 * resource that exists outside the app — a repo on the Hub — and `/sizer/google/gemma-3-12b-pt`
 * reads as that repo's page in a way `?model=google%2Fgemma-3-12b-pt` never will. The slash in a
 * model id is the point: it survives into the path unescaped, so the URL is the id.
 *
 * `next.config.ts` rewrites `/sizer/:path*` to `/`. There is no second route and no second render
 * tree — this app is one prerendered page, and adding a dynamic route to carry a string would give
 * up the CDN story for nothing. The path is read here instead, once, with the same discipline
 * `lib/link.ts` applies to the query: **an initial condition, read once, and a mirror, written on
 * every change.**
 */

import { useMemo, useSyncExternalStore } from "react";

const PREFIX = "/sizer";

/**
 * A model id from a path, or the empty string.
 *
 * Validated only as far as its shape, because the set of valid ids is the Hub's and not ours: two
 * segments at most, no traversal, nothing that would not survive being put back. Anything else opens
 * the sizer empty rather than erroring, which is the only behaviour worth having for a URL somebody
 * re-typed out of a screenshot.
 */
export function decodeSizerPath(pathname: string): {
  sizing: boolean;
  model: string;
} {
  if (pathname !== PREFIX && !pathname.startsWith(`${PREFIX}/`)) {
    return { sizing: false, model: "" };
  }
  const rest = pathname.slice(PREFIX.length).replace(/^\/+|\/+$/g, "");
  if (!rest) return { sizing: true, model: "" };

  const segments = rest.split("/").map(decodeURIComponent);
  const legal =
    segments.length <= 2 &&
    segments.every((part) => part && part !== "." && part !== "..");
  return { sizing: true, model: legal ? segments.join("/") : "" };
}

/** The path for a model, or bare `/sizer` when none has been named yet. */
export function encodeSizerPath(model: string): string {
  const id = model.trim().replace(/^\/+|\/+$/g, "");
  if (!id) return PREFIX;
  return `${PREFIX}/${id.split("/").map(encodeURIComponent).join("/")}`;
}

/**
 * The path as it was when the page loaded, taken while this module is evaluated.
 *
 * Eagerly, for the reason `lib/link.ts` gives about the query: the app rewrites the address bar as
 * the reader moves, and read late this would be the arrival with the arrival already erased.
 */
const ARRIVED_AT =
  typeof window === "undefined" ? "/" : window.location.pathname;

const subscribe = () => () => {};
const onServer = () => "/";
const onClient = () => ARRIVED_AT;

/**
 * Where the reader arrived.
 *
 * `useSyncExternalStore` rather than an effect, matching `useOpenedWith`: the prerendered HTML is
 * always `/`, so the server snapshot and the hydration render that has to match it must both say so,
 * and only the render after hydration may see the real path.
 */
export function useSizerRoute(): { sizing: boolean; model: string } {
  const pathname = useSyncExternalStore(subscribe, onClient, onServer);
  return useMemo(() => decodeSizerPath(pathname), [pathname]);
}
