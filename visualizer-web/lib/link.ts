"use client";

/**
 * The diagram's identity in the address bar.
 *
 * Four things are worth a link, and they are the four a reader cannot describe
 * in words to someone else without them having to go clicking: which
 * architecture is drawn, what it is drawn against, which point's card is open,
 * and which of two stacked panes that point is in. The sliders, the naming
 * stack and the search field are ways of looking at those rather than things to
 * look at, and are left out so the query stays short enough to paste into a
 * sentence.
 *
 * The query is an **initial condition, read once, and a mirror, written on
 * every change** — never a live input. `useVisualizer` rewrites the address bar
 * as the diagram moves, and reading those writes back in would be a loop with
 * the reader's own state in it.
 *
 * Defaults are omitted rather than spelled out: a plain visit stays at `/`, and
 * a link carries only what it changed.
 */

import { useMemo, useSyncExternalStore } from "react";

import {
  CUSTOM_ARCHITECTURE_ID,
  DEFAULT_ARCHITECTURE_ID,
  architecture,
} from "@/data/architectures";
import { DIMENSIONS } from "@/data/dimensions";
import { parseAddress } from "@/data/engines";
import { pointSpec } from "@/data/points";

export interface Link {
  /** A named family, or null for whichever one the app opens on. */
  arch: string | null;
  /** The family in the second pane, or null for a single diagram. */
  vs: string | null;
  /** A point's address, spelled exactly as `GraphNode.id` spells it. */
  point: string | null;
  /** Which pane `point` is in. Zero unless there are two and it is the lower. */
  pane: number;
}

/** The deepest a slider goes. An address past it names nothing drawable. */
const LIMIT = {
  layer: DIMENSIONS.find((spec) => spec.key === "layers")?.max ?? 0,
  stream: DIMENSIONS.find((spec) => spec.key === "streams")?.max ?? 0,
};

/**
 * Whatever of a query string this app recognises, and nothing else.
 *
 * Every field is validated here so that the rest of the app can treat a `Link`
 * as a set of things that exist. A misspelt family, a point that is not in the
 * table, a comparison of an architecture against itself, a lock on a second
 * pane that was never asked for: each is dropped on its own, leaving the rest
 * of the link to work. A link that has been hand-edited into nonsense opens the
 * default diagram rather than an error, which is the only behaviour worth
 * having for a URL someone re-typed out of a screenshot.
 */
export function decodeLink(search: string): Link {
  const params = new URLSearchParams(search);

  const arch = named(params.get("arch"));
  const point = drawable(params.get("point"));

  // Against itself is not a comparison, and the second pane's picker excludes
  // the first pane's choice for the same reason.
  const other = named(params.get("vs"));
  const vs = other === (arch ?? DEFAULT_ARCHITECTURE_ID) ? null : other;

  return {
    arch,
    vs,
    point,
    pane: vs !== null && point !== null && params.get("pane") === "1" ? 1 : 0,
  };
}

/** The query for a link, `?` and all, or the empty string for a plain visit. */
export function encodeLink(link: Link): string {
  const params = new URLSearchParams();

  // A hand-edited trait set has no name to write down. The link then says
  // nothing about the architecture rather than naming the family the toggles
  // started from, which would be a link to a diagram the sender is not looking
  // at — the toggles themselves are not in the URL.
  if (
    link.arch !== null &&
    link.arch !== DEFAULT_ARCHITECTURE_ID &&
    link.arch !== CUSTOM_ARCHITECTURE_ID
  ) {
    params.set("arch", link.arch);
  }
  if (link.vs !== null) params.set("vs", link.vs);
  if (link.point !== null) {
    params.set("point", link.point);
    if (link.pane === 1) params.set("pane", "1");
  }

  const query = params.toString();
  return query === "" ? "" : `?${query}`;
}

/** An architecture id, if it names one of the families this app knows. */
function named(id: string | null): string | null {
  return id !== null && architecture(id) !== undefined ? id : null;
}

/** An address, if it names a point the diagram could draw at that depth. */
function drawable(address: string | null): string | null {
  if (address === null) return null;
  const at = parseAddress(address);
  if (!at || !pointSpec(at.point)) return null;
  if (at.layer !== null && at.layer >= LIMIT.layer) return null;
  if (at.stream !== null && at.stream >= LIMIT.stream) return null;
  return address;
}

/**
 * The query as it was when the page loaded, taken while this module is being
 * evaluated and never read again.
 *
 * Eagerly, rather than at the first render that is allowed to look, because by
 * then the address bar is no longer evidence: the app rewrites it as the diagram
 * changes, and the first of those writes happens in an effect that React runs
 * *before* the parent's store subscription reads anything (children commit
 * first). Read late, this would be the link with the reader's link already
 * erased from it.
 */
const ARRIVED_AT = typeof window === "undefined" ? "" : window.location.search;

/** Never fires: this is where the reader arrived, which cannot change. */
const subscribe = () => () => {};
const onServer = () => "";
const onClient = () => ARRIVED_AT;

/**
 * The link the page was opened with.
 *
 * `useSyncExternalStore` rather than an effect that sets state, the same shape
 * as `useHydrated` next door and for the same reason: there is no query during
 * the prerender, so the server snapshot and the hydration render that has to
 * match it both have to be the plain visit the prerendered HTML shows, and only
 * the render after hydration may see the real one. That is one render later than
 * a `useState` initialiser can read, which is why `Page` keys the app on this
 * value rather than passing it into one.
 */
export function useOpenedWith(): Link {
  const search = useSyncExternalStore(subscribe, onClient, onServer);
  return useMemo(() => decodeLink(search), [search]);
}
