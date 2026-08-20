"use client";

/**
 * All of the app's state, in one hook.
 *
 * The invariants worth knowing: traits are always run through `resolveTraits`
 * so implications and conflicts hold, dimensions are always clamped to the
 * floors the active traits impose, and the architecture id follows the trait
 * set rather than the other way round — hand-editing the toggles until they
 * match a family snaps the picker back to that family's name. The one thing
 * the trait set cannot decide is which of two identically-wired families is
 * meant, so the last explicit pick is kept to break that tie.
 *
 * Which point the card is open on is two pieces of state, not one: `hovered`
 * follows the pointer and `locked` is pinned by a click. `shown` is the answer,
 * and the only one the diagram reads — a lock outranks the pointer, so nothing
 * downstream has to know which of the two put it there.
 *
 * Some of that state came from the address bar and all of it goes back there.
 * The `Link` this is handed is where the reader arrived and cannot change under
 * a mounted hook — `Page` remounts on a new one — so it is read in the
 * initialisers below and nowhere else. See `lib/link.ts`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  ARCHITECTURES,
  CUSTOM_ARCHITECTURE_ID,
  DEFAULT_ARCHITECTURE_ID,
  architecture,
  matchArchitecture,
} from "@/data/architectures";
import { DEFAULT_DIMENSIONS, DIMENSIONS } from "@/data/dimensions";
import { DEFAULT_ENGINE_ID, nameMatches, parseAddress } from "@/data/engines";
import { formulaChanges } from "@/data/formulas";
import { minLayersFor, resolveTraits } from "@/data/traits";
import { alignmentFor, buildGraph } from "@/lib/buildGraph";
import { diffGraphs, pointEffects } from "@/lib/diff";
import { encodeLink, type Link } from "@/lib/link";
import { useHydrated } from "@/lib/useHydrated";
import { NARROW, useMediaQuery } from "@/lib/useMediaQuery";
import type {
  Dimensions,
  EngineId,
  Graph,
  GraphNode,
  PointName,
  TraitId,
} from "@/lib/types";

/** How long a point stays pulsed after the trait change that reworked it. */
const FLASH_MS = 1600;

/** The points to pulse, and a token that restarts the animation on re-toggle. */
export interface Flash {
  points: Set<PointName>;
  token: number;
}

const NO_FLASH: Flash = { points: new Set(), token: 0 };

/** A node the card is open on, and which diagram it is in when two are stacked. */
export interface Hover {
  node: GraphNode;
  pane: number;
}

/** A pinned point, tagged with the graph it was pinned on. See `locked`. */
interface Lock extends Hover {
  graph: Graph;
}

export function useVisualizer(link: Link) {
  const [rawDims, setRawDims] = useState<Dimensions>(() => linkedDims(link));
  const [traits, setTraits] = useState<Set<TraitId>>(() =>
    traitsFor(link.arch ?? DEFAULT_ARCHITECTURE_ID),
  );
  const [engineId, setEngineId] = useState<EngineId>(DEFAULT_ENGINE_ID);
  const [hovered, setHoveredState] = useState<Hover | null>(null);
  const [query, setQuery] = useState("");
  const [compareId, setCompareId] = useState<string | null>(link.vs);
  const [picked, setPicked] = useState(link.arch ?? DEFAULT_ARCHITECTURE_ID);
  const [focusTrait, setFocusTrait] = useState<TraitId | null>(null);
  const narrow = useMediaQuery(NARROW);

  const compareTraits = useMemo(() => {
    const arch = compareId === null ? null : architecture(compareId);
    return arch ? resolveTraits(arch.traits) : null;
  }, [compareId]);
  const comparing = compareTraits !== null;

  // Both diagrams share one set of dimensions, so the only thing that varies
  // between them is the architecture. The floors of both have to hold.
  const activeTraits = useMemo(
    () => (compareTraits ? new Set([...traits, ...compareTraits]) : traits),
    [traits, compareTraits],
  );

  const dims = useMemo(
    () => clamp(rawDims, activeTraits),
    [rawDims, activeTraits],
  );
  const architectureId = useMemo(
    () => matchArchitecture(traits, picked),
    [traits, picked],
  );

  // Watching the trait set rather than hooking the two setters catches every
  // path into it — a pill, the architecture picker, and a trait pulled in by
  // another one's `implies`.
  const [flash, setFlash] = useState<Flash>(NO_FLASH);
  const previousTraits = useRef(traits);
  useEffect(() => {
    const before = previousTraits.current;
    previousTraits.current = traits;
    const changed = formulaChanges(before, traits);
    if (changed.size === 0) return;
    setFlash({ points: changed, token: Date.now() });
    const timer = setTimeout(() => setFlash(NO_FLASH), FLASH_MS);
    return () => clearTimeout(timer);
  }, [traits]);

  // One alignment, both graphs: the same point lands in the same column and on
  // the same row on both sides, which is what makes them worth stacking.
  const align = useMemo(
    () => alignmentFor(dims, { traits }, { traits: compareTraits ?? traits }),
    [dims, traits, compareTraits],
  );

  // Tighter rows only where the height is genuinely scarce: two diagrams on a
  // phone. One diagram on a phone has the whole screen and reads better with the
  // room, and neither of those is true of a laptop in either mode.
  const compact = narrow && comparing;

  const graph = useMemo(
    () => buildGraph({ dims, traits, align, compact }),
    [dims, traits, align, compact],
  );

  const compareGraph = useMemo(
    () =>
      compareTraits && compareId
        ? buildGraph({ dims, traits: compareTraits, align, compact })
        : null,
    [dims, compareTraits, compareId, align, compact],
  );

  // Down here rather than beside the rest of the state: a lock names the graph
  // it was taken on, so a linked one has nothing to resolve against until both
  // graphs have been built.
  const [held, setHeld] = useState<Lock | null>(() =>
    linkedLock(link, graph, compareGraph),
  );

  const allDiffs = useMemo(
    () =>
      compareGraph && compareTraits
        ? diffGraphs(
            { graph, traits },
            { graph: compareGraph, traits: compareTraits },
          )
        : null,
    [graph, traits, compareGraph, compareTraits],
  );

  // Pointing at one architecture difference asks a narrower question than the
  // red rings answer by default: not "where do these two disagree" but "where
  // does *this* disagreement show up". Attribution comes from the same
  // `pointEffects` that writes the trait's card, so the rings that light up are
  // the points the card names.
  const focused = useMemo(() => {
    if (!allDiffs || !focusTrait || !compareTraits) return null;
    const base = traits.has(focusTrait) ? traits : compareTraits;
    const { touched } = pointEffects(focusTrait, base);
    const ids = new Set<string>();
    for (const node of [...graph.nodes, ...(compareGraph?.nodes ?? [])]) {
      if (allDiffs.has(node.id) && touched.has(node.point)) ids.add(node.id);
    }
    return ids;
  }, [allDiffs, focusTrait, traits, compareTraits, graph, compareGraph]);

  const diff = focused ?? allDiffs;

  // Where the focused difference actually is, taken across both panes so the
  // two agree on whether it needs scrolling to. Sorted, so element zero is the
  // first one in reading order.
  const focusColumns = useMemo(() => {
    if (!focused) return null;
    const xs = new Set<number>();
    for (const node of [...graph.nodes, ...(compareGraph?.nodes ?? [])]) {
      if (focused.has(node.id)) xs.add(node.x);
    }
    return [...xs].sort((a, b) => a - b);
  }, [focused, graph, compareGraph]);

  // Normalized once, here, so the diagram and the field's counter agree on what
  // counts as a hit.
  const search = query.trim().toLowerCase();
  const matchCount = useMemo(() => {
    if (!search) return null;
    const hits = (side: typeof graph | null, set: Set<TraitId>) =>
      (side?.nodes ?? []).filter((node) =>
        nameMatches(
          engineId,
          {
            point: node.point,
            layer: node.layer,
            stream: node.stream,
            traits: set,
          },
          search,
        ),
      ).length;
    return (
      hits(graph, traits) +
      (compareTraits ? hits(compareGraph, compareTraits) : 0)
    );
  }, [graph, compareGraph, engineId, traits, compareTraits, search]);

  const setDimension = useCallback((key: keyof Dimensions, value: number) => {
    setRawDims((prev) => ({ ...prev, [key]: value }));
  }, []);

  // The pane travels with the node so only the pane the pointer is in opens a
  // card, while the other still rings its copy of the same point.
  const setHovered = useCallback((node: GraphNode | null, pane = 0) => {
    setHoveredState(node ? { node, pane } : null);
  }, []);

  const setLocked = useCallback(
    (node: GraphNode | null, pane = 0) => {
      const on = pane === 0 ? graph : compareGraph;
      setHeld(node && on ? { node, pane, graph: on } : null);
    },
    [graph, compareGraph],
  );

  // A lock outlives the pointer, so unlike a hover it can outlive the point it
  // was taken on: turning off MoE deletes the router the card is describing.
  // Honoured only while the graph it was taken on is still the one being drawn,
  // rather than cleared from an effect — `buildGraph` is memoised on the traits
  // and dimensions, so this comparison is the identity check it looks like, and
  // it costs one render fewer than reacting to the change after the fact.
  const locked =
    held && held.graph === (held.pane === 0 ? graph : compareGraph)
      ? held
      : null;

  /** What the card is open on: a lock if one is held, the pointer otherwise. */
  const shown: Hover | null = locked ?? hovered;

  // The address bar follows the diagram, so the link to what the reader is
  // looking at is the one already in it and there is nothing to press.
  //
  // `replaceState` rather than a router navigation, for two reasons: this is the
  // same page renamed rather than a different one, and a history entry per point
  // clicked would make Back an undo button for a card that the next click closes
  // anyway. The comparison against what is already there is what keeps arriving
  // quiet — a link that is already canonical is not rewritten, and one that is
  // not, `?arch=` naming the default or a parameter this app never wrote, is
  // tidied once.
  //
  // Not until hydration, and this is the whole reason `useHydrated` is called
  // here: until then this mount is the plain-visit one the prerender produced,
  // whatever the query says, and writing from it would erase the link the reader
  // followed before `Page` has been allowed to read it.
  const hydrated = useHydrated();
  const mirror = encodeLink({
    arch: architectureId,
    vs: compareId,
    point: locked?.node.id ?? null,
    pane: locked?.pane ?? 0,
  });
  useEffect(() => {
    if (!hydrated || window.location.search === mirror) return;
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${mirror}`,
    );
  }, [hydrated, mirror]);

  /**
   * The linked point, for the pane that holds it to bring into view on arrival.
   * A point ten layers along is off the side of a canvas thousands of pixels
   * wide, and a card open on a point the reader cannot see reads as a card about
   * nothing. Only ever the point the page was opened at: every later lock was a
   * click on something already on screen.
   */
  const reveal = useMemo(
    () => (link.point === null ? null : { id: link.point, pane: link.pane }),
    [link],
  );

  const toggleTrait = useCallback((id: TraitId) => {
    setTraits((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return resolveTraits(next);
    });
  }, []);

  // Opening a comparison against a near-identical family would look broken, so
  // the default partner is whichever one differs in the most traits. Ties go to
  // list order, so the same architecture always opens the same pairing.
  const startCompare = useCallback(() => {
    setCompareId((current) => {
      if (current !== null) return current;
      let best: string | null = null;
      let bestScore = -1;
      for (const arch of ARCHITECTURES) {
        const other = resolveTraits(arch.traits);
        if (sameTraits(other, traits)) continue;
        const score = [...new Set([...other, ...traits])].filter(
          (id) => other.has(id) !== traits.has(id),
        ).length;
        if (score > bestScore) {
          best = arch.id;
          bestScore = score;
        }
      }
      return best;
    });
  }, [traits]);

  const selectArchitecture = useCallback((id: string) => {
    if (id === CUSTOM_ARCHITECTURE_ID) return;
    const arch = architecture(id);
    if (!arch) return;
    setPicked(id);
    setTraits(resolveTraits(arch.traits));
  }, []);

  return {
    dims,
    traits,
    activeTraits,
    engineId,
    architectureId,
    compareId,
    compareTraits,
    comparing,
    graph,
    compareGraph,
    diff,
    focusTrait,
    focusColumns,
    setFocusTrait,
    narrow,
    compact,
    locked,
    shown,
    reveal,
    flash,
    query,
    search,
    matchCount,
    setQuery,
    setDimension,
    toggleTrait,
    selectArchitecture,
    startCompare,
    setCompareId,
    setEngineId,
    setHovered,
    setLocked,
  };
}

export type VisualizerState = ReturnType<typeof useVisualizer>;

/** The trait set a named family resolves to. */
function traitsFor(id: string): Set<TraitId> {
  return resolveTraits(architecture(id)?.traits ?? ARCHITECTURES[0].traits);
}

/**
 * The dimensions a linked point needs in order to exist at all.
 *
 * `resid_post.9` is a point on a ten-layer stack and nothing whatever on the
 * four-layer default, so the depth follows the address rather than the address
 * being dropped for being deep. Same for a stream coordinate. Both are already
 * known to be inside the sliders' range: `decodeLink` drops an address that
 * names a layer no slider reaches.
 */
function linkedDims(link: Link): Dimensions {
  const at = link.point === null ? null : parseAddress(link.point);
  if (!at) return DEFAULT_DIMENSIONS;
  return {
    ...DEFAULT_DIMENSIONS,
    layers: Math.max(DEFAULT_DIMENSIONS.layers, (at.layer ?? 0) + 1),
    streams: Math.max(DEFAULT_DIMENSIONS.streams, (at.stream ?? 0) + 1),
  };
}

/**
 * The lock a linked point takes, on whichever pane's graph draws it.
 *
 * Absent rather than an error when the point is not there: the address may name
 * a point this architecture does not have — no `resid_mid` on a parallel block,
 * no router on a dense layer — and the diagram the link asked for is still worth
 * opening without a card on it.
 */
function linkedLock(
  link: Link,
  graph: Graph,
  compareGraph: Graph | null,
): Lock | null {
  if (link.point === null) return null;
  const on = link.pane === 1 ? compareGraph : graph;
  const node = on?.nodes.find((candidate) => candidate.id === link.point);
  return on && node ? { node, pane: link.pane, graph: on } : null;
}

/**
 * Keep every dimension inside the range its trait set allows. Clamping on read
 * rather than on write means turning a trait off restores whatever the user had
 * chosen before it forced a floor.
 */
function sameTraits(a: Set<TraitId>, b: Set<TraitId>): boolean {
  return a.size === b.size && [...a].every((id) => b.has(id));
}

function clamp(dims: Dimensions, traits: Set<TraitId>): Dimensions {
  const out = { ...dims };
  out.layers = Math.max(out.layers, minLayersFor(traits, dims.windowRatio));

  for (const spec of DIMENSIONS) {
    const value = out[spec.key];
    let capped = Math.min(Math.max(value, spec.min), spec.max);
    if (spec.boundedBy) capped = Math.min(capped, out[spec.boundedBy]);
    out[spec.key] = capped;
  }
  return out;
}
