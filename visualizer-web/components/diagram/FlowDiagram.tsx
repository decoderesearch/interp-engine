"use client";

/**
 * The canvas.
 *
 * Drawn as one wide SVG at 1:1 pixel scale inside a horizontally scrolling
 * container, so the popover can be positioned in the same coordinate space as
 * the nodes without any projection maths. The container can be dragged to pan,
 * and the scrubber underneath it navigates the full width.
 */

import { AnimatePresence, motion } from "motion/react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";

import {
  HookPopover,
  POPOVER_NARROW_WIDTH,
  POPOVER_WIDTH,
} from "@/components/HookPopover";
import { HookLabel, HookPoint } from "@/components/diagram/HookPoint";
import { Scrubber } from "@/components/diagram/Scrubber";
import { PLACEHOLDER_HF_ID } from "@/data/snippets";
import type { VisualizerState } from "@/lib/state";
import type {
  EngineId,
  Graph,
  GraphNode,
  LayerBand,
  Role,
  TraitId,
} from "@/lib/types";

const EDGE_STROKE: Record<Role, string> = {
  resid: "stroke-role-resid",
  attn: "stroke-role-attn",
  mlp: "stroke-role-mlp",
  route: "stroke-role-route",
  global: "stroke-role-global",
};

const KIND_LABEL: Record<LayerBand["kind"], string | null> = {
  full_attention: null,
  sliding_attention: "sliding",
  linear_attention: "linear",
};

/** What is worth saying about a band beyond its number. Often nothing. */
function bandDetail(band: LayerBand): string {
  return [KIND_LABEL[band.kind], band.isMoe ? "moe" : null]
    .filter(Boolean)
    .join(" · ");
}

/** Pointer travel that turns a click into a pan. */
const PAN_THRESHOLD = 4;

/** A point this close to an edge is technically on screen and no use there. */
const EDGE_MARGIN = 48;

/** Long enough for a smooth scroll to land before `sync` is obeyed again. */
const SETTLE_MS = 700;

/** Grace period for the pointer to cross the gap from a point to its card. */
const CLOSE_DELAY_MS = 140;

interface Props {
  state: VisualizerState;
  /** Which architecture this pane draws, and under which traits. */
  graph: Graph;
  traits: Set<TraitId>;
  totalLayers: number;
  /**
   * A checkpoint of the architecture this pane draws, for the snippets in the
   * hover card to load. Absent while the traits match no named family, where
   * there is no checkpoint to name.
   */
  hfId?: string;
  /**
   * Distinguishes the two panes of a comparison. Both ring the point under the
   * pointer; only the one the pointer is actually in opens a card.
   */
  pane?: number;
  /** Drives the scroll when a difference needs bringing into view. */
  lead?: boolean;
  /** Node ids that differ from the other pane. */
  diff?: Set<string> | null;
  /** One scrubber serves both panes, so only the last one draws it. */
  scrubber?: boolean;
  /**
   * Scroll position shared with the other pane. Both axes, because an
   * alignment gives the two graphs the same rows as well as the same columns.
   */
  sync?: Scroll;
  onSync?: (at: Scroll) => void;
}

interface Scroll {
  left: number;
  top: number;
}

export function FlowDiagram({
  state,
  graph,
  traits,
  totalLayers,
  hfId,
  pane = 0,
  lead = false,
  diff = null,
  scrubber = true,
  sync,
  onSync,
}: Props) {
  const {
    engineId,
    setHovered,
    setLocked,
    locked,
    shown,
    reveal,
    flash,
    search,
    focusTrait,
    focusColumns,
    narrow,
    compact,
  } = state;

  // Pointing at one architecture difference asks about those points and no
  // others, so the rest of the drawing steps back the way it does for a search.
  const focused = focusTrait !== null;

  // Paint order, which is the only depth SVG has: see the group below.
  const shownId = shown?.node.id ?? null;
  const orderedNodes = useMemo(
    () =>
      shownId === null || !graph.nodes.some((node) => node.id === shownId)
        ? graph.nodes
        : [
            ...graph.nodes.filter((node) => node.id !== shownId),
            ...graph.nodes.filter((node) => node.id === shownId),
          ],
    [graph.nodes, shownId],
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState({
    left: 0,
    top: 0,
    width: 0,
    height: 0,
    rectLeft: 0,
    rectTop: 0,
  });

  // Kept in refs: a pan has to survive re-renders and must not cause them.
  const pan = useRef<{
    x: number;
    y: number;
    left: number;
    top: number;
  } | null>(null);
  const didPan = useRef(false);
  const [panning, setPanning] = useState(false);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** Timestamp until which this pane is animating itself and ignores `sync`. */
  const settling = useRef(0);

  const show = useCallback(
    (node: GraphNode | null) => {
      // A lock is exactly the pointer *not* deciding what the card is on, so
      // hover is inert until it is released — including the close on leaving,
      // which would otherwise queue up behind the lock and fire on release.
      if (locked) return;
      if (closeTimer.current) clearTimeout(closeTimer.current);
      if (node) {
        if (!panning) setHovered(node, pane);
      } else {
        closeTimer.current = setTimeout(() => setHovered(null), CLOSE_DELAY_MS);
      }
    },
    [panning, setHovered, pane, locked],
  );

  const select = useCallback(
    (node: GraphNode) => {
      // A drag that started on this point is a pan, not a choice.
      if (didPan.current) return;
      // Dropped so releasing the lock does not hand the card back to whichever
      // point the pointer happened to be over when the lock was taken.
      setHovered(null);
      setLocked(node, pane);
    },
    [setHovered, setLocked, pane],
  );

  const keepOpen = useCallback(() => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
  }, []);

  // Released on the way down rather than on click, so a drag to pan lets go of
  // the card at the moment the drag starts. Points and the card itself are
  // exempt: a click on another point moves the lock, and the card is full of
  // things to click — tabs, a copy button — none of which is "outside".
  useEffect(() => {
    if (!locked) return;
    const onDown = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (target?.closest("[data-hook-point],[data-no-pan]")) return;
      setLocked(null);
    };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  }, [locked, setLocked]);

  useEffect(
    () => () => void (closeTimer.current && clearTimeout(closeTimer.current)),
    [],
  );

  // Track the viewport for the scrubber. Re-runs on width changes too, since a
  // taller stack resizes the content rather than the container.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => {
      // The rect as well as the scroll offsets: the card is drawn in a layer over
      // the whole window rather than inside this pane, so it needs to know where
      // this pane starts. Nothing here scrolls the window itself, so a scroll of
      // this container or a resize of it is every way the rect can move.
      const rect = el.getBoundingClientRect();
      setView({
        left: el.scrollLeft,
        top: el.scrollTop,
        width: el.clientWidth,
        height: el.clientHeight,
        rectLeft: rect.left,
        rectTop: rect.top,
      });
      onSync?.({ left: el.scrollLeft, top: el.scrollTop });
    };
    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => {
      el.removeEventListener("scroll", measure);
      observer.disconnect();
    };
  }, [graph.width, onSync]);

  // Follow the other pane. The tolerance is what stops the two from driving
  // each other in a loop, since applying a scroll fires another scroll event.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !sync) return;
    // A smooth scroll of our own emits positions the other pane echoes back a
    // frame late; applying the echo would snap this pane to where it was and
    // cancel the animation.
    if (performance.now() < settling.current) return;
    if (Math.abs(el.scrollLeft - sync.left) > 1) el.scrollLeft = sync.left;
    if (Math.abs(el.scrollTop - sync.top) > 1) el.scrollTop = sync.top;
  }, [sync]);

  // Pointing at a difference whose points are all off-screen looks like
  // pointing at nothing, so bring the first of them to the middle. Only the
  // lead pane moves: the other is dragged along by `sync`, frame by frame,
  // which keeps the pair together through the animation.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !lead || !focusColumns?.length) return;

    // A canvas narrower than its container is centred by `m-auto`, so scroll
    // zero is not canvas zero.
    const pad = Math.max(0, (el.clientWidth - graph.width) / 2);
    const seen = focusColumns.some((x) => {
      const at = x + pad;
      return (
        at > el.scrollLeft + EDGE_MARGIN &&
        at < el.scrollLeft + el.clientWidth - EDGE_MARGIN
      );
    });
    if (seen) return;

    settling.current = performance.now() + SETTLE_MS;
    el.scrollTo({
      left: Math.max(
        0,
        Math.min(
          focusColumns[0] + pad - el.clientWidth / 2,
          el.scrollWidth - el.clientWidth,
        ),
      ),
      behavior: "smooth",
    });
  }, [focusColumns, lead, graph.width]);

  // A point arrived at by link is usually off the side of a canvas thousands of
  // pixels wide, so the pane holding it opens looking at it. Instantly, unlike
  // the scroll above: this is where the reader asked to be rather than somewhere
  // they are being taken, and the card is already open on it.
  //
  // Once, tracked in a ref rather than in state: this re-runs every time the
  // graph is rebuilt, and a second scroll would drag a reader who had panned
  // away back to a point they had left.
  const revealed = useRef(false);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || revealed.current || !reveal || reveal.pane !== pane) return;
    const node = graph.nodes.find((candidate) => candidate.id === reveal.id);
    if (!node) return;
    revealed.current = true;

    // Both axes, because a canvas smaller than its container is centred by
    // `m-auto` on each of them, and a phone in compare mode has neither to
    // spare.
    const padX = Math.max(0, (el.clientWidth - graph.width) / 2);
    const padY = Math.max(0, (el.clientHeight - graph.height) / 2);
    el.scrollTo({
      left: clamp(
        node.x + padX - el.clientWidth / 2,
        0,
        el.scrollWidth - el.clientWidth,
      ),
      top: clamp(
        node.y + padY - el.clientHeight / 2,
        0,
        el.scrollHeight - el.clientHeight,
      ),
    });
  }, [reveal, pane, graph]);

  const scrollTo = useCallback((left: number) => {
    const el = scrollRef.current;
    if (el)
      el.scrollLeft = Math.max(
        0,
        Math.min(left, el.scrollWidth - el.clientWidth),
      );
  }, []);

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const start = pan.current;
      const el = scrollRef.current;
      if (!start || !el) return;
      const dx = event.clientX - start.x;
      const dy = event.clientY - start.y;
      if (!didPan.current && Math.hypot(dx, dy) < PAN_THRESHOLD) return;
      didPan.current = true;
      setPanning(true);
      setHovered(null);
      el.scrollLeft = start.left - dx;
      el.scrollTop = start.top - dy;
    };
    const onUp = () => {
      pan.current = null;
      setPanning(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [setHovered]);

  return (
    <div className="relative flex h-full w-full flex-col">
      <div
        ref={scrollRef}
        className={`thin-scrollbar relative flex min-h-0 flex-1 overflow-auto ${
          panning ? "cursor-grabbing select-none" : "cursor-grab"
        }`}
        onPointerDown={(event) => {
          // Touch and trackpad scrolling already work; hijacking them would
          // fight the browser's own momentum.
          if (event.pointerType !== "mouse" || event.button !== 0) return;
          if ((event.target as HTMLElement).closest("[data-no-pan]")) return;
          const el = scrollRef.current;
          if (!el) return;
          didPan.current = false;
          pan.current = {
            x: event.clientX,
            y: event.clientY,
            left: el.scrollLeft,
            top: el.scrollTop,
          };
        }}
      >
        <div
          className="relative m-auto"
          style={{ width: graph.width, height: graph.height }}
        >
          <svg
            width={graph.width}
            height={graph.height}
            viewBox={`0 0 ${graph.width} ${graph.height}`}
            className="block"
          >
            {/* Layer bands, behind everything. Inert: every layer is drawn in
                full, so there is nothing for a click to select. */}
            <g>
              {graph.layers.map((band) => (
                <g key={band.index}>
                  <rect
                    x={band.x - 24}
                    y={18}
                    width={band.width}
                    height={graph.height - 36}
                    rx={10}
                    className={
                      band.index % 2 === 0
                        ? "fill-slate-100/50"
                        : "fill-transparent"
                    }
                  />
                  {/* One `text` with two `tspan`s rather than two `text`s at
                      fixed x. "Layer 12" is four times the width of "L12" and
                      grows with the layer count, so a hardcoded x for the kind
                      beside it collides on wide indices. `dx` flows from
                      wherever the first span actually ended. */}
                  <text x={band.x - 12} y={34}>
                    <tspan className="fill-slate-600 font-mono text-[10px] font-bold">
                      {`Layer ${band.index}`}
                    </tspan>
                    {bandDetail(band) && (
                      <tspan dx={8} className="fill-slate-400 text-[9px]">
                        {bandDetail(band)}
                      </tspan>
                    )}
                  </text>
                </g>
              ))}
            </g>

            {/* Edges. Carried the low half of a two-layer treatment until the
                animated one above it was removed, so the weights here are set
                to read on their own. */}
            <g fill="none" strokeLinecap="round" opacity={focused ? 0.35 : 1}>
              {graph.edges.map((edge) => (
                <path
                  key={edge.id}
                  d={edge.path}
                  className={EDGE_STROKE[edge.role]}
                  strokeWidth={edge.kind === "spine" ? 2.25 : 1.5}
                  opacity={
                    edge.dimmed ? 0.14 : edge.kind === "spine" ? 0.5 : 0.42
                  }
                />
              ))}
            </g>

            {/* Glyphs, then names, in two passes: a name overlaps its neighbours
                and their targets, so all of them are painted after all the glyphs.
                See `HookPoint`. Within each pass the point being pointed at goes
                last, since SVG paints in document order and has no `z-index`, and
                the point you are reading was otherwise drawn underneath its
                neighbour's name about half the time. */}
            <g>
              {orderedNodes.map((node) => (
                <HookPoint
                  key={node.id}
                  node={node}
                  engineId={engineId}
                  traits={traits}
                  active={shown?.node.id === node.id}
                  locked={locked?.node.id === node.id && locked.pane === pane}
                  search={search}
                  differs={diff?.has(node.id) ?? false}
                  faded={focused && !diff?.has(node.id)}
                  flashToken={
                    [node.point, ...node.alsoKnownAs].some((point) =>
                      flash.points.has(point),
                    )
                      ? flash.token
                      : null
                  }
                  onHover={show}
                  onSelect={select}
                />
              ))}
            </g>

            <g>
              {orderedNodes.map((node) => (
                <HookLabel
                  key={node.id}
                  node={node}
                  engineId={engineId}
                  traits={traits}
                  active={shown?.node.id === node.id}
                  compact={compact}
                  search={search}
                  differs={diff?.has(node.id) ?? false}
                  faded={focused && !diff?.has(node.id)}
                  onHover={show}
                  onSelect={select}
                />
              ))}
            </g>
          </svg>

          {/* Keyed by node, so moving between points crossfades rather than
              snapping the card from one place to the next. */}
          <AnimatePresence>
            {shown?.pane === pane && (
              <ActivePopover
                key={shown.node.id}
                node={shown.node}
                graph={graph}
                traits={traits}
                engineId={engineId}
                hfId={hfId}
                view={view}
                narrow={narrow}
                onEnter={keepOpen}
                onLeave={() => show(null)}
              />
            )}
          </AnimatePresence>
        </div>
      </div>

      {scrubber && !narrow && (
        <Scrubber
          graph={graph}
          totalLayers={totalLayers}
          scrollLeft={view.left}
          viewportWidth={view.width}
          onScrollTo={scrollTo}
        />
      )}
    </div>
  );
}

/** Clear of the point's mark, whose capsule is 28 wide. */
const POPOVER_GAP = 20;

/** The margin the card keeps from the edge of the window. */
const EDGE = 8;

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), Math.max(low, high));
}

/**
 * Beside the point on a wide screen; above or below it on a narrow one.
 *
 * Beside is the better placement and reading order is why: the diagram flows left
 * to right, a card to the right of a point is where the eye is already going, and
 * a consistent side means the card lands in the same place relative to every
 * point rather than above some and below others. It also leaves the point, and
 * the ones stacked with it, uncovered. None of that survives a phone, where a
 * card wide enough to hold a code snippet is most of the screen and "beside"
 * would put it over the diagram either way. There it goes under the point and
 * flips above when the bottom is closer, which is the one axis a phone has to
 * spare.
 *
 * **Positioned in the window, in a layer over everything, not inside the pane it
 * belongs to.** In compare mode a pane is half the screen tall, and a card drawn
 * inside one was clipped by that pane's scroll box — so the tallest cards, which
 * are the ones worth reading, were cut off at the fold while half the window sat
 * empty below them. The card is not part of either drawing; it is about a point
 * in one of them, and it should use the window the same way a menu does.
 *
 * That is what the arithmetic below converts: `node.x/y` are canvas coordinates,
 * and `view` carries where the pane is on screen and how far it is scrolled, which
 * together put the point in window coordinates.
 *
 * The height is measured rather than assumed. A card runs from four lines to a
 * full page depending on how many traits bear on the point, and a fixed guess
 * would either waste room or put a tall one off the bottom.
 */
function ActivePopover({
  node,
  graph,
  traits,
  engineId,
  hfId,
  view,
  narrow,
  onEnter,
  onLeave,
}: {
  node: GraphNode;
  graph: Graph;
  traits: Set<TraitId>;
  engineId: EngineId;
  hfId?: string;
  view: {
    left: number;
    top: number;
    width: number;
    height: number;
    rectLeft: number;
    rectTop: number;
  };
  narrow: boolean;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [cardHeight, setCardHeight] = useState(0);
  // The snippets are most of the card's height, so the stack it is written in
  // has to re-measure it the way a trait change does.
  useLayoutEffect(() => {
    setCardHeight(cardRef.current?.offsetHeight ?? 0);
  }, [node.id, traits, engineId, narrow]);

  const isMoe =
    graph.layers.find((band) => band.index === node.layer)?.isMoe ?? false;

  // Read during render rather than tracked: this component only exists after a
  // pointer event, so it never renders on the server, and the resize that would
  // change these also resizes the pane and re-measures `view`.
  const winW = window.innerWidth;
  const winH = window.innerHeight;

  const width = narrow
    ? Math.min(POPOVER_NARROW_WIDTH, winW - 2 * EDGE)
    : POPOVER_WIDTH;

  // A canvas smaller than its container is centred by `m-auto`, which shifts the
  // origin away from scroll zero on that axis.
  const originX = view.rectLeft + Math.max(0, (view.width - graph.width) / 2);
  const originY = view.rectTop + Math.max(0, (view.height - graph.height) / 2);
  const pointX = originX - view.left + node.x;
  const pointY = originY - view.top + node.y;

  let left: number;
  let top: number;
  if (narrow) {
    left = clamp(pointX - width / 2, EDGE, winW - width - EDGE);
    const below = pointY + POPOVER_GAP;
    const above = pointY - POPOVER_GAP - cardHeight;
    top =
      below + cardHeight <= winH - EDGE
        ? below
        : above >= EDGE
          ? above
          : clamp(below, EDGE, winH - cardHeight - EDGE);
  } else {
    const toRight = pointX + POPOVER_GAP;
    const toLeft = pointX - POPOVER_GAP - width;
    left =
      toRight + width <= winW - EDGE
        ? toRight
        : toLeft >= EDGE
          ? toLeft
          : clamp(toRight, EDGE, winW - width - EDGE);
    top = clamp(pointY - cardHeight / 2, EDGE, winH - cardHeight - EDGE);
  }

  return createPortal(
    <motion.div
      // Over the header, which is `z-30`, since a point near the top of the first
      // pane opens a card that reaches it.
      className="pointer-events-none fixed z-40"
      style={{ left, top }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.12, ease: "easeOut" }}
    >
      <div
        ref={cardRef}
        className="pointer-events-auto"
        data-no-pan
        onPointerEnter={onEnter}
        onPointerLeave={onLeave}
      >
        <HookPopover
          node={node}
          traits={traits}
          isMoe={isMoe}
          engineId={engineId}
          hfId={hfId ?? PLACEHOLDER_HF_ID}
          narrow={narrow}
        />
      </div>
    </motion.div>,
    document.body,
  );
}
