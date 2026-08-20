"use client";

/**
 * A minimap of the whole stack under the canvas.
 *
 * The diagram is far wider than any screen, so the scroll bar is the only thing
 * telling you where you are in a deep model — and it says nothing about what is
 * on either side. This draws the bands at scale and lets you drag or click
 * straight to a position.
 */

import { useCallback, useEffect, useRef } from "react";

import type { Graph, Role } from "@/lib/types";

interface Props {
  graph: Graph;
  totalLayers: number;
  scrollLeft: number;
  viewportWidth: number;
  onScrollTo: (left: number) => void;
}

const EDGE_STROKE: Record<Role, string> = {
  resid: "stroke-role-resid",
  attn: "stroke-role-attn",
  mlp: "stroke-role-mlp",
  route: "stroke-role-route",
  global: "stroke-role-global",
};

export function Scrubber({
  graph,
  totalLayers,
  scrollLeft,
  viewportWidth,
  onScrollTo,
}: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  /** Where inside the window the pointer grabbed it, so a drag does not jump. */
  const grabOffset = useRef(0);

  const pct = (value: number) => `${(value / graph.width) * 100}%`;
  const windowWidth = Math.min(viewportWidth, graph.width);
  const scrollable = graph.width > viewportWidth + 1;

  const moveTo = useCallback(
    (clientX: number, offset: number) => {
      const track = trackRef.current;
      if (!track) return;
      const rect = track.getBoundingClientRect();
      onScrollTo(((clientX - rect.left - offset) / rect.width) * graph.width);
    },
    [graph.width, onScrollTo],
  );

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (!dragging.current) return;
      event.preventDefault();
      moveTo(event.clientX, grabOffset.current);
    };
    const onUp = () => {
      dragging.current = false;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [moveTo]);

  return (
    <div className="flex h-11 shrink-0 items-center gap-x-2.5 border-t border-slate-200 bg-white px-3">
      <span className="shrink-0 text-[9px] font-medium tracking-wide text-slate-400 tabular-nums uppercase">
        L0–{totalLayers - 1}
      </span>
      <div
        ref={trackRef}
        className={`relative h-7 flex-1 overflow-hidden rounded-sm border border-slate-200 bg-slate-50 ${
          scrollable ? "cursor-pointer" : ""
        }`}
        onPointerDown={(event) => {
          const track = trackRef.current;
          if (!scrollable || !track) return;
          const rect = track.getBoundingClientRect();
          const windowLeft = (scrollLeft / graph.width) * rect.width;
          const windowPx = (windowWidth / graph.width) * rect.width;
          const x = event.clientX - rect.left;
          const grabbedWindow = x >= windowLeft && x <= windowLeft + windowPx;
          // Grabbing the window preserves its offset; clicking the track centres it.
          grabOffset.current = grabbedWindow ? x - windowLeft : windowPx / 2;
          dragging.current = true;
          moveTo(event.clientX, grabOffset.current);
        }}
      >
        {/* The diagram itself, squashed to the strip. Non-scaling strokes keep
            it legible as a texture at 1/20th the height. */}
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${graph.width} ${graph.height}`}
          preserveAspectRatio="none"
          aria-hidden
        >
          {graph.edges.map((edge) => (
            <path
              key={edge.id}
              d={edge.path}
              fill="none"
              className={EDGE_STROKE[edge.role]}
              strokeWidth={edge.kind === "spine" ? 1.5 : 1}
              vectorEffect="non-scaling-stroke"
              opacity={edge.dimmed ? 0.12 : 0.55}
            />
          ))}
          {/* Boundaries, so the strip can be counted as layers rather than read
              as one continuous smear. */}
          {graph.layers.map((band) => (
            <line
              key={band.index}
              x1={band.x - 24}
              x2={band.x - 24}
              y1={0}
              y2={graph.height}
              className="stroke-slate-300"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
        {scrollable && (
          <div
            className="pointer-events-none absolute inset-y-0 rounded-sm border-2 border-sky-600 bg-sky-500/10"
            style={{ left: pct(scrollLeft), width: pct(windowWidth) }}
          />
        )}
      </div>
    </div>
  );
}
