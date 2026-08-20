"use client";

/**
 * Filter the diagram by hook name.
 *
 * Searches the names the toggle beside it is currently printing, so switching
 * stacks changes what a given term finds — which is the point: `hook_z` is a
 * TransformerLens question and `attentions_output` is an nnterp one.
 */

import { Search, X } from "lucide-react";

import type { VisualizerState } from "@/lib/state";

interface Props {
  state: VisualizerState;
}

export function SearchField({ state }: Props) {
  const { query, setQuery, matchCount } = state;

  return (
    <div className="relative flex w-full items-center">
      <Search className="pointer-events-none absolute left-2 h-3.5 w-3.5 text-slate-400" />
      <input
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") setQuery("");
        }}
        placeholder="hook name"
        aria-label="Search hook names"
        className="w-full rounded-md border border-slate-300 bg-white py-1.5 pr-12 pl-7 font-mono text-[10px] text-slate-700 placeholder:font-sans placeholder:text-slate-400 focus:border-sky-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500/40 sm:py-1 [&::-webkit-search-cancel-button]:hidden"
      />
      {matchCount !== null && (
        <div className="absolute right-1.5 flex items-center gap-x-1">
          <span
            className={`text-[9px] font-semibold tabular-nums ${
              matchCount === 0 ? "text-amber-600" : "text-slate-400"
            }`}
          >
            {matchCount}
          </span>
          <button
            type="button"
            onClick={() => setQuery("")}
            aria-label="Clear search"
            className="cursor-pointer rounded-sm p-0.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}
