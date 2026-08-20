"use client";

/**
 * The architecture picker as a field you type into, for when you know the name.
 *
 * It stands in for the dropdown rather than sitting beside it, so the band does
 * not grow a control while you use it, and the list below stays open the whole
 * time: typing narrows it, and what is left is still a list you can pick from.
 * A field that only accepted an exact name would be worse than the dropdown it
 * replaced — the point is to get close and then look.
 *
 * Two kinds of choice, deliberately different. The first match is selected as
 * you type, so the diagram follows the field and three keystrokes are usually
 * enough; that is *provisional* and keeps search mode open. Committing — Enter,
 * or a click on a row — hands the band back to the dropdown, because at that
 * point you have what you came for and a text field is the wrong thing to leave
 * in front of you.
 *
 * Matching reads the example checkpoints as well as the label, since "gemma-3"
 * and "Qwen3-Next" are how these are known outside this app, and the class name,
 * which is what someone arriving from a `config.json` has in hand.
 *
 * Dismissal is not here. `PaneHeader` owns it, because the toggle that opens this
 * is outside it and a click on that toggle has to count as inside.
 */

import { Search, X } from "lucide-react";
import { useMemo, useState } from "react";

import { ARCHITECTURES, displayModel } from "@/data/architectures";
import type { Architecture } from "@/lib/types";
import { cn } from "@/lib/utils";

interface Props {
  value: string | null;
  /** Provisional or committed, both: the diagram follows the highlight. */
  onChange: (id: string) => void;
  /** Leave search mode. Called on commit, on Escape, and on an outside click. */
  onClose: () => void;
  /** The other pane's choice, which this one may not land on. */
  exclude?: string | null;
  className?: string;
}

function optionsFor(query: string, exclude: string | null): Architecture[] {
  const term = query.trim().toLowerCase();
  return ARCHITECTURES.filter((arch) => {
    if (arch.id === exclude) return false;
    if (!term) return true;
    const haystack = [arch.label, arch.id, ...arch.exampleModels]
      .join(" ")
      .toLowerCase();
    return haystack.includes(term);
  });
}

export function ArchitectureSearch({
  value,
  onChange,
  onClose,
  exclude = null,
  className,
}: Props) {
  const [query, setQuery] = useState("");
  const [at, setAt] = useState(0);

  const found = useMemo(() => optionsFor(query, exclude), [query, exclude]);

  // Clamped rather than reset: a highlight past the end of a narrower list would
  // otherwise index nothing and Enter would do nothing.
  const index = Math.min(at, Math.max(found.length - 1, 0));

  const provisional = (next: number) => {
    setAt(next);
    const pick = found[next];
    if (pick && pick.id !== value) onChange(pick.id);
  };

  const commit = (arch: Architecture | undefined) => {
    if (arch) onChange(arch.id);
    onClose();
  };

  return (
    <div className={cn("relative", className)}>
      {/* Matches the picker it stands in for at both heights. */}
      <div className="flex h-9 items-center gap-x-2 rounded-md border border-sky-300 bg-white pr-2 pl-2.5 shadow-sm ring-2 ring-sky-500/20 sm:h-11">
        <Search className="h-4 w-4 shrink-0 text-slate-400" />
        <input
          autoFocus
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            const next = optionsFor(event.target.value, exclude);
            setAt(0);
            if (next[0] && next[0].id !== value) onChange(next[0].id);
          }}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              provisional(Math.min(index + 1, found.length - 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              provisional(Math.max(index - 1, 0));
            } else if (event.key === "Enter") {
              event.preventDefault();
              commit(found[index]);
            } else if (event.key === "Escape") {
              event.preventDefault();
              onClose();
            }
          }}
          placeholder="Type a name, a checkpoint or a class"
          // Not "search architectures", which is the toggle that opened this:
          // two controls under one name is a coin toss from a screen reader.
          aria-label="Architecture name, checkpoint or class"
          className="min-w-0 flex-1 bg-transparent font-mono text-xs font-medium text-slate-700 placeholder:font-sans placeholder:text-[11px] placeholder:text-slate-400 focus:outline-none"
        />
        <button
          type="button"
          onClick={onClose}
          aria-label="Close search"
          className="shrink-0 cursor-pointer rounded text-slate-400 transition-colors hover:text-slate-600"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Wider than the field, because a checkpoint id is longer than the label
          above it. The viewport cap is `76vw` rather than the full width less a
          margin: the field itself starts a toggle-width in from the left edge,
          and the list is anchored to the field. */}
      <div className="absolute top-[calc(100%+4px)] left-0 z-50 w-[min(320px,76vw)] overflow-hidden rounded-md border border-slate-200 bg-white shadow-[0px_10px_38px_-10px_rgba(22,23,24,0.35),0px_10px_20px_-15px_rgba(22,23,24,0.2)]">
        {found.length === 0 ? (
          <p className="px-3 py-2.5 text-[11px] text-slate-400">
            Nothing matches {`"${query.trim()}"`}. The list is architectures,
            not checkpoints — try a family name.
          </p>
        ) : (
          <div
            // Short enough to clear the bottom of the window from the lower
            // pane's band, which sits at half the viewport in compare mode.
            className="thin-scrollbar max-h-[min(320px,34dvh)] overflow-y-auto"
          >
            {found.map((arch, i) => (
              <button
                key={arch.id}
                type="button"
                onClick={() => commit(arch)}
                onPointerEnter={() => setAt(i)}
                className={`flex w-full cursor-pointer flex-col items-start gap-y-0.5 border-b border-slate-100 px-3 py-2 text-left last:border-b-0 ${
                  i === index ? "bg-sky-50" : "hover:bg-slate-50"
                }`}
              >
                <span className="font-mono text-xs font-medium text-sky-700">
                  {arch.label}
                </span>
                <span className="text-[10px] leading-tight text-slate-400">
                  {arch.exampleModels.slice(0, 2).map(displayModel).join(", ")}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
