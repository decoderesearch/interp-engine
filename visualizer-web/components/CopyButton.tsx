"use client";

/**
 * Puts a snippet on the clipboard.
 *
 * Shared by the hook card and Riz's answers, which want it for the same
 * reason: both are transient surfaces a drag-select cannot safely leave. The
 * card closes as soon as the pointer leaves it, and an answer scrolls under a
 * composer while it is still being written.
 *
 * Silent when the clipboard refuses (an insecure origin, or a browser that
 * declines): claiming a copy that did not happen is worse than no feedback,
 * since the failure only surfaces at the paste.
 */

import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/**
 * The treatment for one action in the row under a snippet, exported because
 * `NotebookButton` sits next to this one there and a second copy of these classes
 * would drift into a row of two buttons that do not match.
 */
export const SNIPPET_ACTION_CLASS =
  "flex shrink-0 cursor-pointer items-center gap-x-1 rounded-sm px-1 py-0.5 text-[9px] text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600";

export function CopyButton({
  text,
  className,
  iconClassName,
  label = "Copy",
}: {
  text: string;
  /** Overrides the type scale. The hook card's 9px is the default. */
  className?: string;
  /** Overrides the glyph, which does not scale with the label on its own. */
  iconClassName?: string;
  label?: string;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1400);
    return () => clearTimeout(timer);
  }, [copied]);

  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
        } catch {
          setCopied(false);
        }
      }}
      className={cn(SNIPPET_ACTION_CLASS, className)}
    >
      {copied ? (
        <Check className={cn("h-2.5 w-2.5", iconClassName)} />
      ) : (
        <Copy className={cn("h-2.5 w-2.5", iconClassName)} />
      )}
      {copied ? "Copied" : label}
    </button>
  );
}
