"use client";

/**
 * Runs the snippet somewhere, which on a page with no runtime means Colab.
 *
 * One press, two halves: the snippet goes on the clipboard here, and the notebook that
 * opens installs what that snippet's backend needs and ends with the cell to paste it
 * into. The split is Colab's — see `lib/colab.ts` for why the cells cannot ride along in
 * the URL — and the template is written so the paste is the next thing a reader sees.
 *
 * An anchor rather than a button, so the browser opens the tab. `window.open` after
 * awaiting the clipboard is the shape a popup blocker cancels, since by then the click is
 * over. For the same reason the write is started inside the gesture and not waited for,
 * and there is no `Copied` state: this tab loses focus at once, so the only place a
 * confirmation can be read is the notebook, which says what to paste and what to do if
 * the clipboard turns out to be empty.
 */

import { NotebookPen } from "lucide-react";

import { SNIPPET_ACTION_CLASS } from "@/components/CopyButton";
import type { Variant } from "@/data/snippets";
import { colabTemplateUrl } from "@/lib/colab";
import { cn } from "@/lib/utils";

export function NotebookButton({
  text,
  variant,
  className,
}: {
  text: string;
  /** Chooses the template, which is to say the install. */
  variant: Variant;
  className?: string;
}) {
  return (
    <a
      href={colabTemplateUrl(variant)}
      target="_blank"
      rel="noreferrer"
      title="Copy this snippet and open a Colab notebook that installs interp-engine"
      onClick={() => {
        // Optional, because an insecure origin has no `clipboard` at all; the chain
        // short-circuits there rather than throwing past the navigation.
        void navigator.clipboard?.writeText(text).catch(() => undefined);
      }}
      className={cn(SNIPPET_ACTION_CLASS, className)}
    >
      <NotebookPen className="h-2.5 w-2.5" />
      Notebook
    </a>
  );
}
