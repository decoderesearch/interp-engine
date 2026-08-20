"use client";

/**
 * The header's icon cluster: the tour, the docs, and the three outbound links.
 *
 * One copy, not two. The tour's dialog has to stay mounted — a first visit
 * opens it by itself, and a Radix popover that unmounted this tree on close
 * would take the dialog with it. So the hamburger on a phone is just a class
 * on this row: hidden until opened, then a labelled list under the button.
 * From `sm` up the same items sit in the header as glyphs and the hamburger
 * is not shown.
 *
 * Ask Riz is not in here. On a phone it is already the floating button, and
 * that is the placement that stays.
 */

import { BookText, Mail, Menu } from "lucide-react";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";

import { GithubMark } from "@/components/GithubMark";
import { Welcome } from "@/components/Welcome";

const NEURONPEDIA_URL = "https://www.neuronpedia.org";
const CONTACT_EMAIL = "johnny@neuronpedia.org";
const CONTACT_SUBJECT = "interp-engine";

const ICON =
  "flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-900";

/** Icon in the header; labelled row in the phone menu. */
const ITEM = `${ICON} max-sm:h-auto max-sm:w-full max-sm:justify-start max-sm:gap-x-2.5 max-sm:px-2.5 max-sm:py-1.5 max-sm:text-[12px] max-sm:font-medium max-sm:text-slate-700`;

function Label({ children }: { children: string }) {
  return <span className="sm:hidden">{children}</span>;
}

export function HeaderIcons({ repoUrl }: { repoUrl: string }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (root.current?.contains(event.target as Node)) return;
      setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={root} className="relative flex items-center">
      <button
        type="button"
        onClick={() => setOpen((was) => !was)}
        aria-expanded={open}
        aria-controls="header-icons"
        aria-label="More"
        title="More"
        className={`${ICON} sm:hidden`}
      >
        <Menu className="h-[18px] w-[18px]" />
      </button>

      <div
        id="header-icons"
        onClick={(event) => {
          if ((event.target as HTMLElement).closest("a, button")) setOpen(false);
        }}
        className={
          open
            ? "absolute top-full right-0 z-40 mt-1 flex min-w-[11rem] flex-col rounded-md border border-slate-200 bg-white p-1 shadow-lg sm:static sm:mt-0 sm:min-w-0 sm:flex-row sm:border-0 sm:bg-transparent sm:p-0 sm:shadow-none"
            : "hidden sm:flex sm:items-center"
        }
      >
        <Welcome repoUrl={repoUrl} className={ITEM}>
          <Label>Tutorial</Label>
        </Welcome>

        {/* A plain anchor, and it has to be: `/docs` is a separate Docusaurus
            build served out of `public/docs`, so there is no route for the
            client router to navigate to. A `Link` here would prefetch an RSC
            payload that does not exist. */}
        <a href="/docs" aria-label="Docs" title="Docs" className={ITEM}>
          <BookText className="h-[18px] w-[18px] shrink-0" />
          <Label>Docs</Label>
        </a>

        <a
          href={NEURONPEDIA_URL}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Neuronpedia (opens in a new tab)"
          title="Neuronpedia"
          className={ITEM}
        >
          {/* The mark is already its own brand blue, so it cannot dim to
              slate on rest the way the GitHub path beside it does. Opacity
              is what keeps the two reading as one row of links. */}
          <Image
            src="/neuronpedia.png"
            alt=""
            width={149}
            height={149}
            className="h-[18px] w-[18px] shrink-0 opacity-70"
          />
          <Label>Neuronpedia</Label>
        </a>

        {/* No `target`: a mailto in a new tab leaves an empty one behind
            when the mail client takes over. */}
        <a
          href={`mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(CONTACT_SUBJECT)}`}
          aria-label="Contact"
          title="Contact"
          className={ITEM}
        >
          <Mail className="h-[18px] w-[18px] shrink-0" />
          <Label>Contact</Label>
        </a>

        <a
          href={repoUrl}
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub (opens in a new tab)"
          title="GitHub"
          className={ITEM}
        >
          <GithubMark className="h-[18px] w-[18px] shrink-0" />
          <Label>GitHub</Label>
        </a>
      </div>
    </div>
  );
}
