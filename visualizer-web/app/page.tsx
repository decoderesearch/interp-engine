"use client";

import { PanelRight } from "lucide-react";
import Image from "next/image";
import { useEffect, useState } from "react";

import { HeaderIcons } from "@/components/HeaderIcons";
import { Sidebar } from "@/components/Sidebar";
import { Tagline } from "@/components/Tagline";
import { AskRizLauncher } from "@/components/ask/AskRizLauncher";
import { ModeToggle, type Mode } from "@/components/controls/ModeToggle";
import { FlowDiagram } from "@/components/diagram/FlowDiagram";
import { PaneHeader } from "@/components/diagram/PaneHeader";
import { Sizer } from "@/components/sizer/Sizer";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { CUSTOM_ARCHITECTURE_ID, architecture } from "@/data/architectures";
import { trait } from "@/data/traits";
import { traitDiff } from "@/lib/diff";
import { encodeLink, useOpenedWith, type Link } from "@/lib/link";
import { encodeSizerPath, useSizerRoute } from "@/lib/sizer-link";
import { useVisualizer } from "@/lib/state";

const REPO_URL = "https://github.com/decoderesearch/interp-engine";

/**
 * Nothing but the address bar, which is why it is a component of its own.
 *
 * Keyed on the URL rather than threading it into a setter. Neither the query nor
 * the path can be read during the prerender, nor during the hydration render
 * that has to match it, so both arrive one render late — and applying them from
 * there would be a setState in an effect body, which `lib/useHydrated.ts`
 * documents as the thing this app writes around. A remount takes them as initial
 * conditions instead, and only ever happens when there is something to take: a
 * plain visit reads `/` both times and keeps the tree it hydrated.
 */
export default function Page() {
  const link = useOpenedWith();
  const route = useSizerRoute();
  return (
    <Visualizer
      key={`${route.sizing ? encodeSizerPath(route.model) : "/"}${encodeLink(link)}`}
      link={link}
      route={route}
    />
  );
}

function Visualizer({
  link,
  route,
}: {
  link: Link;
  route: { sizing: boolean; model: string };
}) {
  const state = useVisualizer(link);
  const [controls, setControls] = useState(false);
  // Both panes scroll as one, so the point above the pointer in the top diagram
  // is the point above it in the bottom one.
  const [sync, setSync] = useState({ left: 0, top: 0 });

  // Local rather than part of `useVisualizer`: the other two modes are a
  // property of the diagram — which architectures are on screen — and encode as
  // the `vs` query, while this one replaces the diagram and carries a model id
  // in the path instead. `lib/sizer-link.ts` owns that half of the address bar.
  const [sizing, setSizing] = useState(route.sizing);
  const [sizerModel, setSizerModel] = useState(route.model);
  const mode: Mode = sizing ? "sizer" : state.comparing ? "compare" : "single";

  const setMode = (next: Mode) => {
    setSizing(next === "sizer");
    if (next === "single") state.setCompareId(null);
    if (next === "compare") state.startCompare();
  };

  // The path half of the mirror. `useVisualizer` writes the query half and keeps
  // the pathname; this keeps the query, so the two never overwrite each other
  // and the architecture behind the sizer survives a trip through it.
  useEffect(() => {
    const path = sizing ? encodeSizerPath(sizerModel) : "/";
    if (window.location.pathname === path) return;
    try {
      window.history.replaceState(null, "", `${path}${window.location.search}`);
    } catch {
      // A browser refusing the write leaves the address bar stale, which is the
      // smaller of the two failures available here.
    }
  }, [sizing, sizerModel]);

  const current = architecture(state.architectureId);
  const compare = state.compareId ? architecture(state.compareId) : null;
  const isCustom = state.architectureId === CUSTOM_ARCHITECTURE_ID;
  const focusTrait = state.focusTrait ? trait(state.focusTrait) : null;

  // One architecture's examples would be a caption for half the screen, so the
  // notes step aside while two are on it.
  const note = state.comparing ? undefined : current?.note;
  const examples = state.comparing ? [] : (current?.exampleModels ?? []);

  return (
    <main className="flex h-dvh flex-col overflow-hidden">
      {/* Three columns rather than a flex row with an overlaid centre. The
          toggle has to sit on the header's midpoint, not on the midpoint of
          whatever is left beside the wordmark, and `1fr auto 1fr` puts it
          there without taking it out of the flow — which absolute positioning
          would, and this one is interactive. */}
      <header className="relative z-30 grid shrink-0 grid-cols-[1fr_auto] items-center gap-x-3 border-b border-sky-700 bg-white/85 px-3 py-2 backdrop-blur sm:grid-cols-[1fr_auto_1fr]">
        <div className="flex min-w-0 items-center gap-x-3">
          {/* A real navigation, not a client `Link`. The query the page was
              opened with is captured once when `lib/link.ts` is evaluated, and
              the address bar after that is `replaceState` — Next already thinks
              this is `/`, and a client navigation would not re-read the arrival. */}
          {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
          <a
            href="/"
            aria-label="Interp Engine Visualizer home"
            title="Home"
            className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md transition-transform hover:scale-105"
          >
            <Image
              src="/ielogo.png"
              alt=""
              width={464}
              height={464}
              priority
              className="h-8 w-8"
            />
          </a>
          <div className="min-w-0">
            {/* Mono, because it is the package name: what you would type to
                install it, not a product title. */}
            <div className="font-mono text-sm font-semibold whitespace-nowrap text-sky-800">
              interp-engine
            </div>
            {/* Steps aside a breakpoint earlier while comparing, so the
                caption beside it has the room instead. Which of the two to
                drop is a question about the mode, and the mode is not
                something a media query can ask. */}
            <Tagline
              repoUrl={REPO_URL}
              className={state.comparing ? "sm:hidden xl:block" : undefined}
            />
          </div>

          {/* Beside the wordmark rather than under it: the tagline has that
              space now, and stacking a third line would set the header's
              height from a caption that is only present in one mode. */}
          {state.comparing && !sizing && (
            <span className="hidden min-w-0 truncate border-l border-slate-200 pl-3 text-[10px] whitespace-nowrap text-slate-400 lg:block">
              Comparing{" "}
              <span className="font-medium text-slate-600">
                {current?.label ?? "Custom"}
              </span>{" "}
              vs{" "}
              <span className="font-medium text-slate-600">
                {compare?.label}
              </span>
            </span>
          )}
        </div>

        {/* Pulled down past the header's bottom padding and its hairline —
            8px + 1px — so the tabs sit on the edge rather than floating above
            it, and the active one runs straight into the content below. */}
        <div className="-mb-[9px] hidden self-end justify-self-center sm:block">
          <ModeToggle mode={mode} onMode={setMode} />
        </div>

        <div className="flex min-w-0 items-center justify-self-end gap-x-2">
          {/* Only while a difference is being pointed at, and mostly for the
              case where its points are all off the side of a canvas thousands
              of pixels wide: "nothing lit up" and "what lit up is off-screen"
              look identical without it. */}
          {focusTrait && !sizing && (
            <span className="hidden text-[10px] whitespace-nowrap text-slate-500 sm:inline">
              highlighting {state.diff?.size ?? 0} from{" "}
              <span className="font-medium text-slate-700">
                {focusTrait.label}
              </span>
            </span>
          )}

          {/* Everything in it is about the diagram, and in sizer mode there is
              no diagram behind it to change. */}
          {!sizing && (
            <Sheet open={controls} onOpenChange={setControls}>
              <SheetTrigger asChild>
                <button
                  type="button"
                  className="flex cursor-pointer items-center gap-x-1.5 rounded-md border border-slate-300 bg-white px-2 py-2 text-[11px] font-medium text-slate-600 transition-colors hover:bg-slate-50 sm:px-3 lg:hidden"
                >
                  <PanelRight className="h-3.5 w-3.5 text-slate-400" />
                  <span className="sr-only sm:not-sr-only">Controls</span>
                </button>
              </SheetTrigger>
              <SheetContent
                side="right"
                className="w-[min(360px,90vw)] overflow-y-auto"
              >
                <SheetHeader>
                  <SheetTitle className="text-sm">Controls</SheetTitle>
                  <SheetDescription className="text-[11px]">
                    What the diagram shows, and what it is called.
                  </SheetDescription>
                </SheetHeader>
                <Sidebar state={state} note={note} examples={examples} touch />
              </SheetContent>
            </Sheet>
          )}

          <AskRizLauncher state={state} />

          {/* Tour, then docs, then the outbound links. On a phone this is the
              hamburger; Ask Riz stays the floating button of its own. */}
          <HeaderIcons repoUrl={REPO_URL} />
        </div>
      </header>

      {/* On a phone the header cannot hold the mode toggle beside the wordmark
          and the icons, so it floats at the bottom centre — the same intercom
          treatment Ask Riz gets in the opposite corner. */}
      <div className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 sm:hidden">
        <div className="rounded-md shadow-lg shadow-slate-900/20">
          <ModeToggle mode={mode} onMode={setMode} floating />
        </div>
      </div>

      {sizing ? (
        /* The whole content area, sidebar included: the sidebar sets what the
           diagram draws, and there is no diagram here.

           From `lg` the scrolling belongs to the two columns rather than to this
           box, so each one ends on the bottom edge of the window and a long
           result list does not push the model field off the top. Below `lg` the
           columns are stacked and there is only one thing to scroll, so the box
           takes it back. No padding of its own either way — the columns carry
           their own, which is what lets the left one paint white to both edges.

           Full width rather than a centred `max-w`: the left column is a white
           panel, and a centred track leaves a slate gutter beside it on a wide
           screen, which reads as the panel having stopped short of the edge. */
        <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto lg:overflow-hidden">
          <div className="w-full pb-24 sm:pb-5 lg:h-full lg:pb-0">
            <Sizer initialModel={route.model} onModelChange={setSizerModel} />
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className="flex min-h-0 min-w-0 flex-1 flex-col">
            <div className="flex min-h-0 flex-1 flex-col">
              <PaneHeader
                value={isCustom ? null : state.architectureId}
                placeholder={isCustom ? "Custom" : "Pick one"}
                onChange={(id) => id && state.selectArchitecture(id)}
                exclude={state.compareId}
                compare={
                  state.comparing && state.compareTraits
                    ? {
                        selfLabel: current?.label ?? "Custom",
                        otherLabel: compare?.label ?? "the other",
                        traits: state.traits,
                        deltas: traitDiff(state.traits, state.compareTraits),
                        onFocus: state.setFocusTrait,
                      }
                    : undefined
                }
              />

              <div className="min-h-0 flex-1">
                <FlowDiagram
                  state={state}
                  graph={state.graph}
                  traits={state.traits}
                  totalLayers={state.dims.layers}
                  hfId={current?.exampleModels[0]}
                  pane={0}
                  lead
                  diff={state.diff}
                  scrubber={!state.comparing}
                  sync={state.comparing ? sync : undefined}
                  onSync={state.comparing ? setSync : undefined}
                />
              </div>
            </div>

            {state.comparing && state.compareGraph && state.compareTraits && (
              <>
                <div className="h-px shrink-0 bg-slate-200" />
                <div className="flex min-h-0 flex-1 flex-col">
                  <PaneHeader
                    value={state.compareId}
                    placeholder="Pick one"
                    onChange={state.setCompareId}
                    exclude={isCustom ? null : state.architectureId}
                    compare={{
                      selfLabel: compare?.label ?? "the other",
                      otherLabel: current?.label ?? "Custom",
                      traits: state.compareTraits,
                      deltas: traitDiff(state.compareTraits, state.traits),
                      onFocus: state.setFocusTrait,
                    }}
                  />
                  <div className="min-h-0 flex-1">
                    <FlowDiagram
                      state={state}
                      graph={state.compareGraph}
                      traits={state.compareTraits}
                      totalLayers={state.dims.layers}
                      hfId={compare?.exampleModels[0]}
                      pane={1}
                      diff={state.diff}
                      sync={sync}
                      onSync={setSync}
                    />
                  </div>
                </div>
              </>
            )}
          </div>

          <aside className="thin-scrollbar hidden w-[320px] shrink-0 overflow-y-auto border-l border-slate-200 bg-white lg:block">
            <Sidebar state={state} note={note} examples={examples} />
          </aside>
        </div>
      )}
    </main>
  );
}
