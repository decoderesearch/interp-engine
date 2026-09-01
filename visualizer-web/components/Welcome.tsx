"use client";

/**
 * The introduction, and the button that reopens it.
 *
 * One component rather than two mounted side by side, for the reason
 * `AskRizLauncher` is one: which slide is showing and whether the dialog is open
 * is the only state either half needs, and splitting it would push that state
 * into `page.tsx`, which has no other reason to know a tour exists.
 *
 * It opens by itself on a first visit and never again — `lib/firstVisit.ts` holds
 * that, and holds why it is a store rather than an effect. The circled question
 * mark in the header is the way back to it, and the reason the dialog is allowed
 * to be dismissed with one press: nothing here is behind it.
 *
 * Five slides, and the first three are the repository's own claims with the
 * evidence attached rather than prose about them — the throughput chart and the
 * point table are the same components the caption's hover cards open, from
 * `components/evidence/`. The last two are the only ones about this page, and
 * they are last on purpose: what the diagram is for is a question about
 * interp-engine, and it does not read as an answer until the three before it
 * have been made.
 *
 * A Radix `Tabs` inside the dialog rather than a hand-rolled index. The tab row
 * is both the progress indicator and the way back to a slide already read, and
 * as tabs it gets the roles and the arrow keys that a row of buttons would need
 * hand-written. Its content unmounts when it is not showing, which is what keeps
 * the 2.3 MB recording on the last slide from being fetched by a reader who
 * closed the dialog on the first.
 */

import { ArrowLeft, ArrowRight, CircleQuestionMark, XIcon } from "lucide-react";
import Image from "next/image";
import { Dialog, Tabs } from "radix-ui";
import { useCallback, useRef, useState, type ReactNode } from "react";

import { GithubMark } from "@/components/GithubMark";
import { PointGrid } from "@/components/evidence/PointSupport";
import { Throughput } from "@/components/evidence/Throughput";
import { ALL_POINTS } from "@/data/points";
import {
  DEMO_GIF,
  DEMO_GIF_HEIGHT,
  DEMO_GIF_WIDTH,
  SIZER_VIDEO,
} from "@/lib/assets";
import { markVisited, useFirstVisit } from "@/lib/firstVisit";
import { cn } from "@/lib/utils";

/**
 * The two calls the third slide shows. Imports are left off on purpose: the
 * lede already said the model is one line, and a preamble of names is not the
 * thing a newcomer is here to copy.
 */
const READING = `model = load_model("meta-llama/Llama-3.1-8B")
point = Address("resid_post", 12)
cache = run_with_cache(model, model.to_tokens("Hello, world"), [point])
cache[point]  # [batch, pos, ...]`;

const STEERING = `model = load_model("Qwen/Qwen3-8B")
spec = SteeringSpec(layers={
        10: LayerSteeringSpec(operations=[
            AddSpec(vector=torch.randn(model.d_model), scale=4.0)
        ])
    })
with steer(model, spec):
    for step in generate_stream(model, model.to_tokens("Hello, world"), max_tokens=32):
        print(step.token_str, end="")`;

interface Slide {
  id: string;
  /** The tab label, and the claim the slide is evidence for. */
  label: string;
  /** What the evidence under it is evidence *of*. */
  lede: ReactNode;
  body: (repoUrl: string) => ReactNode;
}

const SLIDES: Slide[] = [
  {
    id: "fast",
    label: "Fast",
    lede: (
      <>
        interp-engine is very fast and scales with concurrency - supporting
        interp methods like capturing and steering at all hook points with
        minimal performance impact.
      </>
    ),
    body: (repoUrl) => <Throughput repoUrl={repoUrl} compact />,
  },
  {
    id: "standardized",
    label: "Standardized",
    lede: (
      <>
        interp-engine standardizes {ALL_POINTS.length} points across all
        architectures, and even natively supports DeepSeek&apos;s multiple
        residual streams.
      </>
    ),
    body: () => <PointGrid />,
  },
  {
    id: "easy",
    label: "Easy to Use",
    lede: (
      <>
        interp-engine is designed as simply as possible. Load the models with
        one line, pick the point, and read it.
      </>
    ),
    body: () => <Reading />,
  },
  {
    id: "page",
    label: "Visualizer",
    lede: (
      <>
        This website (interp-engine.org) hosts docs and examples, an AI helpbot
        (Riz Streem), and an interactive visualizer to show model architectures
        and their points. Here, we scrub through architectures from GPT-2 to
        Gemma 4. You can also click &apos;Compare&apos; to compare models.
      </>
    ),
    body: () => <Demo />,
  },
  {
    id: "sizer",
    label: "GPU Sizer",
    lede: (
      <>
        What GPU do you need to run interp-engine on a model? Use the GPU sizer
        to choose a model, set the performance you want, and get exact GPU
        configs that will fit without OOMing - copy the code instantly.
        There&apos;s also a{" "}
        {/* A new tab, so the tour is still here to come back to: this is the one
            slide whose prose sends the reader somewhere, and a same-tab
            navigation out of a first-visit dialog closes it for good. */}
        <a
          href="/docs/gpu-sizer-api"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GPU sizer API documentation (opens in a new tab)"
          className="text-sky-700 underline decoration-sky-700/30 underline-offset-2 hover:decoration-sky-700"
        >
          GPU sizer API
        </a>
        .
      </>
    ),
    body: () => <SizerDemo />,
  },
];

export function Welcome({
  repoUrl,
  className,
  children,
}: {
  repoUrl: string;
  className?: string;
  /** Shown beside the mark in the phone menu; the header itself stays a glyph. */
  children?: ReactNode;
}) {
  const first = useFirstVisit();
  // `null` while the dialog is still following the first-visit check, which is
  // what lets it open once without an effect writing to state to open it. Any
  // press after that is an answer of its own and outranks the check.
  const [override, setOverride] = useState<boolean | null>(null);
  const [at, setAt] = useState(0);
  const open = override ?? first;
  const content = useRef<HTMLDivElement>(null);

  const close = useCallback(() => {
    markVisited();
    setOverride(false);
  }, []);

  const last = at === SLIDES.length - 1;

  return (
    <Dialog.Root
      open={open}
      onOpenChange={(next) => {
        if (next) {
          // From the top, wherever it was left. Reopening onto slide three of
          // four is reopening onto the middle of an argument.
          setAt(0);
          setOverride(true);
        } else {
          close();
        }
      }}
    >
      {/* `Dialog.Trigger`, not a sibling that writes `open`. The question mark
          sits outside the content, so the click that opens is also a click
          "outside" -- without Trigger, DismissableLayer treats that same event
          as a dismiss and the dialog never appears. That is the returning-visit
          path: first visit still opens it by passing `open`, which is why a
          private window (empty storage) looked fine. */}
      <Dialog.Trigger asChild>
        <button
          type="button"
          aria-label="Tutorial"
          title="Tutorial"
          // Only the invariants here. What the trigger looks like is the
          // header's business -- it is a pill beside the docs button there and
          // a labelled row in the phone menu, and both come from `className`.
          className={cn(
            "flex shrink-0 cursor-pointer items-center transition-colors",
            className,
          )}
        >
          <CircleQuestionMark className="h-[18px] w-[18px] shrink-0 sm:h-4 sm:w-4" />
          {children}
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        {/* The same overlay the Controls sheet uses, at the same weight: this
            is the one thing in the app that is genuinely modal, and a scrim
            dark enough to be a curtain would hide the diagram it is
            introducing. */}
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/10 duration-100 supports-backdrop-filter:backdrop-blur-xs data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        {/* Centred by `inset-0 m-auto` over a fitted height rather than by
            translating half its own size, which is the usual spelling and the
            one that fights the animation: `animate-in` writes the whole
            `transform` in its own keyframe, so a dialog holding its position in
            one would zoom *and* slide in from half a dialog away. */}
        <Dialog.Content
          ref={content}
          // The dialog itself rather than the first thing in it, which Radix
          // would otherwise focus: that is the close button, so a keyboard
          // reader arrives at the way out of the introduction and a mouse
          // reader arrives at a focus ring around an X. The container takes
          // focus instead -- it is what the reader is being handed -- and Tab
          // from there walks the tabs, the slide and the buttons in order.
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            content.current?.focus();
          }}
          className="fixed inset-0 z-50 m-auto flex h-fit max-h-[calc(100dvh-16px)] w-[min(820px,calc(100vw-16px))] flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl focus:outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95"
        >
          {/* Outside the scroll box below it, so the title and the way out of
              the dialog stay put while a slide taller than the window is read
              — which the throughput slide is, on a laptop. */}
          <div className="flex shrink-0 items-start justify-between gap-x-3 px-4 py-2">
            {/* The header's own mark, at the header's own size, so the dialog
                reads as this page introducing itself rather than as a notice
                that happens to be about it. Not a link home: the reader is
                already here, and the way out of a modal is the modal's. */}
            <Image
              src="/ielogo.png"
              alt=""
              width={464}
              height={464}
              priority
              className="mt-0.5 h-8 w-8 shrink-0"
            />

            <div className="min-w-0 flex-1">
              <Dialog.Title className="font-mono text-sm font-semibold text-slate-800">
                interp-engine
              </Dialog.Title>
              <Dialog.Description className="mt-0.5 text-[11px] text-slate-500">
                A fast, standardized, and easy-to-use interpretability engine.
              </Dialog.Description>
            </div>

            <div className="flex shrink-0 items-center gap-x-1">
              <a
                href={repoUrl}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="interp-engine on GitHub (opens in a new tab)"
                title="interp-engine on GitHub"
                className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-900"
              >
                <GithubMark className="h-[18px] w-[18px]" />
              </a>
              <Dialog.Close
                aria-label="Close"
                className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-900"
              >
                <XIcon className="h-[18px] w-[18px]" />
              </Dialog.Close>
            </div>
          </div>

          {/* Not `flex-1`: the dialog is `h-fit`, so a `flex-1 min-h-0` child
              has a basis of 0 and no leftover space to grow into. The slide
              then collapses and the dialog's `overflow-hidden` clips it —
              which is why the Fast / Standardized bodies had no height.
              `min-h-0` stays so this section can shrink when the dialog hits
              `max-h`, and the box below is what actually scrolls. */}
          <Tabs.Root
            value={SLIDES[at].id}
            onValueChange={(id) =>
              setAt(SLIDES.findIndex((slide) => slide.id === id))
            }
            className="flex min-h-0 flex-col"
          >
            {/* Scrolls sideways rather than wrapping: four labels fit from
                `sm` up, and on a phone a second row of tabs would take a
                quarter of the dialog before the slide starts. */}
            <Tabs.List
              loop={false}
              className="no-scrollbar flex shrink-0 gap-x-1 overflow-x-auto border-b border-slate-200 px-4 py-1.5 sm:w-full"
            >
              {SLIDES.map((slide, index) => (
                <Tabs.Trigger
                  key={slide.id}
                  value={slide.id}
                  className="shrink-0 cursor-pointer rounded-full bg-slate-100 px-3 py-1.5 text-center text-[13px] font-medium whitespace-nowrap text-slate-600 transition-colors hover:bg-slate-200 hover:text-slate-800 data-active:bg-sky-700 data-active:text-white data-active:hover:bg-sky-700 sm:min-w-0 sm:flex-1"
                >
                  <span
                    className="mr-1.5 font-mono text-[11px] opacity-60"
                    aria-hidden
                  >
                    {index + 1}
                  </span>
                  {slide.label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>

            {/* One scroll box around all four rather than one per slide, and a
                floor under it, so the footer does not walk up and down the
                window as the slides change height. Same reason this is not
                `flex-1`: under `h-fit` that would zero the height the floor
                is meant to protect. */}
            <div className="thin-scrollbar min-h-0 overflow-y-auto px-4 pt-3 pb-4 sm:min-h-[340px]">
              {SLIDES.map((slide) => (
                <Tabs.Content
                  key={slide.id}
                  value={slide.id}
                  className="focus:outline-none"
                >
                  <p className="text-[12px] leading-relaxed text-slate-600 sm:text-[13px]">
                    {slide.lede}
                  </p>
                  <div className="mt-3">{slide.body(repoUrl)}</div>
                </Tabs.Content>
              ))}
            </div>
          </Tabs.Root>

          <div className="flex shrink-0 gap-x-2 border-t border-slate-200 px-4 py-3">
            <button
              type="button"
              onClick={() => setAt((was) => was - 1)}
              disabled={at === 0}
              className="flex flex-1 cursor-pointer items-center justify-center gap-x-1.5 rounded-md px-3 py-2.5 text-[13px] font-medium text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-40"
            >
              <ArrowLeft className="h-4 w-4" />
              Back
            </button>
            <button
              type="button"
              onClick={() => (last ? close() : setAt((was) => was + 1))}
              className="flex flex-1 cursor-pointer items-center justify-center gap-x-1.5 rounded-md bg-sky-700 px-3 py-2.5 text-[13px] font-medium text-white transition-colors hover:bg-sky-800"
            >
              {last ? "Get Started" : "Next"}
              {!last && <ArrowRight className="h-4 w-4" />}
            </button>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/** The third slide: read, then steer, both as the samples print them. */
function Reading() {
  return (
    <div>
      <Code>{READING}</Code>
      <p className="mt-3 text-[12px] leading-relaxed text-slate-600 sm:text-[13px]">
        Steering is simple too. Define your steering spec and steer either sync
        or async.
      </p>
      <Code className="mt-3">{STEERING}</Code>
    </div>
  );
}

function Code({
  children,
  className,
}: {
  children: string;
  className?: string;
}) {
  return (
    <pre
      className={cn(
        "thin-scrollbar overflow-x-auto rounded-sm bg-slate-50 px-3 pt-2 pb-2.5 font-mono text-[11px] leading-relaxed text-slate-700",
        className,
      )}
    >
      {children}
    </pre>
  );
}

/**
 * The fifth slide, waiting for its recording.
 *
 * Same shape as {@link Demo} so the two read as one pair: a figure holding one
 * piece of media at the column's full width. Swap the placeholder for the `img`
 * or `video` when there is a capture to put here, and put any prose above it in
 * the slide's `lede` rather than in this body.
 */
function SizerDemo() {
  return (
    <>
      <video
        src={SIZER_VIDEO}
        autoPlay
        muted
        loop
        playsInline
        aria-label="GPU Sizer: choose a model, set the performance you want, and get exact GPU configs that will fit without OOMing - copy the code instantly."
        className="mb-3 w-full rounded-md"
      />
    </>
  );
}

/** The fourth slide: the page, doing the two things worth doing to it. */
function Demo() {
  return (
    <figure>
      {/* eslint-disable-next-line @next/next/no-img-element -- `next/image`
          optimizes an animated GIF into a still, and `lib/assets.ts` carries the
          rest of why this one is a plain `img`. */}
      <img
        src={DEMO_GIF}
        width={DEMO_GIF_WIDTH}
        height={DEMO_GIF_HEIGHT}
        alt="The visualizer: the release timeline dragged from one architecture to the next, and a point's card opening under the pointer."
        className="h-auto w-full rounded-md border border-slate-200"
      />
    </figure>
  );
}
