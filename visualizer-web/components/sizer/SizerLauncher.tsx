"use client";

/**
 * The button, and the modal it owns.
 *
 * One component rather than two mounted side by side, for the reason `Welcome` and `AskRizLauncher`
 * are each one: open/closed is the only state either half needs, and splitting it would push that
 * state into `page.tsx`, which has no reason to know a sizer exists.
 *
 * A centred dialog rather than the Controls sheet. The sheet is for settings that change what the
 * diagram behind it shows, so it is narrow and the diagram stays visible on purpose; this is a tool
 * with its own question and its own answer, nothing behind it is relevant while it is open, and its
 * content is two-column-wide tables of numbers. Same overlay weight as the tour, which is the other
 * genuinely modal thing here.
 *
 * `Dialog.Trigger` rather than a sibling button writing `open`: the button sits outside the content,
 * so the click that opens is also a click "outside", and without Trigger the dismissable layer treats
 * that same event as a dismiss and the dialog never appears.
 */

import { Cpu, XIcon } from "lucide-react";
import { Dialog } from "radix-ui";
import { useRef, useState, type ReactNode } from "react";

import { Sizer } from "@/components/sizer/Sizer";
import { cn } from "@/lib/utils";

export function SizerLauncher({
  className,
  children,
}: {
  className?: string;
  /** Shown beside the glyph in the phone menu; the header itself stays a glyph. */
  children?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const content = useRef<HTMLDivElement>(null);

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <Dialog.Trigger asChild>
        <button
          type="button"
          aria-label="GPU Sizer"
          title="GPU Sizer"
          className={cn(
            "flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-900",
            className,
          )}
        >
          <Cpu className="h-[18px] w-[18px] shrink-0" />
          {children}
        </button>
      </Dialog.Trigger>

      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/10 duration-100 supports-backdrop-filter:backdrop-blur-xs data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        {/* Centred by `inset-0 m-auto` over a fitted height rather than by translating half its own
            size, which is the spelling that fights the animation: `animate-in` writes the whole
            `transform` in its own keyframe, so a dialog holding its position in one would zoom *and*
            slide in from half a dialog away. */}
        <Dialog.Content
          ref={content}
          // The dialog itself rather than the first thing in it, which Radix would otherwise focus:
          // that is the model field, and a keyboard popping up over a tool nobody has read yet is not
          // what opening it should look like. Tab from here reaches the field first anyway.
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            content.current?.focus();
          }}
          className="fixed inset-0 z-50 m-auto flex h-fit max-h-[calc(100dvh-16px)] w-[min(860px,calc(100vw-16px))] flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl focus:outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95"
        >
          {/* Outside the scroll box below it, so the title and the way out stay put while a long
              list of cards is read. */}
          <div className="flex shrink-0 items-start justify-between gap-x-3 border-b border-slate-200 px-4 py-2.5">
            <div className="min-w-0">
              <Dialog.Title className="font-heading text-sm font-semibold text-slate-800">
                GPU Sizer
              </Dialog.Title>
              <Dialog.Description className="mt-0.5 text-[11px] text-slate-500">
                Find a config that gets the best performance for your model
                without OOMing?
              </Dialog.Description>
            </div>
            <Dialog.Close
              aria-label="Close"
              className="flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-md text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-900"
            >
              <XIcon className="h-[18px] w-[18px]" />
            </Dialog.Close>
          </div>

          <div className="thin-scrollbar min-h-0 overflow-y-auto px-4 py-3">
            <Sizer />
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
