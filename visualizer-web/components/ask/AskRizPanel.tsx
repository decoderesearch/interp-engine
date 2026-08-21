"use client";

/**
 * The chat surface.
 *
 * Portalled to `document.body` and positioned `fixed`, the same treatment the
 * hook card gets in `FlowDiagram`, and for the same reason: the header it
 * hangs from is a flex row inside a `overflow-hidden` column, so a panel
 * rendered in place would be clipped to the height of the row that opened it.
 * `z-40` puts it over the `z-30` header and under the `z-50` dropdowns, which
 * is the ladder the rest of the app already uses.
 *
 * Not a Radix `Popover`, though one would hand over focus management for free.
 * A popover dismisses on outside click, and the entire point of this panel is
 * to be read while pointing at the diagram behind it -- every click that makes
 * the answer worth having would close it. Escape and the button do the
 * dismissing instead.
 *
 * What the reader currently has on screen travels with each message, passed at
 * the `sendMessage` call rather than configured on the transport. The transport
 * is built once and would close over the state of whichever render built it;
 * the state that matters is the state when the question was asked.
 */

import { DefaultChatTransport, type UIMessage } from "ai";
import { useChat } from "@ai-sdk/react";
import { ArrowUp, RotateCcw, Square, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { MessageBody } from "@/components/ask/Message";
import { architecture } from "@/data/architectures";
import { engine } from "@/data/engines";
import type { ViewerContext } from "@/knowledge/prompt";
import type { VisualizerState } from "@/lib/state";
import { useHydrated } from "@/lib/useHydrated";

/** Openers, chosen to show the four things Riz is actually good at. */
const SUGGESTIONS = [
  "Give me a simple example of how to use interp-engine.",
  "How do I steer the residual stream during generation?",
  "What's the difference between vllm and vllm-static backends?",
  "Read z from layer 12 and take the per-head norms",
  "Why are there multiple resid_post for Deepseek V4?",
  "Where do I find all supported hooks/addresses?",
];

export function AskRizPanel({
  id,
  open,
  onClose,
  state,
  focusOnOpen,
}: {
  id: string;
  open: boolean;
  onClose: () => void;
  state: VisualizerState;
  /** Whether this opening was a press. The launcher is what knows. */
  focusOnOpen: boolean;
}) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/ask" }),
    [],
  );

  const {
    messages,
    sendMessage,
    setMessages,
    status,
    stop,
    error,
    clearError,
  } = useChat({ transport });

  const busy = status === "submitted" || status === "streaming";
  const mounted = useHydrated();

  /** One question, with the diagram it was asked about. */
  const ask = (text: string) => {
    clearError();
    sendMessage({ text }, { body: { context: describe(state) } });
  };

  /**
   * Back to an empty thread.
   *
   * Worth a control of its own rather than leaving it to a page reload, and not
   * only because a reload closes the panel. The cached prefix is the same for
   * every thread, so the history is the one part of the prompt that is billed
   * at full rate and the one part that grows -- twelve messages deep, a
   * question about something else entirely is still carrying every earlier
   * answer. Starting over is the cheap path as well as the clear one.
   */
  const reset = () => {
    if (busy) stop();
    setMessages([]);
    clearError();
    setInput("");
    inputRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  /**
   * Focus on being opened by the reader, and only then.
   *
   * Focusing a text field nobody asked for is the wrong thing twice: on a
   * phone it raises the keyboard over the diagram they came for, and
   * everywhere it takes the caret away from a page whose own shortcuts they
   * may want. So two conditions, and neither is the panel merely being open.
   * The transition keeps a press from re-focusing a panel that is already
   * showing, and `focusOnOpen` keeps the openings that were nobody's press --
   * arriving open, and the phone's opening as the tour closes -- from counting
   * as one.
   */
  const wasOpen = useRef(open);
  useEffect(() => {
    const justOpened = open && !wasOpen.current;
    wasOpen.current = open;
    if (justOpened && focusOnOpen) inputRef.current?.focus();
  }, [open, focusOnOpen]);

  // Follows the answer as it streams. `messages` is a new array on every chunk,
  // so this runs per chunk, which is what makes it follow rather than jump.
  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages, open]);

  const submit = () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    ask(text);
  };

  if (!mounted) return null;

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          id={id}
          role="dialog"
          // Matches the visible title rather than the button's label: a dialog
          // that announces itself as something other than its own heading is
          // the screen-reader equivalent of a mislabelled tab.
          aria-label="Riz Streem"
          initial={{ opacity: 0, y: -6, scale: 0.985 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -6, scale: 0.985 }}
          transition={{ duration: 0.14, ease: "easeOut" }}
          // Anchored to whichever corner its launcher is in: bottom right on a
          // phone, growing up out of the floating button, and below the header
          // from `sm` up where the button is in the header row. `bottom-20`
          // clears the 48px button and its 16px inset. The phone height is
          // shorter again, for a window that is shorter to begin with.
          className="fixed right-3 bottom-20 z-40 flex h-[min(320px,calc(100dvh-7rem))] w-[min(400px,calc(100vw-1.5rem))] flex-col overflow-hidden rounded-lg border border-slate-200 bg-white shadow-xl shadow-slate-900/10 sm:top-12 sm:bottom-auto sm:h-[min(380px,calc(100dvh-4.5rem))]"
        >
          <div className="flex shrink-0 items-center gap-x-2 border-b border-slate-200 px-3 py-2">
            <span className="relative block h-6 w-6 shrink-0 overflow-hidden rounded-full bg-lime-50">
              <Image
                src="/riz.png"
                alt=""
                width={818}
                height={818}
                className="absolute inset-0 h-full w-full scale-[1.35] object-cover object-[50%_38%]"
              />
            </span>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-semibold text-slate-700">
                Ask Riz Streem
              </div>
              <div className="truncate text-[10px] text-slate-400">
                interp-engine, from its own docs
              </div>
            </div>
            {/* Disabled rather than hidden on an empty thread: appearing only
                once there is something to clear would shift the close button
                sideways at the moment the first answer arrives. */}
            <button
              type="button"
              onClick={reset}
              disabled={messages.length === 0}
              aria-label="Start over"
              title="Start over"
              className="cursor-pointer rounded-sm p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 disabled:cursor-default disabled:text-slate-200 disabled:hover:bg-transparent"
            >
              <RotateCcw className="h-3.5 w-3.5" />
            </button>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="cursor-pointer rounded-sm p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="thin-scrollbar min-h-0 flex-1 overflow-y-auto px-3 py-2">
            {messages.length === 0 ? (
              <Empty onPick={ask} />
            ) : (
              messages.map((message) => (
                <Turn
                  key={message.id}
                  role={message.role}
                  text={text(message)}
                />
              ))
            )}

            {/* Only before the first token. Once text is arriving, the text is
                the progress indicator and a second one below it is noise. */}
            {status === "submitted" && (
              <div className="my-1.5 text-[11px] text-slate-400">Thinking…</div>
            )}

            {error && <Failure error={error} />}
            <div ref={bottomRef} />
          </div>

          <div className="shrink-0 border-t border-slate-200 p-2">
            <div className="flex items-center gap-x-1.5 rounded-md border border-slate-300 bg-white px-2 py-1.5 focus-within:border-sky-500 focus-within:ring-2 focus-within:ring-sky-500/40">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  // Enter sends. A chat composer that needs a button click for
                  // every message is a chat composer nobody uses twice; the
                  // newline keeps its usual modifier.
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submit();
                  }
                }}
                rows={1}
                placeholder="What models are supported?"
                aria-label="Ask Riz a question"
                className="thin-scrollbar max-h-24 min-w-0 flex-1 flex resize-none bg-transparent text-[11.5px] leading-relaxed text-slate-700 placeholder:text-slate-400 focus:outline-none"
              />
              <button
                type="button"
                onClick={busy ? stop : submit}
                disabled={!busy && input.trim().length === 0}
                aria-label={busy ? "Stop" : "Send"}
                className="flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-full bg-slate-700 text-white transition-colors hover:bg-slate-800 disabled:cursor-default disabled:bg-slate-200"
              >
                {busy ? (
                  <Square className="h-2.5 w-2.5 fill-current" />
                ) : (
                  <ArrowUp className="h-3 w-3" />
                )}
              </button>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

/** Every text part of a message, joined. Riz emits no other part type. */
function text(message: UIMessage): string {
  return (message.parts ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}

function Turn({ role, text }: { role: string; text: string }) {
  if (role === "user") {
    return (
      <div className="my-2 flex justify-end">
        <div className="max-w-[85%] rounded-lg rounded-br-sm bg-sky-50 px-2.5 py-1.5 text-[11.5px] leading-relaxed whitespace-pre-wrap text-slate-700">
          {text}
        </div>
      </div>
    );
  }
  return (
    <div className="my-2">
      <MessageBody text={text} />
    </div>
  );
}

function Empty({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="py-2">
      <p className="text-[11.5px] leading-relaxed text-slate-500">
        <strong>Hi! I&apos;m Riz Streem, the interp-engine helpbot.</strong>
        <br />
        Interp-engine is an open source interpretability engine designed to be
        fast, standardized, and easy to use. Ask me for example code, supported
        architectures, or anything interp-engine!
      </p>
      <div className="mt-2.5 flex flex-col items-start gap-y-1">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onPick(suggestion)}
            // Three on a phone: the shorter pane cannot hold six without
            // scrolling the greeting off the top. `sm` restores the rest.
            className="cursor-pointer rounded-md border border-slate-200 px-2 py-1 text-left text-[10.5px] leading-snug text-slate-600 transition-colors hover:border-sky-300 hover:bg-sky-50 hover:text-sky-800 max-sm:nth-[n+4]:hidden"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * The two failures worth telling apart.
 *
 * A refused request arrives as an `Error` whose message is the route's JSON
 * body, because that is what the transport does with a non-2xx response. Rate
 * limits and an unconfigured deployment both come through there and both have
 * a sentence written for the reader; anything else is a network fault or a
 * provider outage, and gets the generic line rather than a JSON blob.
 */
function Failure({ error }: { error: Error }) {
  let detail: string | null = null;
  try {
    const parsed: unknown = JSON.parse(error.message);
    if (parsed && typeof parsed === "object" && "error" in parsed) {
      detail = String((parsed as { error: unknown }).error);
    }
  } catch {
    detail = null;
  }

  return (
    <div className="my-2 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-900">
      {detail ?? "Riz could not be reached. Check your connection and retry."}
    </div>
  );
}

/**
 * The diagram, as the model is told about it.
 *
 * Trait ids rather than their labels: the ids are what `traits.ts` and
 * `arch.py` both use, and that file is in the model's context, so an id is a
 * key it can look up while a label is a phrase it has to match.
 */
function describe(state: VisualizerState): ViewerContext {
  const current = architecture(state.architectureId);
  const compare = state.compareId ? architecture(state.compareId) : null;
  const node = state.shown?.node;

  return {
    architecture: current
      ? `${current.label} (${current.id})`
      : "a hand-edited trait set matching no named family",
    traits: [...state.traits],
    comparingWith: compare ? `${compare.label} (${compare.id})` : null,
    comparingTraits: state.compareTraits ? [...state.compareTraits] : undefined,
    naming: engine(state.engineId).label,
    dimensions: {
      layers: state.dims.layers,
      heads: state.dims.heads,
      kv_heads: state.dims.kvHeads,
      neurons: state.dims.neurons,
      experts: state.dims.experts,
      active_experts: state.dims.activeExperts,
      streams: state.dims.streams,
    },
    focusedPoint: node ? node.point : null,
    focusedRefusal: node?.refusal ?? null,
  };
}
