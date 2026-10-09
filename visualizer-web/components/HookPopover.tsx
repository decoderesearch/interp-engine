"use client";

/**
 * What one point is, how it is derived, and which traits shaped that
 * derivation. The name itself is printed on the diagram, under whichever stack
 * the toggle has selected.
 *
 * With interp-engine selected the card also carries the call that reads the
 * point, over a tab per backend. Only then: the code is this engine's API, and
 * printing it under a TransformerLens or nnterp name would answer a question in
 * one vocabulary with a snippet written in another.
 */

import { Fragment, useState } from "react";

import { CopyButton } from "@/components/CopyButton";
import { NotebookButton } from "@/components/NotebookButton";
import { formatAddress } from "@/data/engines";
import { formulaFor, traitImpacts } from "@/data/formulas";
import { ALL_POINTS, pointSpec } from "@/data/points";
import {
  addressCall,
  DEFAULT_VARIANT,
  FORM_LABEL,
  FORMS,
  readingSnippet,
  readingSnippets,
  SECTIONS,
  VARIANT_LABEL,
  type Form,
  type Located,
  type Snippet,
  type Variant,
} from "@/data/snippets";
import type {
  EngineId,
  GraphNode,
  PointName,
  Role,
  TraitId,
  Width,
} from "@/lib/types";

const WIDTH_LABEL: Record<Width, string> = {
  d_model: "d_model wide",
  heads: "one per attention head",
  kv_heads: "one per key/value head",
  neurons: "one per MLP neuron",
  routing: "one per expert",
  scores: "one pattern per head",
  vocab: "vocabulary wide",
  streams: "one per residual stream",
};

const ROLE_TEXT: Record<Role, string> = {
  resid: "text-role-resid",
  attn: "text-role-attn",
  mlp: "text-role-mlp",
  route: "text-role-route",
  global: "text-role-global",
};

/** Longest first, so `attn_out_post` is matched before `attn_out`. */
const POINT_PATTERN = new RegExp(
  `\\b(${[...ALL_POINTS.map((p) => p.name)]
    .sort((a, b) => b.length - a.length)
    .join("|")})\\b(\\.[\\w-]+)?`,
  "g",
);

const ROLE_OF = new Map<PointName, Role>(
  ALL_POINTS.map((p) => [p.name, p.role]),
);

/**
 * The card's width, which the diagram needs in order to decide which side of a
 * point the card fits on. Must match the `w-[…px]` classes below — Tailwind reads
 * those literally, so the numbers cannot be interpolated from here.
 *
 * The narrow one is not a scaled-down card, it is a different shape: 484px does
 * not fit beside a point on a phone, or on a phone at all.
 */
export const POPOVER_WIDTH = 484;
export const POPOVER_NARROW_WIDTH = 320;

interface Props {
  node: GraphNode;
  traits: Set<TraitId>;
  isMoe: boolean;
  /** Which stack the card is speaking. Only interp-engine gets the snippets. */
  engineId: EngineId;
  /** The checkpoint the snippets load, from the pane's own architecture. */
  hfId: string;
  /** Phone width: a narrower card, opened above or below the point. */
  narrow?: boolean;
}

export function HookPopover({
  node,
  traits,
  isMoe,
  engineId,
  hfId,
  narrow,
}: Props) {
  const spec = pointSpec(node.point);
  const aliases = [node.point, ...node.alsoKnownAs];
  const addressString = formatAddress(node.point, node.layer, node.stream);

  // A merged node stands for more than one point, and each has its own
  // derivation — that is the whole content of the merge.
  const derivations = aliases
    .map((point) => ({
      point,
      formula: formulaFor(point, {
        traits,
        layer: node.layer,
        stream: node.stream,
        isMoe,
      }),
    }))
    .filter((d) => d.formula !== null);

  const impacts = traitImpacts(aliases, traits, isMoe);

  const snippets =
    engineId === "interp-engine" ? readingSnippets(node, hfId) : [];

  return (
    <div
      className={`thin-scrollbar overflow-y-auto overscroll-contain rounded-md border border-slate-200 bg-white p-3 shadow-[0px_10px_38px_-10px_rgba(22,23,24,0.35),0px_10px_20px_-15px_rgba(22,23,24,0.2)] ${
        narrow
          ? // Half the height, because the card is above or below the point on a
            // phone rather than beside it, and the point has to stay visible.
            "max-h-[min(46dvh,420px)] w-[min(320px,calc(100vw-16px))]"
          : "max-h-[min(74dvh,620px)] w-[484px]"
      }`}
    >
      <div className="flex items-baseline justify-between gap-x-2">
        <span className="font-mono text-xs font-semibold text-slate-800">
          {node.point}
        </span>
        <span className="shrink-0 text-[9px] tracking-wide text-slate-400 uppercase">
          {spec ? WIDTH_LABEL[spec.width] : ""}
        </span>
      </div>

      {/* This point rather than the kind of point: the heading above names the
          tensor, and there are as many of it as there are layers and streams.
          Both forms of the one address, because they are accepted in different
          places -- the string wherever an address is taken as input, the
          constructor where a `Cache` hands one back and a string is a KeyError
          on a dict that visibly holds the tensor.

          Wraps between the two rather than shrinking: a long point at full
          depth on a stream runs past even the wide card, and the constructor
          is the half that is better off on its own line. */}
      <div className="mt-1 flex flex-wrap items-baseline gap-x-3 font-mono text-[10.5px] leading-relaxed">
        {/* Dropped when the address adds nothing to the heading -- a point with
            no layer and no stream formats to its own name, and `embeddings`
            twice on consecutive lines reads as a rendering bug. */}
        {addressString !== node.point && (
          <span className="break-all text-slate-600">{addressString}</span>
        )}
        <span className="break-all text-slate-400">{addressCall(node)}</span>
      </div>

      {spec && (
        <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
          {spec.description}
        </p>
      )}

      {node.alsoKnownAs.length > 0 && (
        <p className="mt-1.5 rounded-sm bg-slate-50 px-2 py-1 text-[10px] leading-relaxed text-slate-500">
          Also{" "}
          <span className="font-mono text-slate-700">
            {node.alsoKnownAs.join(", ")}
          </span>{" "}
          here — the same tensor, because this architecture has nothing between
          them.
        </p>
      )}

      {node.refusal && (
        <p className="mt-1.5 rounded-sm bg-amber-50 px-2 py-1 text-[10px] leading-relaxed text-amber-800">
          {node.refusal}
        </p>
      )}

      {derivations.length > 0 && (
        <div className="mt-2.5 border-t border-slate-100 pt-2">
          <div className="text-[9px] font-medium tracking-wide text-slate-400 uppercase">
            Derivation
          </div>
          {derivations.map(({ point, formula }) => (
            <div key={point} className="mt-1">
              <div className="rounded-sm bg-slate-50 px-2 py-1.5 font-mono text-[10.5px] leading-relaxed break-words text-slate-700">
                <Expression text={formula!.expr} />
              </div>
              {formula!.note && (
                <p className="mt-1 text-[10px] leading-relaxed text-slate-400">
                  {formula!.note}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {impacts.length > 0 && (
        <div className="mt-2.5 border-t border-slate-100 pt-2">
          <div className="text-[9px] font-medium tracking-wide text-slate-400 uppercase">
            Why it reads this way
          </div>
          {/* A two-column grid so every impact starts at the same x: the pill
              column is as wide as the widest label rather than as wide as each
              one, which is what left a stack of ragged first lines. A
              description list rather than a `ul` of flex rows, since that is
              what makes `dt` and `dd` direct children of the grid — a `li`
              between them would have to be `display: contents` to get out of
              the way, and that drops the row from the accessibility tree. */}
          <dl className="mt-1.5 grid grid-cols-[auto_1fr] items-start gap-x-1.5 gap-y-1.5">
            {impacts.map(({ trait, text }) => (
              <Fragment key={trait.id}>
                <dt className="mt-px justify-self-start rounded-full border border-sky-600 bg-sky-100 px-1.5 py-px text-[9px] leading-4 font-medium whitespace-nowrap text-sky-700">
                  {trait.label}
                </dt>
                <dd className="text-[10px] leading-relaxed text-slate-500">
                  {text}
                </dd>
              </Fragment>
            ))}
          </dl>
        </div>
      )}

      {snippets.length > 0 && (
        <ReadingIt
          key={node.point}
          snippets={snippets}
          node={node}
          hfId={hfId}
        />
      )}
    </div>
  );
}

/**
 * The one snippet, over a tab per variant and a switch per form. Tabs rather
 * than a stack because the eager and vLLM readings are the same call and differ
 * by one argument: side by side that reads as repetition, while switching a tab
 * moves exactly the line that changed.
 *
 * The tabs sit in two sections, one per machine, with the eyebrow saying
 * which: a reader is on one of them, and the section is what tells them which
 * tabs are theirs. The form switch is a second row because it is a second
 * question -- how to write the call, not where to run it -- and it stays put
 * across tabs, so a reader who writes servers sees `await` on every backend they
 * click.
 *
 * The card opens on `DEFAULT_VARIANT` when the point has a path there, else on
 * the first tab that does, so a point vLLM cannot serve opens on eager with its
 * code showing instead of on a refusal the reader has to click out of. The
 * refused tabs stay in the row: which backend cannot reach a point is part of
 * the answer.
 *
 * The selection is keyed on the point, so opening a different point's card
 * starts from that default again rather than inheriting a choice made about
 * another point — the tab that was selected may not even be available here.
 */
/** Copy and Notebook inside the code block: bordered and lifted off its ground. */
const ACTION_ON_SLATE = "border border-slate-300 bg-white hover:bg-slate-100";

function ReadingIt({
  snippets,
  node,
  hfId,
}: {
  /** One per tab, in the sync form: what the row needs to mark `no path`. */
  snippets: Snippet[];
  node: Located;
  hfId: string;
}) {
  const [selected, setSelected] = useState<Variant>(
    (
      snippets.find((s) => s.variant === DEFAULT_VARIANT && s.code !== null) ??
      snippets.find((s) => s.code !== null) ??
      snippets[0]
    ).variant,
  );
  const [form, setForm] = useState<Form>("sync");
  const active = readingSnippet(selected, node, hfId, form);

  // `onSlate` is the form switch inside the code block: a border a shade darker
  // than that ground keeps an unselected pill from sinking into it, and the
  // selected one is a shade lighter than a backend tab, so the two rows read
  // as first and second choice.
  const tab = (checked: boolean, onSlate = false) =>
    `cursor-pointer rounded-sm px-1.5 font-mono text-[9px] whitespace-nowrap transition-colors ${
      onSlate ? "py-0.5" : "py-1"
    } ${
      checked
        ? onSlate
          ? "border border-slate-500 bg-slate-500 text-white"
          : "bg-slate-700 text-white"
        : onSlate
          ? "border border-slate-300 bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700"
          : "bg-slate-100 text-slate-500 hover:bg-slate-200 hover:text-slate-700"
    }`;

  return (
    <div className="mt-2.5 border-t border-slate-100 pt-2">
      <div className="text-[9px] font-medium tracking-wide text-slate-400 uppercase">
        Reading it
      </div>

      {/* The sections wrap rather than overflow: on the 320px card a phone gets,
          two sections with `no path` suffixes are wider than one line. */}
      <div className="mt-2 flex flex-wrap items-start justify-between gap-x-3 gap-y-1.5">
        {SECTIONS.map((section) => (
          <div key={section.eyebrow}>
            <div className="text-[8px] font-medium tracking-wide text-slate-400 uppercase">
              {section.eyebrow}
            </div>
            {/* A radiogroup rather than buttons in a row: these select one view of
                the same content, and that is what tells a screen reader the set is
                one control. `aria-checked` carries the state the ring shows. */}
            <div
              role="radiogroup"
              aria-label={section.eyebrow}
              className="mt-0.5 flex flex-wrap items-center gap-1"
            >
              {section.variants.map((variant) => {
                const { code } = snippets.find((s) => s.variant === variant)!;
                return (
                  <button
                    key={variant}
                    type="button"
                    role="radio"
                    aria-checked={variant === selected}
                    onClick={() => setSelected(variant)}
                    className={tab(variant === selected)}
                  >
                    {VARIANT_LABEL[variant]}
                    {/* The em dash keeps the tab honest before it is opened: a
                        point vLLM cannot serve at all would otherwise look like a
                        working variant until you click through to it. */}
                    {code === null && (
                      <span
                        className={
                          variant === selected
                            ? "text-slate-300"
                            : "text-slate-400"
                        }
                      >
                        {" \u2014 no path"}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* The form switch and both actions sit inside the code block, on its top
          edge, so they read as acting on the code they share a box with -- they
          carry that pair's code, not the card's. Which is also why Notebook is
          here and not once per card: the template it opens is chosen by the
          backend, so `eager` and `vllm` do not lead to the same one. Each gets a
          border a shade darker than the block, so a control on a slate ground
          still reads as one.

          The code's own bottom padding is for the horizontal scrollbar a long
          checkpoint id brings, so it never lands on the last line. */}
      {active.code !== null && (
        <div className="mt-1.5 rounded-sm bg-slate-50">
          <div className="flex min-h-[18px] items-center justify-between gap-x-2 px-1.5 pt-1.5">
            <div role="radiogroup" className="flex items-center gap-1">
              {FORMS.map((f) => (
                <button
                  key={f}
                  type="button"
                  role="radio"
                  aria-checked={f === form}
                  onClick={() => setForm(f)}
                  className={tab(f === form, true)}
                >
                  {FORM_LABEL[f]}
                </button>
              ))}
            </div>
            <div className="flex shrink-0 items-center gap-x-1">
              <CopyButton text={active.code} className={ACTION_ON_SLATE} />
              {/* `notebook` rather than `code`: on the vLLM tabs' sync form those
                  differ, and the one that runs in a notebook is the one that
                  should travel to one. It is null only when `code` is, so this
                  narrows a type rather than describing a case. */}
              {active.notebook !== null && (
                <NotebookButton
                  text={active.notebook}
                  variant={active.variant}
                  className={ACTION_ON_SLATE}
                />
              )}
            </div>
          </div>
          <pre className="thin-scrollbar overflow-x-auto px-2 pt-1 pb-2 font-mono text-[10px] leading-relaxed text-slate-700">
            <Code text={active.code} />
          </pre>
        </div>
      )}
      {active.note && (
        <p className="mt-1.5 text-[10px] leading-relaxed text-slate-400">
          {active.note}
        </p>
      )}
    </div>
  );
}

/**
 * One snippet: the point's own address picked out, and the trailing shape
 * comment stepped back. The same treatment `Expression` gives a formula, since
 * the address is the only part of the line that changes from point to point.
 */
function Code({ text }: { text: string }) {
  return (
    <>
      {text.split("\n").map((line, i) => {
        const at = line.indexOf("  #");
        return (
          <Fragment key={i}>
            {i > 0 && "\n"}
            <Expression text={at === -1 ? line : line.slice(0, at)} />
            {at !== -1 && (
              <span className="text-slate-400">{line.slice(at)}</span>
            )}
          </Fragment>
        );
      })}
    </>
  );
}

/** Colours the point names inside an expression so they read as hook points. */
function Expression({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  let cursor = 0;
  for (const match of text.matchAll(POINT_PATTERN)) {
    const start = match.index;
    if (start > cursor) parts.push(text.slice(cursor, start));
    const role = ROLE_OF.get(match[1]) ?? "global";
    parts.push(
      <span key={start} className={`font-semibold ${ROLE_TEXT[role]}`}>
        {match[0]}
      </span>,
    );
    cursor = start + match[0].length;
  }
  if (cursor < text.length) parts.push(text.slice(cursor));
  return (
    <>
      {parts.map((part, i) => (
        <Fragment key={i}>{part}</Fragment>
      ))}
    </>
  );
}
