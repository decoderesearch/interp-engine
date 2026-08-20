"use client";

/**
 * One turn in the conversation, as markdown.
 *
 * The renderers below exist because the defaults are wrong for a 400px column
 * inside an app whose type scale tops out at 11px: unstyled markdown arrives at
 * 16px with browser margins and immediately looks like a different website
 * bolted onto the side of this one. Every override here is that, and the code
 * block deliberately reuses the hook card's treatment -- same slate-50 plate,
 * same 10px mono, same copy affordance -- because a snippet from Riz and a
 * snippet from a point's card are the same kind of thing.
 *
 * Markdown is re-parsed on every streamed chunk. An unterminated fence parses
 * as a code block running to the end of the input, which is exactly what a
 * half-written snippet should look like, so nothing here special-cases the
 * partial state.
 */

import type { ComponentPropsWithoutRef, ReactNode } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { CopyButton } from "@/components/CopyButton";

/**
 * A fenced block, with the copy button laid over its top-right corner.
 *
 * `react-markdown` hands the fence to `code` wrapped in a `pre`, so the plate
 * is drawn here and `pre` is flattened to a fragment below. Doing it the other
 * way round puts the button outside the element it acts on and loses the
 * language, which is only on the `code` node.
 */
function CodeBlock({ text }: { text: string }) {
  return (
    // Asymmetric margins: more below than above. A snippet is nearly always
    // followed by a sentence about it, and at 11.5px that sentence needs to
    // read as a caption on the plate rather than as its last line.
    <div className="group relative mt-2 mb-3">
      {/* `pb-3` rather than matching `pt-1.5`, because a horizontally
          scrolling snippet puts a scrollbar inside this padding. At
          `scrollbar-width: thin` that is around 11px, so the smaller value
          left it sitting across the descenders of the last line of code. */}
      <pre className="thin-scrollbar overflow-x-auto rounded-sm bg-slate-50 px-2 pt-1.5 pb-3 font-mono text-[10px] leading-relaxed text-slate-700">
        <code>{text}</code>
      </pre>
      <div className="absolute top-1 right-1 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
        <CopyButton text={text} className="bg-slate-50/90 hover:bg-slate-200" />
      </div>
    </div>
  );
}

/** Flattened text of a node, for the clipboard and for the inline test. */
function plain(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(plain).join("");
  return "";
}

export function MessageBody({ text }: { text: string }) {
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        // Fences become the plate above; a single-backtick span stays inline.
        // The distinction is the newline: remark gives both the same node type
        // and only a fence carries one.
        code({ children, className }: ComponentPropsWithoutRef<"code">) {
          const text = plain(children);
          if (!text.includes("\n") && !className?.startsWith("language-")) {
            return (
              <code className="rounded-sm bg-slate-100 px-1 py-px font-mono text-[10.5px] text-slate-700">
                {children}
              </code>
            );
          }
          return <CodeBlock text={text.replace(/\n$/, "")} />;
        },
        // `CodeBlock` already draws the plate, and a `pre` wrapping a `div` is
        // invalid HTML that React will complain about in development.
        pre: ({ children }) => <>{children}</>,

        p: ({ children }) => (
          <p className="my-1.5 text-[11.5px] leading-relaxed text-slate-700">
            {children}
          </p>
        ),
        ul: ({ children }) => (
          <ul className="my-1.5 list-disc space-y-0.5 pl-4 text-[11.5px] leading-relaxed text-slate-700">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="my-1.5 list-decimal space-y-0.5 pl-4 text-[11.5px] leading-relaxed text-slate-700">
            {children}
          </ol>
        ),
        // Riz is told not to use headings in a reply this short. It sometimes
        // does anyway, and one flattened to bold text beats one at 2em.
        h1: ({ children }) => <Heading>{children}</Heading>,
        h2: ({ children }) => <Heading>{children}</Heading>,
        h3: ({ children }) => <Heading>{children}</Heading>,
        h4: ({ children }) => <Heading>{children}</Heading>,
        strong: ({ children }) => (
          <strong className="font-semibold text-slate-800">{children}</strong>
        ),
        a: ({ children, href }) => (
          <a
            href={href}
            target="_blank"
            rel="noreferrer noopener"
            className="text-sky-700 underline underline-offset-2 hover:text-sky-800"
          >
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-1.5 border-l-2 border-slate-200 pl-2 text-[11px] text-slate-500">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-2 border-slate-200" />,
        // A table in a 400px column scrolls rather than squeezing its cells to
        // one character wide, which is what `table-auto` does unprompted.
        table: ({ children }) => (
          <div className="thin-scrollbar my-2 overflow-x-auto">
            <table className="w-full border-collapse text-[10.5px]">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b border-slate-200 px-1.5 py-1 text-left font-medium text-slate-500">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border-b border-slate-100 px-1.5 py-1 align-top text-slate-700">
            {children}
          </td>
        ),
      }}
    >
      {text}
    </Markdown>
  );
}

function Heading({ children }: { children: ReactNode }) {
  return (
    <div className="mt-2.5 mb-1 text-[11px] font-semibold text-slate-800">
      {children}
    </div>
  );
}
