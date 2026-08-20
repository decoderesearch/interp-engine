/**
 * The generated documents, framed as one string for the model.
 *
 * Hand-written and kept apart from `bundle.generated.ts` so that the wording
 * below can be edited without regenerating, and so regenerating never argues
 * with an edit. The generated half is data; this half is presentation.
 *
 * Everything here is byte-stable given the same bundle. It is the head of a
 * prompt marked `cache_control: ephemeral`, and anything that varies per
 * request -- a timestamp, the reader's state, a request id -- belongs in the
 * second, uncached block that `app/api/ask/route.ts` appends, not in here.
 */

import { KNOWLEDGE_DOCUMENTS } from "./bundle.generated";

/**
 * A contents page ahead of the documents themselves.
 *
 * Without it the model has to scan 350 KB to find out whether a question is
 * even covered, and the failure mode when it does not is answering from
 * pre-training about some other interpretability library. With it, "which
 * document would say" is a question it can answer before reading.
 */
function contents(): string {
  const rows = KNOWLEDGE_DOCUMENTS.map(
    (doc) => `- \`${doc.path}\` — ${doc.title}. ${doc.why}`,
  ).join("\n");

  return `The documents below are the whole of what you know. Their paths are what you cite.\n\n${rows}`;
}

/**
 * Each document fenced in its own language and tagged with its path.
 *
 * The fence matters for the Python and TypeScript sources: unfenced, a
 * docstring full of prose reads as instructions rather than as the contents of
 * a file, and `points.py`'s refusal strings are exactly the text most likely to
 * be mistaken for something addressed to the model.
 */
function documents(): string {
  return KNOWLEDGE_DOCUMENTS.map(
    (doc) =>
      `<document path="${doc.path}" title="${doc.title}">\n` +
      `\`\`\`${doc.kind}\n${doc.text}\n\`\`\`\n` +
      `</document>`,
  ).join("\n\n");
}

/** The cacheable half of the system prompt: contents page, then sources. */
export const KNOWLEDGE = `${contents()}\n\n${documents()}`;

/**
 * Size of the cacheable prefix, in bytes.
 *
 * Bytes and not an estimated token count, because there is no single token
 * count to state. Dividing by four gives ~93k; the same string measures 103k
 * on Haiku 4.5 and 132k on Sonnet 5, since the tokenizer changed at Claude 4.7
 * and the newer one runs about 30% hotter. A per-model number belongs to the
 * model, so the route logs the provider's own `cacheReadTokens` /
 * `cacheWriteTokens` for whichever one ran; this is for a sense of scale only.
 */
export const KNOWLEDGE_BYTES = KNOWLEDGE.length;
