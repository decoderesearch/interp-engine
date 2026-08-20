/**
 * Riz's one endpoint.
 *
 * The only route in this app that is not prerendered, and the reason the
 * deployment is no longer purely a CDN. Everything it does is in service of one
 * shape: a system prompt split into a large block that never changes and a
 * small block that changes every message.
 *
 *   block 1  instructions + this repo's docs -- 103k tokens on the default
 *            model, 132k on Sonnet 5, the same bytes counted by different
 *            tokenizers. Marked `cache_control: ephemeral` with a one-hour
 *            TTL, refreshed free on every hit, so a reader's second question
 *            costs a tenth of the input price of their first -- and so does
 *            the next reader's first, since the prefix is keyed on its
 *            content rather than on a session.
 *   block 2  the diagram they are currently looking at. Deliberately outside
 *            the cache: it changes when they move a slider, and folding it
 *            into block 1 would rewrite the cached prefix on every message --
 *            paying the 1.25x write premium each time and never reading.
 *
 * That split is the whole cost model, and it fails silently when broken: the
 * request still succeeds, the answer is still correct, and the bill is ten
 * times larger with nothing in the logs saying why. Hence `logCacheUsage`.
 */

import { anthropic } from "@ai-sdk/anthropic";
import {
  convertToModelMessages,
  createUIMessageStreamResponse,
  streamText,
  toUIMessageStream,
  type LanguageModelUsage,
  type UIMessage,
} from "ai";

import { KNOWLEDGE, KNOWLEDGE_BYTES } from "@/knowledge";
import {
  SYSTEM_PROMPT,
  viewerContext,
  type ViewerContext,
} from "@/knowledge/prompt";
import {
  MAX_MESSAGES,
  MAX_MESSAGE_CHARS,
  MAX_OUTPUT_TOKENS,
  checkLimit,
  clientKey,
  configured,
  exempt,
} from "@/lib/ratelimit";

/**
 * Node rather than edge: the knowledge bundle is a 350 KB string imported into
 * the module, which the edge runtime's smaller bundle budget does not love.
 */
export const runtime = "nodejs";

/** A long answer over a cold cache write is comfortably inside this. */
export const maxDuration = 60;

/**
 * The model. Overridable because Anthropic's slugs move faster than this
 * repository does, and pinning one in source means a deploy to change it.
 *
 * Haiku rather than Sonnet on the evidence of `npm run eval`, where the two
 * tie at 8/8 on the cases this corpus makes expensive to get wrong. It is
 * about 2.6x cheaper, which is more than the rate card implies: half the price
 * per token, and 22% fewer tokens for the same prefix, because Haiku 4.5
 * predates the hungrier tokenizer Sonnet 5 uses. Re-run the eval before
 * changing this -- eight cases separate a model that can do the job from one
 * that cannot, which is exactly the question a swap asks.
 *
 * The undated alias, matching how the Sonnet default was written. `4.5` with a
 * dot is a 404; the API spells it `4-5`.
 */
const MODEL = process.env.ANTHROPIC_MODEL ?? "claude-haiku-4-5";

interface AskRequest {
  messages: UIMessage[];
  context: ViewerContext | null;
}

export async function POST(request: Request) {
  if (!process.env.ANTHROPIC_API_KEY) {
    return problem(503, "Riz is not configured on this deployment.");
  }

  // Refusing rather than serving unlimited. See lib/ratelimit.ts: a limiter
  // that fails open is indistinguishable from no limiter until the invoice.
  if (!configured && !exempt) {
    return problem(503, "Riz is not configured on this deployment.");
  }

  // Shape before allowance. A request that was never well-formed should not
  // spend one of the reader's questions, and checking it first also keeps
  // garbage traffic off the Redis round trip.
  let body: AskRequest;
  try {
    body = (await request.json()) as AskRequest;
  } catch {
    return problem(400, "Malformed request.");
  }

  const messages = accept(body.messages);
  if (messages === null) {
    return problem(400, "Ask something first.");
  }
  if (messages === TOO_LONG) {
    return problem(
      400,
      `Ask something shorter than ${MAX_MESSAGE_CHARS} characters.`,
    );
  }

  const verdict = await checkLimit(clientKey(request.headers));
  if (!verdict.ok) {
    return problem(429, verdict.reason, {
      "retry-after": String(verdict.retryAfter),
    });
  }

  const result = streamText({
    model: anthropic(MODEL),
    system: [
      {
        role: "system",
        content: `${SYSTEM_PROMPT}\n\n${KNOWLEDGE}`,
        providerOptions: {
          // An hour rather than the default five minutes. The prefix is shared
          // across every reader -- it is keyed on content, not on session -- so
          // the question is how often a visitor arrives to find it already
          // warm. Five minutes almost never spans two visitors to a docs site,
          // which made nearly every conversation pay a full write; an hour
          // usually does. The write costs 2x instead of 1.25x, so this is worth
          // roughly two conversations an hour and loses below one. Reads
          // refresh the hour for free, so a steady trickle never pays twice.
          anthropic: { cacheControl: { type: "ephemeral", ttl: "1h" } },
        },
      },
      { role: "system", content: viewerContext(body.context ?? null) },
    ],
    messages: await convertToModelMessages(messages),
    maxOutputTokens: MAX_OUTPUT_TOKENS,
    // Grounded answers out of documents in front of it, not invention. Low
    // rather than zero so a rephrase of the same question is not word for word.
    //
    // Honored by the Haiku default. Sonnet 5 does not accept it and the AI SDK
    // logs "temperature is not supported" once per request as it drops it --
    // noise rather than a fault, and worth knowing before reading a log after
    // an `ANTHROPIC_MODEL` swap.
    temperature: 0.2,
    providerOptions: {
      // Reasoning tokens are billed as output and drawn from the same
      // `maxOutputTokens`, and these questions are lookups against documents
      // already in the context rather than problems. The budget would buy a
      // truncated answer instead of a better one, with the reader watching a
      // blank panel while it is spent. Set explicitly rather than left to the
      // default because the default differs per model and the failure is a
      // cut-off answer, not an error.
      anthropic: { thinking: { type: "disabled" } },
    },
    onFinish: ({ usage }) => logCacheUsage(usage),
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({
      stream: result.stream,
      // The default replaces every error with "An error occurred", which is
      // right for provider internals and unhelpful for the two the reader can
      // act on. Neither branch forwards a provider message verbatim.
      onError: (error) => {
        console.error("[ask] stream failed", error);
        return "Riz lost that one. Try asking again.";
      },
    }),
  });
}

/** Distinguishes the oversized case from the empty one, which read alike. */
const TOO_LONG = Symbol("too long");

/**
 * Trim the history and reject an oversized message.
 *
 * Trimming keeps the tail: the last exchange is what the next answer is about,
 * and dropping from the front costs the model context it has usually already
 * summarized into its own replies. The length check looks only at what the
 * reader just sent, since anything earlier was checked when it was sent.
 */
function accept(
  messages: UIMessage[] | undefined,
): UIMessage[] | typeof TOO_LONG | null {
  if (!Array.isArray(messages) || messages.length === 0) return null;

  const recent = messages.slice(-MAX_MESSAGES);
  const last = recent[recent.length - 1];
  const length = (last.parts ?? [])
    .map((part) => (part.type === "text" ? part.text.length : 0))
    .reduce((sum, n) => sum + n, 0);

  if (length === 0) return null;
  return length > MAX_MESSAGE_CHARS ? TOO_LONG : recent;
}

/**
 * One line per answer saying whether the prefix was cached.
 *
 * `cacheWriteTokens` on the first message of a conversation and
 * `cacheReadTokens` on the rest is the healthy pattern. Writes every time means
 * something per-request has leaked into the cached block; zeroes on both means
 * `cacheControl` is not reaching the provider at all, which the API reports as
 * a perfectly ordinary success.
 */
function logCacheUsage(usage: LanguageModelUsage) {
  const read = usage.inputTokenDetails?.cacheReadTokens ?? 0;
  const written = usage.inputTokenDetails?.cacheWriteTokens ?? 0;
  const state = read > 0 ? "hit" : written > 0 ? "write" : "UNCACHED";

  // No token estimate here. A byte-count guess sat beside the measurement for a
  // while and disagreed with it by 42% on Sonnet and 10% on Haiku -- the
  // tokenizer changed at Claude 4.7, so bytes do not convert at one rate.
  // `read` and `write` are the real prefix size, for whichever model ran.
  console.log(
    `[ask] cache ${state} read=${read} write=${written} ` +
      `in=${usage.inputTokens ?? 0} out=${usage.outputTokens ?? 0} ` +
      `(prefix ${Math.round(KNOWLEDGE_BYTES / 1024)} KB)`,
  );
}

/** A refusal the panel can render as a message rather than a stack trace. */
function problem(status: number, detail: string, headers: HeadersInit = {}) {
  return Response.json(
    { error: detail },
    { status, headers: { "cache-control": "no-store", ...headers } },
  );
}
