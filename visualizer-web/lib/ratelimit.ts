/**
 * What stands between `/api/ask` and someone else's Anthropic bill.
 *
 * The route is unauthenticated by design -- a chatbot on a public docs page
 * that asks you to sign in is a chatbot nobody uses -- which means the only
 * thing bounding cost is what is in this file. Four layers, because each
 * catches something the others do not:
 *
 *   - Two sliding windows per IP, in Redis, so the limit survives the lambda
 *     that served the last request being recycled. A per-process counter is
 *     worth roughly nothing here: Vercel will happily hand the next request to
 *     a cold instance whose counter starts at zero.
 *   - `BURST` catches someone holding down enter. `DAILY` catches a slow drip
 *     that never trips `BURST`, which is the shape a script uses.
 *   - `GLOBAL` is the ceiling on the bill. Neither of the first two is: they
 *     bound one address, and addresses are rented by the thousand, so on their
 *     own they set the price of abuse without capping the total.
 *   - The caps below the limiter bound the cost of a *single* request, which
 *     no rate limit does. A 400 KB message inside the allowance is still 100k
 *     tokens of input.
 *
 * Absent configuration the route refuses rather than serving unlimited. That
 * is the whole reason this module exports `configured` instead of quietly
 * returning "allowed" -- a rate limiter that fails open is a rate limiter that
 * is off, and the first you would hear of it is the invoice.
 */

import { Ratelimit } from "@upstash/ratelimit";
import { Redis } from "@upstash/redis";

/**
 * Longest single question. Around 1,100 tokens, which is enough to paste a
 * stack trace or a config and well short of enough to matter beside a 103k
 * cached prefix.
 */
export const MAX_MESSAGE_CHARS = 4000;

/**
 * Messages kept from the tail of the conversation. The prefix is cached and the
 * history is not, so an unbounded thread is the one part of the prompt whose
 * cost grows without limit. Twelve is six exchanges, past which the reader is
 * better served by a fresh thread anyway.
 */
export const MAX_MESSAGES = 12;

/** Enough for an explanation and a snippet, not enough for a tutorial. */
export const MAX_OUTPUT_TOKENS = 1200;

/** Holding down enter. */
const BURST = { limit: 20, window: "5 m" } as const;

/**
 * A script that paces itself under `BURST`.
 *
 * Generous enough that `GLOBAL` is now the binding constraint for anyone
 * determined: ten addresses at this allowance exhaust the shared budget for
 * the day. That is the intended division of labour -- this window is set for
 * the reader who is genuinely working through the docs, and the ceiling below
 * is what a proxy pool runs into.
 */
const DAILY = { limit: 200, window: "24 h" } as const;

/**
 * Every reader together, which is the only limit that bounds the invoice.
 *
 * The two above bound one address, and an address is not scarce. A rotating
 * proxy pool rents a thousand of them for pocket money, and each one is worth
 * `DAILY` questions -- so per-IP limits alone put no ceiling on the bill at
 * all, they only set the price per unit of abuse. This is the ceiling.
 *
 * 2,000 a day is roughly $59 at the worst case a single question can reach on
 * the Haiku default (a full twelve-message history, a maxed answer) and nearer
 * $30 in ordinary use. Overridable because the right number is a budget
 * decision rather than an engineering one, and because the alternative to
 * tuning it is deploying to change it.
 *
 * Both figures assume the prefix is warm, which it nearly always is on a 1h
 * TTL. Cache writes are bounded separately and are small beside this: at most
 * one an hour, 21 cents each, so about $5 a day if the panel is never idle for
 * a full hour.
 */
const GLOBAL = {
  limit: Number(process.env.RIZ_GLOBAL_DAILY ?? 2000),
  window: "24 h",
} as const;

/**
 * The shared Hub token's allowance, per address, for `/api/hub`.
 *
 * A different scarcity from the three above. Nothing here bills by the token:
 * what runs out is the standing this project's Hub account has with
 * huggingface.co, and the punishment for spending it is that *everyone's*
 * lookups start failing at once. So the number is set to the shape of honest
 * use rather than to a budget -- thirty models an hour is more than anyone
 * comparing hardware gets through, and far short of a crawl.
 *
 * Thirty an hour is 720 a day from one address, so {@link HUB_GLOBAL} is what
 * actually bounds the token's exposure -- the same division of labour as Riz's
 * two tiers, and for the same reason: addresses are rented by the thousand, so
 * a per-IP window prices abuse without capping it.
 */
const HUB = { limit: 30, window: "1 h" } as const;

/**
 * Every reader together, against the shared Hub token.
 *
 * 2,400 a day is a little over three addresses running {@link HUB} flat out,
 * which is the right shape: comfortably above what a day of genuine traffic
 * asks for, and well under the volume that gets an account's reads throttled at
 * the other end.
 *
 * This ceiling refuses more gently than Riz's does, because unlike a spent
 * Anthropic budget it is not the end of the road. The build-time cache still
 * answers for the models most readers want without consuming anything here, and
 * the refusal names the way past it: a token of the reader's own, which skips
 * this route and this counter entirely. Overridable for the same reason as
 * `RIZ_GLOBAL_DAILY` -- the right number is a judgement about one Hub account's
 * standing, and the alternative to tuning it is a deploy.
 */
const HUB_GLOBAL = {
  limit: Number(process.env.HUB_GLOBAL_DAILY ?? 2400),
  window: "24 h",
} as const;

/**
 * Vercel's Upstash integration provisions `KV_REST_API_*`; Upstash's own
 * dashboard gives you `UPSTASH_REDIS_REST_*`. `Redis.fromEnv()` reads only the
 * second pair, so both are read here rather than leaving a correctly
 * provisioned project mysteriously unlimited.
 */
function credentials(): { url: string; token: string } | null {
  const url = process.env.UPSTASH_REDIS_REST_URL ?? process.env.KV_REST_API_URL;
  const token =
    process.env.UPSTASH_REDIS_REST_TOKEN ?? process.env.KV_REST_API_TOKEN;
  return url && token ? { url, token } : null;
}

const redis = (() => {
  const creds = credentials();
  return creds ? new Redis(creds) : null;
})();

const burst = redis
  ? new Ratelimit({
      redis,
      prefix: "riz:burst",
      limiter: Ratelimit.slidingWindow(BURST.limit, BURST.window),
    })
  : null;

const daily = redis
  ? new Ratelimit({
      redis,
      prefix: "riz:daily",
      limiter: Ratelimit.slidingWindow(DAILY.limit, DAILY.window),
    })
  : null;

// Not `global`: this is a server module, and shadowing Node's own binding to
// save four characters is a trap for whoever next reaches for it here.
const budget = redis
  ? new Ratelimit({
      redis,
      prefix: "riz:global",
      limiter: Ratelimit.slidingWindow(GLOBAL.limit, GLOBAL.window),
    })
  : null;

const hub = redis
  ? new Ratelimit({
      redis,
      prefix: "hub:ip",
      limiter: Ratelimit.slidingWindow(HUB.limit, HUB.window),
    })
  : null;

const hubBudget = redis
  ? new Ratelimit({
      redis,
      prefix: "hub:global",
      limiter: Ratelimit.slidingWindow(HUB_GLOBAL.limit, HUB_GLOBAL.window),
    })
  : null;

/** Whether a limiter exists. The route refuses when this is false. */
export const configured = redis !== null;

/**
 * Development without Redis is allowed, because the alternative is that nobody
 * can run the panel locally without provisioning a database. The exemption is
 * on `NODE_ENV` rather than on a flag someone could set in production.
 */
export const exempt = process.env.NODE_ENV === "development";

export type LimitVerdict =
  { ok: true } | { ok: false; reason: string; retryAfter: number };

/**
 * Which client this is. Vercel sets `x-forwarded-for` and appends, so the
 * first entry is the client and the rest are proxies; taking the last would
 * bucket every reader behind one edge node together.
 *
 * A shared NAT means several readers share an allowance. That is the right
 * trade for a docs page -- the alternative is a cookie, which is one
 * `curl --cookie-jar` away from being no limit at all.
 */
export function clientKey(headers: Headers): string {
  const forwarded = headers.get("x-forwarded-for");
  const address = forwarded
    ? forwarded.split(",")[0].trim()
    : headers.get("x-real-ip")?.trim() || "unknown";
  return bucket(address);
}

/**
 * The address, narrowed to something the client cannot trivially change.
 *
 * An IPv4 address is one host and is worth keying on whole. An IPv6 address is
 * not: the smallest block anyone is assigned is a /64, and most hosting and
 * consumer ISPs hand out that or larger. A limiter keyed on all 128 bits gives
 * such a client 18 quintillion fresh allowances, so it is not a limiter at all
 * for exactly the population most able to abuse it -- and the failure is
 * invisible, because every request looks like a first request from a new
 * reader. Keying on the /64 buckets a subscriber line the way an IPv4 address
 * buckets a household.
 *
 * Wider than /64 would start merging unrelated customers of the same ISP.
 */
function bucket(address: string): string {
  if (!address.includes(":")) return address;

  // `::` elides a run of zero groups, so an abbreviated address has fewer than
  // eight and expanding it before taking a prefix is not optional: `2001:db8::1`
  // truncated as text is a different /64 from the one it expands to.
  const [head, tail = ""] = address.split("::");
  const left = head ? head.split(":") : [];
  const right = tail ? tail.split(":") : [];
  const groups = address.includes("::")
    ? [
        ...left,
        ...Array(Math.max(0, 8 - left.length - right.length)).fill("0"),
        ...right,
      ]
    : left;

  return `${groups.slice(0, 4).join(":")}::/64`;
}

export async function checkLimit(key: string): Promise<LimitVerdict> {
  if (!burst || !daily || !budget) return { ok: true };

  // Both windows are consumed on every request rather than short-circuiting on
  // the first. Checking `daily` only after `burst` passes would let a caller
  // who is already over the daily cap keep refilling the burst window for free.
  let fast, slow;
  try {
    [fast, slow] = await Promise.all([burst.limit(key), daily.limit(key)]);
  } catch (error) {
    // Redis unreachable. Refusing is the same call as refusing when it was
    // never configured, and for the same reason -- an outage is exactly when
    // someone would notice the limit had stopped applying. Retry soon rather
    // than at a window boundary, since there is no window to wait for.
    console.error("[ask] rate limiter unreachable", error);
    return {
      ok: false,
      reason:
        "Riz is having trouble keeping count right now. Try again in a minute.",
      retryAfter: 60,
    };
  }

  if (!fast.success) {
    return {
      ok: false,
      reason: `That's ${BURST.limit} questions in five minutes, which is where Riz stops. Try again shortly.`,
      retryAfter: retryAfter(fast.reset),
    };
  }
  if (!slow.success) {
    return {
      ok: false,
      reason: `That's ${DAILY.limit} questions today, which is Riz's daily allowance. The docs at github.com/decoderesearch/interp-engine do not run out.`,
      retryAfter: retryAfter(slow.reset),
    };
  }

  // Only now, and deliberately not alongside the two above. The shared budget
  // is the one counter a refused request must not consume: charging it for
  // traffic that was already turned away would let a single blocked address
  // spend everyone else's allowance, turning a rate limit into a denial of
  // service that costs the attacker nothing.
  let total;
  try {
    total = await budget.limit("all");
  } catch (error) {
    console.error("[ask] global budget unreachable", error);
    return {
      ok: false,
      reason:
        "Riz is having trouble keeping count right now. Try again in a minute.",
      retryAfter: 60,
    };
  }

  if (!total.success) {
    // Says nothing about the reader, who has done nothing wrong, and does not
    // hint that the way through is to come back from somewhere else.
    console.warn(`[ask] global daily budget of ${GLOBAL.limit} exhausted`);
    return {
      ok: false,
      reason:
        "Riz has hit his daily limit across everyone asking. He'll be back tomorrow — the docs at github.com/decoderesearch/interp-engine are always up.",
      retryAfter: retryAfter(total.reset),
    };
  }
  return { ok: true };
}

/**
 * The allowance for one lookup against the shared Hub token.
 *
 * Two windows rather than `checkLimit`'s three: one address, then everyone. The
 * middle tier there exists to separate a held-down enter key from a paced
 * script, a distinction with no analogue here, where a lookup is a lookup.
 */
export async function checkHubLimit(key: string): Promise<LimitVerdict> {
  if (!hub || !hubBudget) return { ok: true };

  let window;
  try {
    window = await hub.limit(key);
  } catch (error) {
    console.error("[hub] rate limiter unreachable", error);
    return {
      ok: false,
      reason:
        "the lookup service is having trouble keeping count right now — try again in a minute, or add your own Hugging Face token below",
      retryAfter: 60,
    };
  }

  if (!window.success) {
    return {
      ok: false,
      reason: `that is ${HUB.limit} model lookups in an hour on the shared token, which is where it stops — add your own Hugging Face token below to keep going`,
      retryAfter: retryAfter(window.reset),
    };
  }

  // After the per-address window and never alongside it, for the reason
  // `checkLimit` gives at the same point: a refused request that still spends
  // the shared counter lets one blocked address burn everyone else's
  // allowance, which turns a rate limit into a denial of service.
  let total;
  try {
    total = await hubBudget.limit("all");
  } catch (error) {
    console.error("[hub] global budget unreachable", error);
    return {
      ok: false,
      reason:
        "the lookup service is having trouble keeping count right now — try again in a minute, or add your own Hugging Face token below",
      retryAfter: 60,
    };
  }

  if (!total.success) {
    // Says nothing about this reader, who has done nothing wrong, and offers
    // the way through rather than a time to come back at.
    console.warn(`[hub] global daily budget of ${HUB_GLOBAL.limit} exhausted`);
    return {
      ok: false,
      reason:
        "the shared token has hit its daily lookup limit across everyone using it — cached models still work, and your own Hugging Face token below skips the queue entirely",
      retryAfter: retryAfter(total.reset),
    };
  }
  return { ok: true };
}

/** Seconds until the window reopens, as the `Retry-After` header wants it. */
function retryAfter(reset: number): number {
  return Math.max(1, Math.ceil((reset - Date.now()) / 1000));
}
