/**
 * Model facts, resolved on the server against this project's Hub token.
 *
 * The sizer used to read the Hub straight from the browser and ask anyone with a gated or unlisted
 * model to bring a token. That works, and it is still what happens when a reader supplies one — see
 * `lib/resolve.ts` — but it makes the common case worse than it needs to be: an anonymous browser
 * gets the Hub's anonymous allowance, shared with everything else behind the same NAT, and a repo
 * that merely requires *any* account is unreachable for no reason the reader can act on.
 *
 * So there are three tiers now, in this order: the build-time cache answers without a request at
 * all, this route answers with the shared token, and a reader's own token goes direct from their
 * browser to huggingface.co. The token here is read from the environment and never leaves the
 * server — the response carries facts, which are arithmetic inputs and public information about a
 * public repo.
 *
 * What the token is not is unlimited. `checkHubLimit` bounds one address to thirty lookups an hour,
 * because the thing being spent is this account's standing with the Hub and running it down would
 * break lookups for everyone at once rather than arriving as a bill.
 */

import { HubError, isRepoId, resolveModel } from "@/lib/hub";
import { checkHubLimit, clientKey, configured, exempt } from "@/lib/ratelimit";

/** `lib/hub.ts` is isomorphic, but it reads six URLs per model and wants a real Node fetch. */
export const runtime = "nodejs";

/** Six sequential-ish Hub reads on a cold repo, and the Hub is occasionally slow. */
export const maxDuration = 30;

export async function GET(request: Request) {
  const token = process.env.HF_TOKEN;
  if (!token) {
    return problem(503, "model lookup is not configured on this deployment");
  }

  // Refusing rather than serving unlimited, the same call `/api/ask` makes and for the same reason:
  // an unmetered shared token is one crawler away from being a revoked shared token.
  if (!configured && !exempt) {
    return problem(503, "model lookup is not configured on this deployment");
  }

  const model = new URL(request.url).searchParams.get("model")?.trim() ?? "";
  // Shape before allowance: a request that was never well-formed should not spend a lookup.
  if (!isRepoId(model)) {
    return problem(400, "not a Hugging Face model id");
  }

  const verdict = await checkHubLimit(clientKey(request.headers));
  if (!verdict.ok) {
    return problem(429, verdict.reason, {
      "retry-after": String(verdict.retryAfter),
    });
  }

  try {
    const facts = await resolveModel(model, { token });
    return Response.json(facts, { headers: { "cache-control": "no-store" } });
  } catch (cause) {
    if (cause instanceof HubError) {
      // A network failure reaching the Hub is 502 here rather than the 0 it carries: from the
      // reader's side an upstream that did not answer is this deployment's problem, not theirs.
      return problem(cause.status || 502, cause.message);
    }
    console.error("[hub] resolve failed", cause);
    return problem(502, "could not reach the Hub");
  }
}

function problem(
  status: number,
  error: string,
  headers: Record<string, string> = {},
): Response {
  return Response.json(
    { error },
    { status, headers: { "cache-control": "no-store", ...headers } },
  );
}
