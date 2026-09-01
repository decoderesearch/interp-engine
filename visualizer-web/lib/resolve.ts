"use client";

/**
 * Where the sizer's facts come from, in the order it is worth asking.
 *
 * Three tiers, and each exists because the one after it is worse in a way the reader would feel:
 *
 * 1. **The build-time cache.** No request at all, and it answers for gated repos because it was
 *    built with a token — see `lib/models.ts`. Sixty-odd models, which is most of what anyone types.
 * 2. **`/api/hub`, on this project's token.** For everything else. An anonymous browser gets the
 *    Hub's anonymous allowance, shared with everything else behind the same address, and a repo
 *    that merely requires an account is unreachable for a reason the reader cannot act on. Metered
 *    at thirty an hour per address, because the token is lent rather than given.
 * 3. **The reader's own token, straight from the browser.** Skips this deployment entirely, which
 *    is the point: `lib/hub.ts` promises the token is sent to huggingface.co and nowhere else, and
 *    routing it through a server here would quietly retract that. It is also the only tier that can
 *    reach a private repo, since the shared token has no business being able to.
 *
 * The order is not a fallback chain. Tier 3 is chosen *ahead* of tier 2 whenever a token is held,
 * because someone who typed one did so to reach something the shared token cannot.
 */

import { HubError, hasHubToken, resolveModel } from "@/lib/hub";
import type { ModelMemoryFacts } from "@/lib/hub";
import { cachedModel } from "@/lib/models";

export async function resolveFacts(id: string): Promise<ModelMemoryFacts> {
  const cached = await cachedModel(id);
  if (cached) return cached;
  if (hasHubToken()) return resolveModel(id);
  return viaRoute(id);
}

async function viaRoute(id: string): Promise<ModelMemoryFacts> {
  let response: Response;
  try {
    response = await fetch(`/api/hub?model=${encodeURIComponent(id)}`);
  } catch {
    throw new HubError(`could not reach the Hub for ${id}`, 0);
  }

  if (!response.ok) {
    // The route sends `{ error }` for every refusal it authors. A body that is not that shape came
    // from somewhere else -- a platform 502, an edge timeout -- and has no sentence worth showing.
    const said = await response
      .json()
      .then((body: { error?: string }) => body.error)
      .catch(() => null);
    throw new HubError(
      said || `the lookup service answered ${response.status}`,
      response.status,
      response.status === 401 || response.status === 403,
    );
  }

  return (await response.json()) as ModelMemoryFacts;
}
