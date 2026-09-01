/**
 * The pre-resolved models, read before the Hub is asked.
 *
 * `data/models.generated.ts` is ~90 KB of facts for the sixty-odd models this project has run, and
 * it is loaded with a dynamic `import()` for one reason: the sizer lives behind a dialog most
 * visitors never open, and a static import would put all of it in the first payload of a page whose
 * job is to draw a diagram. Split this way it is a chunk that arrives when the sizer is opened and
 * never at all otherwise.
 *
 * A hit skips six Hub round trips. It also answers for a gated repo without a token, since the cache
 * was built with one — see `scripts/build-model-cache.ts`.
 */

import type { ModelMemoryFacts } from "@/lib/hub";

type Cache = Record<string, { facts: ModelMemoryFacts }>;

let pending: Promise<Cache> | null = null;

function load(): Promise<Cache> {
  pending ??= import("@/data/models.generated").then((mod) => mod.MODEL_CACHE);
  return pending;
}

/**
 * Start fetching the chunk before anything needs it.
 *
 * Called when the sizer mounts, so the chunk is in flight while the reader is still typing an id
 * rather than after they have asked for it.
 */
export function warmModelCache(): void {
  void load().catch(() => {});
}

/** The cached facts for a model id, or null when it was never resolved ahead of time. */
export async function cachedModel(
  id: string,
): Promise<ModelMemoryFacts | null> {
  try {
    const models = await load();
    return models[id.trim().toLowerCase()]?.facts ?? null;
  } catch {
    // A chunk that failed to load is a slower sizer, not a broken one.
    return null;
  }
}

/** Canonically spelled ids, for the field's suggestions. Empty until the chunk lands. */
export async function cachedModelIds(): Promise<string[]> {
  try {
    const models = await load();
    return Object.values(models)
      .map((entry) => entry.facts.modelId)
      .sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  } catch {
    return [];
  }
}
