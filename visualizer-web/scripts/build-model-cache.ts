/**
 * Resolve the models people actually ask for, once, and commit the answer.
 *
 * `lib/hub.ts` resolves a model in up to six Hub requests, and every one of them is on the path
 * between a reader typing an id and the panel showing a number. For a model nobody has heard of that
 * is the price of the tool working at all; for the sixty this project has already run through the
 * validator it is a round trip to learn something that has not changed since the checkpoint was
 * published.
 *
 * So the common case is resolved here and read from `data/models.generated.ts` at runtime. Two
 * things fall out of that beyond the latency:
 *
 * - **A gated repo resolves for everyone.** This runs with the maintainer's `HF_TOKEN`, and what it
 *   writes is dimensions rather than weights, so `google/gemma-3-12b-pt` — the placeholder in the
 *   sizer's own input field — stops being a model that only works if you have a token.
 * - **No token is needed at runtime**, and the browser still talks to the Hub directly for anything
 *   not cached, so the promise `lib/hub.ts` makes about a reader's own token is untouched.
 *
 * The list is derived rather than typed. `validator/comparison/results/` is every model this repo has
 * really run, and `VERIFIED_RUNS` is every one with hardware evidence behind it — those two are the
 * definition of "commonly requested" here, and deriving them means the cache grows as the project
 * does rather than as somebody remembers to edit an array.
 *
 * Usage:
 *   npx tsx scripts/build-model-cache.ts            # rewrite the cache
 *   npx tsx scripts/build-model-cache.ts --check    # fail if the Hub has moved under it
 *
 * `--check` is deliberately not part of `viz-check`: it needs the network and a token, and a static
 * check that fails when a third party edits a config is a check that trains people to ignore it. Run
 * it on a schedule, or before a release.
 */

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { VERIFIED_RUNS } from "@/data/gpus.generated";
import { resolveModel, type ModelMemoryFacts } from "@/lib/hub";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, "..");
const REPO = path.join(WEB, "..");
const VALIDATED = path.join(REPO, "validator", "comparison", "results");
const OUT = path.join(WEB, "data", "models.generated.ts");

/**
 * Models worth caching that neither source names.
 *
 * Only for a model the project has not run but a reader is likely to type. Everything else should
 * arrive by being validated, which is the bar that keeps this list from becoming a wish list.
 *
 * The four below are the sizer's one-click quick picks, which makes them the only ids whose absence
 * from the cache would be visible as a broken button rather than as one slow resolution — and
 * `Llama-3.3-70B-Instruct` is gated, so uncached it would ask a reader for a token before showing
 * them anything. Three are validated already and named here anyway, to pin the dependency where it
 * can be seen from either end; naming them costs nothing, since `wanted()` is a `Set`.
 *
 * `DeepSeek-V4-Flash` is the one that is only here. The project validates and benchmarks the `0731`
 * revision, which the cache picks up from `VALIDATED` on its own, but the quick pick offers the
 * original — it is the release people mean by the name, and the two size differently enough that
 * pointing the button at the checkpoint we happen to have measured would answer a question nobody
 * asked.
 */
const EXTRA: string[] = [
  "google/gemma-2-9b-it",
  "meta-llama/Llama-3.3-70B-Instruct",
  "deepseek-ai/DeepSeek-V4-Flash",
  "Qwen/Qwen3.6-27B",
];

/** How many resolutions are in flight at once. Each is a handful of small requests. */
const CONCURRENCY = 4;

interface CachedModel {
  facts: ModelMemoryFacts;
  sha: string;
}

/** The repo's current revision, so a later `--check` can tell "moved" from "we were wrong". */
async function headSha(id: string, token: string): Promise<string> {
  try {
    const response = await fetch(`https://huggingface.co/api/models/${id}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (!response.ok) return "";
    const body = (await response.json()) as { sha?: string };
    return body.sha ?? "";
  } catch {
    return "";
  }
}

/** The repo keeps a token in `.env` for the Python side; reuse it rather than ask for a second. */
async function envToken(): Promise<string> {
  try {
    const text = await fs.readFile(path.join(REPO, ".env"), "utf8");
    return /^HF_TOKEN=(.*)$/m.exec(text)?.[1]?.trim() ?? "";
  } catch {
    return "";
  }
}

/** Every `org/model` under `validator/comparison/results/`, which is what this repo has run. */
async function validatedModels(): Promise<string[]> {
  const out: string[] = [];
  let orgs: string[];
  try {
    orgs = (await fs.readdir(VALIDATED, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);
  } catch {
    console.warn(`no validator results at ${VALIDATED}; using the verified runs alone`);
    return out;
  }
  for (const org of orgs) {
    const models = await fs.readdir(path.join(VALIDATED, org), {
      withFileTypes: true,
    });
    for (const model of models) {
      if (model.isDirectory()) out.push(`${org}/${model.name}`);
    }
  }
  return out;
}

async function wanted(): Promise<string[]> {
  const ids = new Set<string>([
    ...(await validatedModels()),
    ...VERIFIED_RUNS.map((run) => run.modelId),
    ...EXTRA,
  ]);
  return [...ids].sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
}

/**
 * Resolve every id, a few at a time.
 *
 * A failure is reported and skipped rather than fatal: a model that has been renamed or pulled
 * should cost its own row and not the whole cache, and the run that discovers it is usually the one
 * that also wants the other sixty rewritten.
 */
async function resolveAll(
  ids: string[],
  token: string,
): Promise<Record<string, CachedModel>> {
  const out: Record<string, CachedModel> = {};
  let cursor = 0;
  let failures = 0;

  async function worker(): Promise<void> {
    for (let index = cursor++; index < ids.length; index = cursor++) {
      const id = ids[index];
      try {
        const [facts, sha] = await Promise.all([
          resolveModel(id, { token }),
          headSha(id, token),
        ]);
        out[id.toLowerCase()] = {
          facts: { ...facts, weights: sortElements(facts.weights) },
          sha,
        };
        const gib = (facts.weights.onDiskBytes / 1024 ** 3).toFixed(2);
        const dims = facts.trunkDimsKnown ? `${facts.nLayers}L` : "no dims";
        console.log(`  ok    ${id} — ${gib} GiB, ${dims}`);
      } catch (cause) {
        failures += 1;
        console.log(
          `  FAIL  ${id} — ${cause instanceof Error ? cause.message : String(cause)}`,
        );
      }
    }
  }

  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, ids.length) }, worker),
  );
  if (failures) console.log(`\n${failures} model(s) could not be resolved`);
  return out;
}

/** The Hub returns dtype buckets in no particular order, and an unstable key order is a fake diff. */
function sortElements(
  weights: ModelMemoryFacts["weights"],
): ModelMemoryFacts["weights"] {
  const elementsByDtype: Record<string, number> = {};
  for (const key of Object.keys(weights.elementsByDtype).sort()) {
    elementsByDtype[key] = weights.elementsByDtype[key];
  }
  return { ...weights, elementsByDtype };
}

/**
 * One entry per line-block, indented as a whole.
 *
 * Nothing per-model records when it was written, and the one timestamp that exists sits at the top:
 * a `resolvedAt` on every entry would rewrite all sixty-four rows on every run, so the diff that
 * should say "gemma moved" would say "everything moved" instead.
 */
function render(models: Record<string, CachedModel>): string {
  const body = Object.keys(models)
    .sort()
    .map((key) => {
      const entry = JSON.stringify(models[key], null, 2)
        .split("\n")
        .map((line, index) => (index ? `  ${line}` : line))
        .join("\n");
      return `  ${JSON.stringify(key)}: ${entry}`;
    })
    .join(",\n");

  return `/**
 * Generated by \`npx tsx scripts/build-model-cache.ts\`. Do not edit: run \`make viz-models\`.
 *
 * Facts only. The arithmetic that spends them is \`lib/size.ts\`, which reprices on every control
 * change, so caching a fit would cache the one thing that is never reused.
 *
 * Resolved with a token, which is why the gated repos here carry trunk dims. What is stored is
 * dimensions and byte counts — nothing a model card does not already say.
 */

import type { ModelMemoryFacts } from "@/lib/hub";

export interface CachedModel {
  facts: ModelMemoryFacts;
  /** The repo revision these facts were read from, for \`--check\` to compare against. */
  sha: string;
}

export const GENERATED_AT = ${JSON.stringify(new Date().toISOString())};

/** Keyed by lowercased model id; \`facts.modelId\` keeps the canonical spelling. */
export const MODEL_CACHE: Record<string, CachedModel> = {
${body},
};
`;
}

/** Everything that decides a number, which is everything except when it was read. */
function comparable(model: CachedModel): string {
  return JSON.stringify(model.facts);
}

async function check(fresh: Record<string, CachedModel>): Promise<number> {
  let existing: Record<string, CachedModel>;
  try {
    ({ MODEL_CACHE: existing } = await import("@/data/models.generated"));
  } catch {
    console.error("\nno cache to check; run without --check first");
    return 1;
  }

  let drifted = 0;
  for (const [key, model] of Object.entries(fresh)) {
    const was = existing[key];
    if (!was) {
      console.log(`\nNEW    ${model.facts.modelId} is not in the cache`);
      drifted += 1;
    } else if (comparable(was) !== comparable(model)) {
      console.log(
        `\nDRIFT  ${model.facts.modelId}` +
          (was.sha && model.sha && was.sha !== model.sha
            ? ` — repo moved from ${was.sha.slice(0, 8)} to ${model.sha.slice(0, 8)}`
            : " — same revision, different facts, so this client changed"),
      );
      drifted += 1;
    }
  }
  for (const key of Object.keys(existing)) {
    if (!(key in fresh)) {
      console.log(`\nSTALE  ${existing[key].facts.modelId} no longer resolves`);
      drifted += 1;
    }
  }
  console.log(
    drifted
      ? `\n${drifted} model(s) drifted; run \`make viz-models\``
      : "\nthe cache matches the Hub",
  );
  return drifted ? 1 : 0;
}

async function main(): Promise<void> {
  const checking = process.argv.includes("--check");
  const token = process.env.HF_TOKEN || (await envToken());
  if (!token) {
    // Not fatal, because the cache is still worth having without one -- but a run with no token
    // silently writes `trunkDimsKnown: false` for every gated repo, which is the whole reason the
    // cache is worth building. Better to say so than to ship a cache that quietly sizes nothing.
    console.warn(
      "no HF_TOKEN: gated repos will be cached without trunk dims, which is worse than not caching them\n",
    );
  }

  const ids = await wanted();
  console.log(`resolving ${ids.length} models${token ? " with a token" : ""}\n`);
  const models = await resolveAll(ids, token);

  if (checking) {
    process.exitCode = await check(models);
    return;
  }

  await fs.writeFile(OUT, render(models), "utf8");
  const bytes = (await fs.stat(OUT)).size;
  console.log(
    `\nwrote ${Object.keys(models).length} models to data/models.generated.ts ` +
      `(${Math.round(bytes / 1024)} KB)`,
  );
}

void main();
