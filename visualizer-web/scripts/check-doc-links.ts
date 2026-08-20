/**
 * Every point link in `docs/SUPPORTED_POINTS.md` opens a card, checked against the diagram itself.
 *
 * The doc links each of the 34 points to this app, and a link is only useful if the architecture in
 * it is one that *has* that point: a router on a dense family, `mlp_act` on a sparse layer or
 * `resid_streams` on a single-stream trunk all open a diagram with no card, which looks like the
 * link worked. So rather than trusting the URLs, this builds the graph each one asks for -- the same
 * `decodeLink` -> traits -> dims -> `buildGraph` path `useVisualizer` runs -- and asserts the node
 * is there.
 *
 * Both directions, because both rot: a link naming a point that no longer exists fails here, and so
 * does a point added to `points.ts` that the doc never linked.
 *
 * Usage:
 *   npx tsx scripts/check-doc-links.ts
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { DEFAULT_ARCHITECTURE_ID, architecture } from "@/data/architectures";
import { DEFAULT_DIMENSIONS } from "@/data/dimensions";
import { parseAddress } from "@/data/engines";
import { ALL_POINTS } from "@/data/points";
import { minLayersFor, resolveTraits } from "@/data/traits";
import { buildGraph } from "@/lib/buildGraph";
import { decodeLink } from "@/lib/link";
import type { TraitId } from "@/lib/types";

const here = dirname(fileURLToPath(import.meta.url));
const DOC = join(here, "..", "..", "docs", "SUPPORTED_POINTS.md");
const ORIGIN = "https://interp-engine.org/";

/** `[resid_post]: https://interp-engine.org/?arch=...&point=resid_post.2`, one per point. */
function definitions(markdown: string): { label: string; query: string }[] {
  const out: { label: string; query: string }[] = [];
  for (const line of markdown.split("\n")) {
    const match = /^\[(\w+)\]:\s*(\S+)\s*$/.exec(line);
    if (!match) continue;
    const [, label, url] = match;
    if (!url.startsWith(ORIGIN)) continue;
    out.push({ label, query: url.slice(ORIGIN.length) });
  }
  return out;
}

/** `linkedDims` + `clamp` from lib/state.ts: the depth a linked address needs to exist. */
function dimsFor(address: string, traits: Set<TraitId>) {
  const at = parseAddress(address);
  const dims = {
    ...DEFAULT_DIMENSIONS,
    layers: Math.max(DEFAULT_DIMENSIONS.layers, (at?.layer ?? 0) + 1),
    streams: Math.max(DEFAULT_DIMENSIONS.streams, (at?.stream ?? 0) + 1),
  };
  return {
    ...dims,
    layers: Math.max(dims.layers, minLayersFor(traits, dims.windowRatio)),
  };
}

const markdown = readFileSync(DOC, "utf8");
const defined = definitions(markdown);
const problems: string[] = [];
const linked = new Set<string>();

for (const { label, query } of defined) {
  const link = decodeLink(query);
  const asked = new URLSearchParams(query);

  // `decodeLink` drops what it cannot use, so a silently-dropped parameter is the failure to
  // report: the link would open, on the wrong architecture or with no card at all.
  if (asked.get("arch") !== null && link.arch === null) {
    problems.push(`[${label}] names arch=${asked.get("arch")}, which is not an architecture`);
    continue;
  }
  if (link.point === null) {
    problems.push(`[${label}] names point=${asked.get("point")}, which is not a drawable address`);
    continue;
  }

  const archId = link.arch ?? DEFAULT_ARCHITECTURE_ID;
  const traits = resolveTraits(architecture(archId)?.traits ?? []);
  const graph = buildGraph({ dims: dimsFor(link.point, traits), traits });
  const node = graph.nodes.find((candidate) => candidate.id === link.point);

  if (!node) {
    problems.push(`[${label}] -> ${link.point} is not drawn on ${archId}, so the card cannot open`);
    continue;
  }
  if (node.refusal) {
    problems.push(`[${label}] -> ${link.point} opens refused on ${archId}: ${node.refusal}`);
    continue;
  }
  linked.add(node.point);
}

for (const spec of ALL_POINTS) {
  if (!linked.has(spec.name)) {
    problems.push(`${spec.name} is a point with no working link in docs/SUPPORTED_POINTS.md`);
  }
}

if (problems.length > 0) {
  console.error(`[doc-links] ${problems.length} problem(s):`);
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exitCode = 1;
} else {
  console.log(`[doc-links] ${defined.length} links, every point drawn where it is linked`);
}
