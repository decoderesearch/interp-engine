/**
 * The fitting cards, collapsed into the VRAM tiers people actually shop and
 * rent in. Nobody arrives asking whether an RTX A4000 works; they arrive with
 * "I have 16GB" or "I can rent an 80GB card", and a list of twenty-two rows,
 * several of which are the same board under three names, makes that harder to
 * read rather than easier.
 *
 * A tier is the **board** size — the number on the box. The catalog stores
 * *usable* bytes, which is always less and by an amount that depends on the
 * card: a 24 GiB board reports 23.6 GiB on a 4090 and 22.0 on an L4, because
 * the L4 is a datacenter part with ECC on, and ECC costs about a sixteenth of
 * the array. So the tier is found by rounding usable capacity **up** to the
 * nearest rung, and the rungs are listed rather than computed, since they are a
 * fact about what vendors sell and not an arithmetic series.
 *
 * Two rules keep a tier row from promising more than it can keep:
 *
 * 1. It is split by shard count as well as by tier. Cards on the same rung do
 *    not always need the same number of GPUs — the L4's missing 1.6 GiB is
 *    enough to push a model that fits one 4090 onto two L4s — and one row
 *    reading "24GB" would have to lie about one of them.
 * 2. Everything it reports is priced from the **smallest** card still in the
 *    group, so the context, concurrency and utilisation hold for every card the
 *    row names, not just the roomiest. That card is listed first for the same
 *    reason.
 *
 * Cards that do not fit at all are not in `results` and so are never named.
 */

import type { Gpu } from "@/data/gpus.generated";
import type { ModelMemoryFacts } from "@/lib/hub";
import { evidenceFor, type Evidence, type FitResult } from "@/lib/size";

/**
 * Board capacities cards are sold in, in GiB. Anything past the end of the
 * ladder is rounded up to a multiple of 16 rather than dropped, so a card added
 * to the catalog above today's largest still lands somewhere sensible.
 */
const RUNGS = [16, 20, 24, 32, 40, 48, 64, 80, 96, 141, 180, 288];

export function tierGib(gpu: Gpu): number {
  const usable = gpu.totalBytes / 2 ** 30;
  return RUNGS.find((rung) => usable <= rung) ?? Math.ceil(usable / 16) * 16;
}

/**
 * The name people use, from the name the driver reports.
 *
 * "NVIDIA RTX PRO 6000 Blackwell Server Edition" is what `nvidia-smi` says and
 * "RTX Pro 6000" is what anyone shopping for one says, and the row has space for
 * the second. What comes off is the vendor, the marketing suffixes, and the
 * *interconnect* — `SXM4` and `PCIe` and `HBM3` distinguish two builds of one
 * card, which matters when you are buying and not when you are asking how much
 * VRAM you need. What stays is anything that changes the capacity: `NVL` is a
 * different amount of memory from the plain part, and `80GB` is the whole point.
 *
 * Shortening deliberately collapses some pairs to one name — an A100 80GB PCIe
 * and an A100-SXM4-80GB are both "A100 80GB" — so a caller listing several cards
 * has to de-duplicate afterwards. `examples()` in the sizer does.
 */
export function shortGpuName(name: string): string {
  return (
    name
      .replace(/^NVIDIA /, "")
      .replace(/^(GeForce|Tesla) /, "")
      // A100-SXM4-80GB -> A100 80GB. The board size is the half worth keeping.
      .replace(/-SXM\d*-/, " ")
      .replace(/ (PCIe|HBM\d)\b/g, "")
      .replace(/ Blackwell (Server|Workstation)/, "")
      .replace(/ (Generation|Edition)$/, "")
      .replace(/\bPRO\b/, "Pro")
  );
}

export interface Tier {
  /** Board size, in GiB. The number the row leads with. */
  gib: number;
  count: number;
  /** The smallest card in the group, and what every figure on the row is from. */
  result: FitResult;
  /**
   * Every card on this rung that fits at this count, smallest first, each with
   * its own fit.
   *
   * The whole result and not just the `Gpu`, because a card's evidence is only
   * evidence for the settings *it* was fitted at — see {@link tierEvidence},
   * which could not be written against a bare list of names.
   */
  members: FitResult[];
}

/** The cards a row names. */
export function tierGpus(tier: Tier): Gpu[] {
  return tier.members.map((member) => member.gpu);
}

export function byTier(results: FitResult[]): Tier[] {
  const groups = new Map<string, Tier>();

  for (const result of results) {
    const gib = tierGib(result.gpu);
    const key = `${gib}/${result.count}`;
    const group = groups.get(key);
    if (!group) {
      groups.set(key, {
        gib,
        count: result.count,
        result,
        members: [result],
      });
      continue;
    }
    group.members.push(result);
    if (result.gpu.totalBytes < group.result.gpu.totalBytes) {
      group.result = result;
    }
  }

  for (const group of groups.values()) {
    group.members.sort((a, b) => a.gpu.totalBytes - b.gpu.totalBytes);
  }

  // Largest card first. The list reads as a ladder either way, and this is the
  // end people scan from: the rows that matter are the ones naming hardware you
  // might actually be sitting in front of or renting, and those are at the top
  // of the range. Ascending buried an H100 row under six rungs of small cards
  // that fit only because the model is tiny. Within a rung, fewest cards first,
  // since 2x of the same board is the fallback rather than the offer.
  return [...groups.values()].sort(
    (a, b) => b.gib - a.gib || a.count - b.count,
  );
}

/**
 * The row to open on: the smallest **single** card that holds the model.
 *
 * Not the first row, which is now the largest and would open the panel on an
 * H100 for a 1.5B model. The cheapest thing that works is the answer most
 * people came for, and it is also the most informative default -- the headroom
 * figures on it are the tight ones, so a reader who wants more context can see
 * immediately what it would cost, which the roomiest card hides.
 *
 * Falls back to the fewest cards, then the smallest, for a model that fits
 * nothing on its own. There, "smallest single card" does not exist and the
 * cheapest working configuration is still the right thing to lead with.
 */
/**
 * What hardware has to say about a row, asked of **every** card on it.
 *
 * Asking only the card the row reports from loses most of the evidence there is,
 * because the reporting card is the smallest on the rung and the measured one
 * rarely is: of the eighteen passing runs in the catalog, eleven sat on a row
 * whose figures come from a different card, and every one of them showed as
 * "estimated". The A40 is the clearest case — nearly everything in
 * `VERIFIED.md` was run on one, and it shares its rung with the slightly
 * smaller L40S, which is therefore what the row quotes.
 *
 * Each card is matched against its **own** fit, since a run vouches for the
 * settings it was measured at and a bigger card on the same rung is fitted to a
 * wider context. Where the card is not the one the row reports from, the badge
 * names it, so "verified on A40" beside L40S figures is a claim about the A40
 * rather than a promise about the L40S.
 *
 * A failure outranks a confirmation. Two cards on one rung can disagree, and
 * between "this has been made to work" and "this has been seen to break" the
 * second is the one worth interrupting someone with.
 */
export function tierEvidence(facts: ModelMemoryFacts, tier: Tier): Evidence {
  let verified: Evidence | null = null;

  for (const member of tier.members) {
    const found = evidenceFor(facts, member.gpu, member.estimate.spec);
    if (found.kind === "estimated") continue;

    const named =
      member.gpu.name === tier.result.gpu.name
        ? found
        : {
            ...found,
            label: `${found.label} on ${shortGpuName(member.gpu.name)}`,
          };

    if (found.kind === "fails") return named;
    verified ??= named;
  }

  return verified ?? { kind: "estimated", label: "estimated" };
}

export function defaultTier(tiers: Tier[]): Tier | undefined {
  const single = tiers.filter((tier) => tier.count === 1);
  const pool = single.length ? single : tiers;
  return pool.reduce<Tier | undefined>(
    (best, tier) =>
      !best ||
      tier.count < best.count ||
      (tier.count === best.count && tier.gib < best.gib)
        ? tier
        : best,
    undefined,
  );
}
