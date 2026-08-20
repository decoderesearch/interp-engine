/**
 * Is a cheaper model good enough to be Riz?
 *
 * Anthropic's prices and model lineup both move, and the honest answer to
 * "could we use the small one" is a run rather than an opinion. This asks two
 * models the same questions through the *real* prompt -- `SYSTEM_PROMPT` and
 * `KNOWLEDGE` imported, not paraphrased -- so a pass here is evidence about
 * the deployed configuration and not about a reconstruction of it.
 *
 *   npm run eval
 *   npm run eval -- claude-haiku-4-5 claude-opus-5
 *
 * The cases are questions with a checkable token in the answer, drawn from
 * places the docs say a wrong answer is expensive: the `hook_mlp_out` collision
 * that silently mis-trains an SAE, the unimplemented/unreachable split that
 * decides whether you file a bug or switch backend, and two facts that changed
 * after the models' training cutoffs, where the failure is confident staleness
 * rather than a blank.
 *
 * Scoring is substring matching on identifiers, which is why every `expect` is
 * a symbol rather than a phrase. It cannot tell a good explanation from a
 * lucky one, so the transcript is printed too -- read it before believing the
 * table.
 *
 * `forbid` is the part to be careful with, and the first three runs of this
 * file were all corrections to it rather than findings about a model. A good
 * answer names the wrong tensor in order to say it is the wrong tensor, so
 * forbidding the confusable identifier scores warnings as errors and rewards
 * whichever model explained least. Forbid a claim, never a name.
 */

import { anthropic } from "@ai-sdk/anthropic";
import { generateText } from "ai";

import { KNOWLEDGE } from "../knowledge/index.js";
import { SYSTEM_PROMPT } from "../knowledge/prompt.js";

interface Case {
  id: string;
  question: string;
  /** Every one of these must appear. Identifiers, not prose. */
  expect: RegExp[];
  /** None of these may appear. */
  forbid?: RegExp[];
  /** What a model that half-knows this answers instead. */
  trap: string;
}

const CASES: Case[] = [
  {
    id: "tlens-name",
    question:
      "In TransformerLens, what is the hook name for interp-engine's mlp_act point at layer 5?",
    // Same lesson as `sae-collision`: no forbid on the confusable name. Both
    // models named `hook_mlp_out` here at some point purely to warn it is not
    // the answer, and a scorer that punishes that is measuring terseness.
    expect: [/mlp\.hook_post/],
    trap: "Confusing the neuron activations with the block's MLP output.",
  },
  {
    id: "sae-collision",
    question:
      "I have an SAE for gemma-2-2b that was trained on the TransformerLens hook blocks.4.hook_mlp_out. Which interp-engine point gives me that exact tensor?",
    // No forbid on a bare `mlp_out`. The first version of this case had one,
    // and both models tripped it while answering correctly: naming the raw
    // point is how you explain which tensor you are *not* getting. A regex
    // cannot tell that apart from recommending it, so the requirement is the
    // right point plus the reason, and the transcript settles the rest.
    expect: [/\bmlp_out_post\b/, /sandwich|post-sublayer|after the post/i],
    trap: "Answering raw mlp_out. Gemma-2 is sandwich-norm, so that is a different tensor -- the documented mistake that encodes an SAE off data it was never trained on.",
  },
  {
    id: "mlp-pre-vllm",
    question:
      "Can the vLLM backend capture mlp_pre? If not, is it unimplemented or unreachable, and why?",
    expect: [/unreachable/i, /gate_up_proj/],
    trap: "Saying 'unimplemented', which would send someone to file a bug for a tensor a fused kernel never forms.",
  },
  {
    id: "attn-gate-vllm",
    question: "Is attn_gate available on the vLLM backend?",
    expect: [/unimplemented/i],
    trap: "Saying 'unreachable', the opposite error to the case above.",
  },
  {
    id: "projection-cap",
    question:
      "Does projection_cap steering run on the eager backend, or is it vLLM-only?",
    expect: [/eager/i],
    forbid: [/raises NotImplementedError|not implemented on eager|only.{0,20}vLLM/i],
    trap: "Reporting the NotImplementedError it raised for one release. Stale-but-confident, and the docs mention the old behaviour in the past tense.",
  },
  {
    id: "no-vllm-kwarg",
    question: "How do I pass vllm=True to load_model to get the vLLM backend?",
    expect: [/backend=/],
    trap: "Inventing the argument the question presupposes rather than correcting it.",
  },
  {
    id: "hook-normalized",
    question:
      "TransformerLens ln1.hook_normalized has no interp-engine point. Which two canonical points does it sit between, and how do I recover the tensor itself?",
    // `ln1` is the pre-attention norm, so it is bracketed by resid_pre and
    // attn_in. Naming resid_mid here is the ln2 answer to an ln1 question --
    // the specific way this one goes wrong, and worth a case of its own
    // because the docs spell out ln2 and leave ln1 to be reasoned about.
    expect: [/resid_pre/, /attn_in/, /pre_gain_normalized/],
    trap: "Answering resid_mid, which brackets ln2 rather than ln1. The docs name the ln2 case explicitly, so a model that pattern-matches instead of reading gets a confident wrong answer.",
  },
  {
    id: "steer-methods",
    question: "What steering methods does interp-engine support?",
    // Either vocabulary counts. `STEER_METHODS` holds the strings, but the
    // spec classes are what a caller actually writes, and answering with those
    // is if anything the more useful reply -- an earlier version of this case
    // demanded the strings and failed both models for being helpful.
    expect: [
      /additive|AddSpec/i,
      /orthogonal|OrthogonalDecompSpec/i,
      /projection_cap|ProjectionCapSpec/i,
    ],
    // Three, and only three. A fourth plausible name is the failure here.
    forbid: [/ablat|clamp_spec|ScaleSpec|four steering/i],
    trap: "Adding a plausible fourth method that does not exist.",
  },
];

/** Dollars per million. Keyed by every slug this is likely to be handed. */
const PRICES: Record<string, { cacheRead: number; out: number }> = {
  "claude-sonnet-5": { cacheRead: 0.2, out: 10 },
  "claude-haiku-4-5": { cacheRead: 0.1, out: 5 },
  "claude-haiku-4-5-20251001": { cacheRead: 0.1, out: 5 },
};

async function ask(model: string, question: string) {
  const started = Date.now();
  const result = await generateText({
    model: anthropic(model),
    system: [
      {
        role: "system",
        content: `${SYSTEM_PROMPT}\n\n${KNOWLEDGE}`,
        providerOptions: {
          anthropic: { cacheControl: { type: "ephemeral", ttl: "1h" } },
        },
      },
    ],
    messages: [{ role: "user", content: question }],
    maxOutputTokens: 1200,
    temperature: 0.2,
  });
  return { text: result.text, usage: result.usage, ms: Date.now() - started };
}

async function main() {
  // The deployed default first, so the column a candidate has to beat is the
  // one actually serving readers rather than whatever was default when this
  // was written.
  const models = process.argv.slice(2).length
    ? process.argv.slice(2)
    : ["claude-haiku-4-5", "claude-sonnet-5"];

  const score: Record<string, { pass: number; ms: number; cost: number }> = {};
  for (const model of models) score[model] = { pass: 0, ms: 0, cost: 0 };

  for (const testCase of CASES) {
    console.log(`\n${"=".repeat(78)}\n${testCase.id}: ${testCase.question}`);
    console.log(`trap: ${testCase.trap}`);

    for (const model of models) {
      const { text, usage, ms } = await ask(model, testCase.question);

      const missing = testCase.expect.filter((re) => !re.test(text));
      const tripped = (testCase.forbid ?? []).filter((re) => re.test(text));
      const ok = missing.length === 0 && tripped.length === 0;
      if (ok) score[model].pass++;
      score[model].ms += ms;

      const price = PRICES[model];
      if (price) {
        const read = usage.inputTokenDetails?.cacheReadTokens ?? 0;
        const written = usage.inputTokenDetails?.cacheWriteTokens ?? 0;
        score[model].cost +=
          (read * price.cacheRead) / 1e6 +
          (written * price.cacheRead * 20) / 1e6 +
          ((usage.outputTokens ?? 0) * price.out) / 1e6;
      }

      console.log(`\n--- ${model} ${ok ? "PASS" : "FAIL"} (${ms}ms)`);
      if (missing.length) console.log(`    missing: ${missing.join(", ")}`);
      if (tripped.length) console.log(`    forbidden: ${tripped.join(", ")}`);
      console.log(
        text
          .split("\n")
          .map((line) => `    ${line}`)
          .join("\n"),
      );
    }
  }

  console.log(`\n${"=".repeat(78)}\nSCORE over ${CASES.length} cases\n`);
  for (const model of models) {
    const { pass, ms, cost } = score[model];
    console.log(
      `  ${model.padEnd(28)} ${pass}/${CASES.length}  ` +
        `${Math.round(ms / CASES.length)}ms avg  $${cost.toFixed(3)} total`,
    );
  }
}

// Called rather than top-level awaited: tsx transforms this to CJS, where a
// top-level await is a syntax error.
main().catch((error: unknown) => {
  console.error(error);
  process.exit(1);
});
