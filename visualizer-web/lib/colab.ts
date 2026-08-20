/**
 * Where a snippet goes to be run.
 *
 * Colab opens a notebook that is already on GitHub and nothing else. No URL carries
 * notebook content, there is no import-from-any-URL parameter, and creating a Gist to
 * hold one would mean a credential and a public artefact per click. So the notebook is
 * a **template committed to this repository** and the snippet travels beside it, on the
 * clipboard: `NotebookButton` writes it, and the template's last cell is where it goes.
 *
 * Three templates. Two of them differ only in the install — `interp-engine[vllm]`,
 * which is a multi-gigabyte CUDA wheel, against plain `interp-engine`, which is not —
 * and `vllm async` maps to the vLLM one, since it is a method on a vLLM model and needs
 * the same extra as the free function does. `vllm static` gets its own rather than
 * sharing the vLLM template because what a reader needs before pasting is different
 * there and not smaller: its taps are fixed at load, so the snippet names the point
 * twice and a second point means reloading, and it wants more of the card than the
 * hooked backend does.
 *
 * The path below is a **public contract in two directions**. Colab reads it from GitHub's
 * `main` rather than from this deployment, so a template renamed or moved in the same
 * commit as this file is still a 404 in a new tab until that commit is pushed — and
 * nothing here can detect one. It also ends up in browser history and in links people
 * send each other, which is why the templates sit at the repository root instead of
 * under this app.
 *
 * **`decoderesearch/interp-engine` has to be public for any of this to work.** Colab
 * fetches a `/github/` path anonymously, with no authorization step, which is the whole
 * reason that form exists; against a private repository a reader gets a sign-in wall in a
 * new tab instead of a notebook. The repository was still private when this was written.
 */

import type { Variant } from "@/data/snippets";

const COLAB_GITHUB = "https://colab.research.google.com/github";
const TEMPLATE_DIR = "decoderesearch/interp-engine/blob/main/notebooks";

const TEMPLATE: Record<Variant, string> = {
  vllm: "interp_engine_vllm.ipynb",
  "vllm-async": "interp_engine_vllm.ipynb",
  "vllm-static": "interp_engine_vllm_static.ipynb",
  eager: "interp_engine_eager.ipynb",
};

/** The Colab URL for the template that installs what `variant` needs. */
export function colabTemplateUrl(variant: Variant): string {
  return `${COLAB_GITHUB}/${TEMPLATE_DIR}/${TEMPLATE[variant]}`;
}
