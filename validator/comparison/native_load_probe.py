"""Ad-hoc probe: does transformers' own class load a checkpoint whose repo also ships modeling code?

Not part of the sweep. Answers the one question that decides whether preferring the native
implementation over a checkpoint's bundled `auto_map` code is safe for a given repo id: the class
existing is not the same claim as the weights fitting it, and a silently mismapped tensor is worse
than the ImportError it replaced. So this reports, per checkpoint, the three loading-info key sets
(missing / unexpected / mismatched) and then greedy-decodes the sweep prompt, since a checkpoint that
loads with no missing keys and still cannot finish "the capital of Japan is" has been mismapped
somewhere the key names did not show.

    PYTHONPATH=. .venv-cmp/bin/python -m comparison.native_load_probe <hf_id> [<hf_id> ...]
"""

from __future__ import annotations

import sys
import traceback
from typing import Any, cast

from comparison.spec import PROMPT


def probe(hf_id: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 78}\n{hf_id}\n{'=' * 78}", flush=True)
    try:
        # `output_loading_info=True` makes this return `(model, loading_info)` instead of the model,
        # and transformers does not model that as an overload -- so the tuple is spelled out here.
        loaded = cast(
            "tuple[Any, dict[str, list[str]]]",
            AutoModelForCausalLM.from_pretrained(
                hf_id,
                trust_remote_code=False,
                dtype="auto",
                output_loading_info=True,
                attn_implementation="eager",
            ),
        )
    except Exception:  # noqa: BLE001 - the point is to report which checkpoints cannot do this
        traceback.print_exc()
        print(f"[{hf_id}] NATIVE LOAD FAILED")
        return

    model, info = loaded

    print(f"  class          : {type(model).__name__} ({type(model).__module__})")
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        keys = info.get(name) or []
        # Truncated because a genuine mismatch shows up in the first few names; the count is the verdict.
        print(f"  {name:15s}: {len(keys)}" + (f"  e.g. {keys[:4]}" if keys else ""))

    model = model.to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False)
    ids = tok(PROMPT, return_tensors="pt").to("cuda")
    # One cache-free forward, which is also what a capture does: `use_cache=True` on nemotron_h raises
    # `KeyError: 'mlp'` building a DynamicCache for a layer type transformers' own mapping lacks, and
    # that is a generation-path bug rather than anything the compared activations depend on.
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True, use_cache=False)
    nonfinite = [i for i, h in enumerate(out.hidden_states) if not torch.isfinite(h).all()]
    top = tok.decode(out.logits[0, -1].argmax())
    print(f"  non-finite     : {nonfinite or 'none'}")
    print(f"  prompt         : {PROMPT!r}")
    print(f"  top next token : {top!r}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for hf_id in sys.argv[1:]:
        probe(hf_id)
