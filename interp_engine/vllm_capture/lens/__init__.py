"""Everything the lens does inside the worker: intervention, unembedding, read-out.

jlens is one feature in three parts, and until this package existed they sat ~1200 lines apart
in a single module::

    unembed     locating the final norm and the (possibly sharded) head
       |
    readout     transport through J_bar, unembed, top-k -- what the endpoint serves
    intervene   the write-hook that steers/ablates/swaps a residual

``intervene`` is independent of the other two: it changes the forward,
:mod:`~interp_engine.vllm_capture.requests` installs it per request, and the read-out then
observes whatever the forward produced. Keeping it here anyway is deliberate -- a steer and
the read-out that shows its effect are one feature to whoever is changing them.
"""

from __future__ import annotations

# Internals with callers outside this subpackage -- the demux builds the intervention modifier,
# and tests reach for the unembedding helpers directly. See the note in the parent package's
# ``__init__``: patch the defining module, not one of these aliases.
from interp_engine.vllm_capture.lens.intervene import _make_lens_modifier as _make_lens_modifier
from interp_engine.vllm_capture.lens.intervene import worker_install_lens_intervention
from interp_engine.vllm_capture.lens.readout import _lens_topk as _lens_topk
from interp_engine.vllm_capture.lens.readout import (
    worker_lens_capture_readout,
    worker_lens_readout,
    worker_lens_transport,
    worker_set_lens_jacobians,
)
from interp_engine.vllm_capture.lens.unembed import (
    _assert_applied_logit_scale_agrees as _assert_applied_logit_scale_agrees,
)
from interp_engine.vllm_capture.lens.unembed import _local_lm_head_rows as _local_lm_head_rows
from interp_engine.vllm_capture.lens.unembed import _worker_final_norm as _worker_final_norm
from interp_engine.vllm_capture.lens.unembed import (
    _worker_logits_processor as _worker_logits_processor,
)
from interp_engine.vllm_capture.lens.unembed import _worker_unembed_layer as _worker_unembed_layer
from interp_engine.vllm_capture.lens.unembed import _worker_unembed_weight as _worker_unembed_weight
from interp_engine.vllm_capture.lens.unembed import (
    merge_lm_head_row_payloads,
    worker_lm_head_rows,
    worker_unembed,
)

__all__ = [
    "merge_lm_head_row_payloads",
    "worker_install_lens_intervention",
    "worker_lens_capture_readout",
    "worker_lens_readout",
    "worker_lens_transport",
    "worker_lm_head_rows",
    "worker_set_lens_jacobians",
    "worker_unembed",
]
