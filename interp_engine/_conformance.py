"""Pyright's half of the API contract: each backend, returned as the generated ``EngineAPI``.

Nothing here runs. A backend whose method types drift from ``api/engine.yaml`` fails ``make
check-type`` on the matching line; ``tests/test_api_generated.py`` checks names, kinds and defaults,
which pyright does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from interp_engine.api import EngineAPI
    from interp_engine.model import EagerModel
    from interp_engine.vllm_backend import VLLMModel

    def _eager(model: EagerModel) -> EngineAPI:
        return model

    def _vllm(model: VLLMModel) -> EngineAPI:
        return model
