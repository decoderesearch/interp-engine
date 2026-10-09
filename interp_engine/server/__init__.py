"""An HTTP server for one model: the engine operations, with auth and request limits.

    python -m interp_engine.server --model openai-community/gpt2

Needs the ``server`` extra (``pip install 'interp-engine[server]'``).

Endpoints: ``GET /health`` (no auth), ``GET /v1/describe``, and ``POST`` to ``/v1/tokenize``,
``/v1/serves``, ``/v1/capture``, ``/v1/project``, ``/v1/generate`` and ``/v1/lens`` (NDJSON
streams), ``/v1/unembed_rows`` and ``/v1/decode_residuals``. The OpenAPI schema is at
``/openapi.json``.
"""

from __future__ import annotations

try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
except ImportError as error:  # pragma: no cover - depends on the install
    raise ImportError(
        "interp_engine.server needs FastAPI and uvicorn. Install the extra: pip install 'interp-engine[server]'"
    ) from error

from interp_engine.server.app import create_app
from interp_engine.server.config import ServerConfig

__all__ = ["ServerConfig", "create_app"]
