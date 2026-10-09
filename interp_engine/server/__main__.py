"""``python -m interp_engine.server``: load one model and serve it.

The bearer token comes from ``INTERP_ENGINE_SERVER_TOKEN`` or ``--token-file``, never from a
flag, because a flag shows in the process list. A server bound to a non-loopback address
refuses to start without a token unless ``--insecure-no-auth`` is given.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn

from interp_engine.load import BACKENDS
from interp_engine.server.app import create_app
from interp_engine.server.config import ServerConfig, is_loopback

TOKEN_ENV = "INTERP_ENGINE_SERVER_TOKEN"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m interp_engine.server", description="Load one model and serve it.")
    p.add_argument("--model", required=True, help="Hugging Face repo id")
    p.add_argument("--backend", default="auto", choices=BACKENDS)
    p.add_argument("--device", default=None, help="eager only, e.g. cuda:0, cpu")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--quantization", default="")
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--load-kwargs", default="{}", help="JSON object passed to load_model as keyword arguments")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token-file", type=Path, default=None)
    p.add_argument("--insecure-no-auth", action="store_true", help="serve a non-loopback address with no token")
    p.add_argument("--max-prompt-tokens", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-capture-points", type=int, default=256)
    p.add_argument("--max-directions", type=int, default=65536, help="direction rows per project request")
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--max-queue", type=int, default=64)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--lens-jacobians", default=None, help="a Jacobian lens .pt file: a local path or hf://<owner>/<repo>/<path>"
    )
    p.add_argument("--lens-jacobian-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--log-level", default="info")
    return p


def _token(args: argparse.Namespace) -> str | None:
    if args.token_file is not None:
        token = args.token_file.read_text().strip()
        if not token:
            raise SystemExit(f"{args.token_file} is empty")
        return token
    return os.environ.get(TOKEN_ENV) or None


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = _token(args)
    if token is None and not is_loopback(args.host) and not args.insecure_no_auth:
        raise SystemExit(
            f"--host {args.host} accepts connections from other machines, and no token is set. "
            f"Set {TOKEN_ENV} or --token-file, or pass --insecure-no-auth."
        )
    load_kwargs = json.loads(args.load_kwargs)
    if not isinstance(load_kwargs, dict):
        raise SystemExit("--load-kwargs must be a JSON object")
    config = ServerConfig(
        hf_model_id=args.model,
        backend=args.backend,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        num_gpus=args.num_gpus,
        load_kwargs=load_kwargs,
        token=token,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        max_capture_points=args.max_capture_points,
        max_directions=args.max_directions,
        max_concurrency=args.max_concurrency,
        max_queue=args.max_queue,
        warmup=not args.no_warmup,
        lens_jacobians=args.lens_jacobians,
        lens_jacobian_dtype=args.lens_jacobian_dtype,
    )
    # One worker: the model lives in this process, and a second worker would load a second copy.
    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main(sys.argv[1:])
