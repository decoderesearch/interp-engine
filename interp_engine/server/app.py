"""The FastAPI app: one model, the engine operations as HTTP endpoints.

The endpoints follow ``api/engine.yaml``. The server adds auth, request limits and a wire
format, and nothing else: no model ids other than the Hugging Face one, no product concepts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import secrets
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from interp_engine.address import to_address
from interp_engine.api import MANIFEST_DIGEST, MANIFEST_VERSION, LensSpec
from interp_engine.server import lens
from interp_engine.server.config import ServerConfig
from interp_engine.server.runner import Busy, EngineRunner
from interp_engine.server.schemas import (
    CaptureRequest,
    CaptureResponse,
    DecodeResidualsRequest,
    GenerateRequest,
    GenerateResponse,
    LensRequest,
    LensResponse,
    ProjectRequest,
    ProjectResponse,
    ServesRequest,
    ServesResponse,
    TensorResponse,
    TokenizeRequest,
    TokenizeResponse,
    TopKResponse,
    UnembedRowsRequest,
    decode_tensor,
    encode_tensor,
)
from interp_engine.steer import steer
from interp_engine.steer_specs import SteeringSpec

logger = logging.getLogger(__name__)

Status = Literal["loading", "ready", "failed"]


@dataclasses.dataclass
class _State:
    status: Status = "loading"
    error: str | None = None
    model: Any = None
    backend: str = ""
    vocab_limit: int = 0
    jacobian_layers: list[int] = dataclasses.field(default_factory=list)
    word_mask: torch.Tensor | None = None


def _hf_tokenizer(model: Any) -> Any:
    return getattr(model.tokenizer, "tokenizer", model.tokenizer)


def _vocab_limit(model: Any) -> int:
    """Token ids at or above this are refused before they reach a device index."""
    return len(_hf_tokenizer(model))


def _default_loader(config: ServerConfig) -> Callable[[], Any]:
    def load() -> Any:
        from interp_engine.load import load_model

        return load_model(config.hf_model_id, **config.load_args())

    return load


def create_app(config: ServerConfig, *, loader: Callable[[], Any] | None = None) -> FastAPI:
    """The app for ``config``. ``loader`` replaces :func:`~interp_engine.load_model`, for tests."""
    state = _State()
    runner = EngineRunner()
    load = loader or _default_loader(config)

    async def _load() -> None:
        try:
            model = await runner.call(load)
            if config.warmup:
                await runner.run(model.warmup())
            description = await runner.call(model.describe)
            state.backend = description.backend
            state.vocab_limit = await runner.call(_vocab_limit, model)
            if config.lens_jacobians:
                jacobians = await runner.call(
                    lens.load_jacobians,
                    config.lens_jacobians,
                    n_layers=model.n_layers,
                    d_model=model.d_model,
                    dtype=getattr(torch, config.lens_jacobian_dtype),
                    # Eager reads J_bar where it is; the other backends move it to their read-out.
                    device=model.device if state.backend == "eager" else None,
                )
                held = await runner.run(model.set_lens_jacobians(jacobians))
                state.jacobian_layers = sorted(jacobians)
                logger.info("Holding J_bar for %d layers (%d bytes)", len(jacobians), held)
            runner.set_limits(config.concurrency_for(state.backend), config.max_queue)
            state.model = model
            state.status = "ready"
            logger.info("Loaded %s on %s", config.hf_model_id, state.backend)
        except Exception as error:
            state.status = "failed"
            state.error = f"{type(error).__name__}: {error}"
            logger.exception("Loading %s failed", config.hf_model_id)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(_load())
        try:
            yield
        finally:
            if not task.done():
                task.cancel()
            if state.model is not None:
                try:
                    await runner.run(state.model.shutdown())
                except Exception:
                    logger.exception("Model shutdown failed")
            runner.close()

    app = FastAPI(title="interp-engine server", lifespan=lifespan)

    def authorized(authorization: str | None = Header(default=None)) -> None:
        if config.token is None:
            return
        scheme, _, given = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(given.encode(), config.token.encode()):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")

    def ready(_: None = Depends(authorized)) -> Any:
        if state.status != "ready":
            detail = state.error if state.status == "failed" else "the model is still loading"
            raise HTTPException(status_code=503, detail=detail)
        return state.model

    @asynccontextmanager
    async def heavy() -> AsyncIterator[None]:
        await runner.acquire()
        try:
            yield
        finally:
            runner.release()

    def check_prompt(ids: list[int]) -> None:
        if len(ids) > config.max_prompt_tokens:
            raise ValueError(f"prompt has {len(ids)} tokens; this server accepts {config.max_prompt_tokens}")
        bad = [t for t in ids if not 0 <= t < state.vocab_limit]
        if bad:
            raise ValueError(f"token ids {bad[:5]} are outside this model's vocabulary of {state.vocab_limit}")

    def steering_spec(
        model: Any, req: CaptureRequest | GenerateRequest | ProjectRequest | LensRequest
    ) -> list[SteeringSpec] | None:
        return None if req.steering is None else [s.to_spec(model.d_model) for s in req.steering]

    async def open_heavy(agen: AsyncIterator[Any]) -> AsyncGenerator[Any, None]:
        """A heavy slot and the stream's first item. The caller releases the slot when it is done."""
        await runner.acquire()
        try:
            return await runner.open(agen)
        except BaseException:
            runner.release()
            raise

    def ndjson(
        items: AsyncGenerator[dict[str, Any], None], start: dict[str, Any], done: Callable[[int], dict[str, Any]]
    ) -> StreamingResponse:
        """Stream ``start``, each item, then ``done(n_items)``, and release the slot ``open_heavy`` took."""

        async def lines() -> AsyncIterator[bytes]:
            n = 0
            try:
                yield _line(start)
                async for body in items:
                    n += 1
                    yield _line(body)
                yield _line(done(n))
            except Exception as error:
                logger.exception("Stream failed")
                yield _line({"type": "error", "detail": f"{type(error).__name__}: {error}"})
            finally:
                await items.aclose()
                runner.release()

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.exception_handler(ValueError)
    async def _refused(_: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(Busy)
    async def _busy(_: Request, error: Busy) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)}, headers={"Retry-After": "1"})

    @app.get("/health")
    async def health() -> JSONResponse:
        body: dict[str, Any] = {"status": state.status, "model": config.hf_model_id}
        if state.error:
            body["error"] = state.error
        return JSONResponse(status_code=200 if state.status == "ready" else 503, content=body)

    @app.get("/v1/describe")
    async def describe(model: Any = Depends(ready)) -> dict[str, Any]:
        from interp_engine import __version__

        description = await runner.call(model.describe)
        fields = {
            f.name: [str(a) for a in value] if f.name in ("static_points", "static_writes") else value
            for f in dataclasses.fields(description)
            for value in [getattr(description, f.name)]
        }
        return {
            "description": fields,
            "recommended_sampling": dataclasses.asdict(await runner.call(lambda: model.recommended_sampling)),
            "has_chat_template": await runner.call(model.tok.has_chat_template),
            "engine_version": __version__,
            "manifest": {"version": MANIFEST_VERSION, "digest": MANIFEST_DIGEST},
            "load": {
                "backend": state.backend,
                "dtype": config.dtype,
                "quantization": config.quantization,
                "kv_cache_dtype": config.kv_cache_dtype,
                "num_gpus": config.num_gpus,
            },
            "lens": {"jacobian_layers": state.jacobian_layers, "jacobian_source": config.lens_jacobians},
            "limits": {
                "max_prompt_tokens": config.max_prompt_tokens,
                "max_new_tokens": config.max_new_tokens,
                "max_capture_points": config.max_capture_points,
                "max_directions": config.max_directions,
                "max_concurrency": config.concurrency_for(state.backend),
            },
        }

    @app.post("/v1/tokenize")
    async def tokenize(req: TokenizeRequest, model: Any = Depends(ready)) -> TokenizeResponse:
        def _run() -> TokenizeResponse:
            spans = None
            if req.text is not None:
                ids = [int(t) for t in model.to_tokens(req.text, prepend_bos=req.prepend_bos)[0].tolist()]
            else:
                messages = [m.model_dump() for m in req.messages or []]
                options = {
                    "add_generation_prompt": req.add_generation_prompt,
                    "continue_final_message": req.continue_final_message,
                    **req.template_kwargs,
                }
                ids = list(model.tok.apply_chat_template(messages, tokenize=True, **options))
                if req.spans:
                    spans = [dataclasses.asdict(s) for s in model.tok.message_spans(messages, **options)]
            str_tokens = model.to_str_tokens(torch.tensor(ids, dtype=torch.long)) if ids else []
            return TokenizeResponse(token_ids=ids, str_tokens=str_tokens, spans=spans)

        return await runner.call(_run)

    @app.post("/v1/serves")
    async def serves(req: ServesRequest, model: Any = Depends(ready)) -> ServesResponse:
        def _run() -> dict[str, str | None]:
            out: dict[str, str | None] = {}
            for point in req.points:
                try:
                    out[point] = model.refuses(point)
                except ValueError as error:
                    out[point] = str(error)
            return out

        return ServesResponse(refusals=await runner.call(_run))

    @app.post("/v1/capture")
    async def capture(req: CaptureRequest, model: Any = Depends(ready)) -> CaptureResponse:
        check_prompt(req.prompt_token_ids)
        if len(req.points) > config.max_capture_points:
            raise ValueError(f"{len(req.points)} points requested; this server accepts {config.max_capture_points}")
        addresses = [to_address(p) for p in req.points]
        spec = steering_spec(model, req)
        async with heavy():
            acts = await runner.run(model.capture(req.prompt_token_ids, addresses, steering_spec=spec, rows=req.rows))
        return CaptureResponse(activations={str(a): encode_tensor(t, req.output_dtype) for a, t in acts.items()})

    @app.post("/v1/generate", response_model=None)
    async def generate(req: GenerateRequest, model: Any = Depends(ready)) -> StreamingResponse | GenerateResponse:
        check_prompt(req.prompt_token_ids)
        if req.max_tokens > config.max_new_tokens:
            raise ValueError(f"max_tokens is {req.max_tokens}; this server accepts {config.max_new_tokens}")
        spec = steering_spec(model, req)
        knobs = {"temperature": req.temperature, "top_k": req.top_k, "top_p": req.top_p}
        knobs["presence_penalty"] = req.presence_penalty
        sampling = (await runner.call(lambda: model.sampling_settings(**knobs))).as_dict()
        eos_id = getattr(getattr(model.tokenizer, "tokenizer", model.tokenizer), "eos_token_id", None)

        async def steps() -> AsyncIterator[Any]:
            stream = model.generate_steps(
                req.prompt_token_ids,
                max_tokens=req.max_tokens,
                stop_at_eos=req.stop_at_eos,
                n_logprobs=req.n_logprobs,
                seed=req.seed,
                **knobs,
            )
            if spec is None:
                async for step in stream:
                    yield step
                return
            with steer(model, spec, prompt_token_ids=req.prompt_token_ids, generated=req.steer_generated):
                async for step in stream:
                    yield step

        def finish(ids: list[int]) -> Literal["eos", "length"]:
            if len(ids) < req.max_tokens or (ids and ids[-1] == eos_id):
                return "eos"
            return "length"

        def step_body(step: Any) -> dict[str, Any]:
            body: dict[str, Any] = {"type": "token", "token_id": step.token_id, "token_str": step.token_str}
            if step.logprobs is not None:
                body["logprobs"] = step.logprobs
            return body

        ids: list[int] = []

        async def bodies() -> AsyncIterator[dict[str, Any]]:
            async for step in steps():
                ids.append(step.token_id)
                yield step_body(step)

        items = await open_heavy(bodies())
        if not req.stream:
            try:
                whole = [_untyped(b) async for b in items]
            finally:
                runner.release()
            return GenerateResponse(
                text="".join(b["token_str"] for b in whole),
                token_ids=ids,
                steps=whole,
                sampling=sampling,
                finish=finish(ids),
            )
        return ndjson(
            items,
            {"type": "start", "sampling": sampling},
            lambda n: {"type": "done", "finish": finish(ids), "n_tokens": n},
        )

    @app.post("/v1/project")
    async def project(req: ProjectRequest, model: Any = Depends(ready)) -> ProjectResponse:
        check_prompt(req.prompt_token_ids)
        if len(req.directions) > config.max_capture_points:
            raise ValueError(
                f"{len(req.directions)} direction sets sent; this server accepts {config.max_capture_points}"
            )
        sets = [d.to_engine() for d in req.directions]
        rows = sum(int(s.vectors.shape[0]) if s.vectors.dim() == 2 else 0 for s in sets)
        if rows > config.max_directions:
            raise ValueError(f"{rows} direction rows sent; this server accepts {config.max_directions}")
        spec = steering_spec(model, req)
        async with heavy():
            values = await runner.run(model.project(req.prompt_token_ids, sets, steering_spec=spec))
        return ProjectResponse(values=[encode_tensor(v, req.output_dtype) for v in values])

    @app.post("/v1/lens", response_model=None)
    async def lens_readout(req: LensRequest, model: Any = Depends(ready)) -> StreamingResponse | LensResponse:
        check_prompt(req.prompt_token_ids)
        if req.max_tokens > config.max_new_tokens:
            raise ValueError(f"max_tokens is {req.max_tokens}; this server accepts {config.max_new_tokens}")
        spec = steering_spec(model, req)
        mask = None
        if req.words_only:
            if state.word_mask is None:
                state.word_mask = await runner.call(lens.word_mask, _hf_tokenizer(model), state.vocab_limit)
            mask = state.word_mask

        def step_body(step: Any) -> dict[str, Any]:
            flat = [step.token_id, *(int(t) for ids in step.top_ids for t in ids.flatten().tolist())]
            strs = model.to_str_tokens(torch.tensor(flat, dtype=torch.long))
            reads, at = [], 1
            for ids, probs in zip(step.top_ids, step.top_probs, strict=True):
                n_layers, k = int(ids.shape[0]), int(ids.shape[1])
                reads.append(
                    {
                        "top_ids": ids.tolist(),
                        "top_probs": probs.tolist(),
                        "top_strs": [strs[at + i * k : at + (i + 1) * k] for i in range(n_layers)],
                    }
                )
                at += n_layers * k
            return {
                "type": "step",
                "position": step.position,
                "token_id": step.token_id,
                "token_str": strs[0],
                "is_generated": step.is_generated,
                "lenses": reads,
            }

        async def bodies() -> AsyncIterator[dict[str, Any]]:
            stream = model.generate_with_lens(
                req.prompt_token_ids,
                [LensSpec(layers=s.layers, jacobian=s.jacobian) for s in req.lenses],
                point=req.point,
                top_n=req.top_n,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                seed=req.seed,
                word_mask=mask,
                skip_before=req.skip_before,
                stream_reduce=req.stream_reduce,
                stream_index=req.stream_index,
                steering_spec=spec,
            )
            async for step in stream:
                yield step_body(step)

        items = await open_heavy(bodies())
        if not req.stream:
            try:
                steps = [_untyped(b) async for b in items]
            finally:
                runner.release()
            return LensResponse(steps=steps)
        return ndjson(
            items,
            {"type": "start", "jacobian_layers": state.jacobian_layers},
            lambda n: {"type": "done", "n_steps": n},
        )

    @app.post("/v1/unembed_rows")
    async def unembed_rows(req: UnembedRowsRequest, model: Any = Depends(ready)) -> TensorResponse:
        check_prompt(req.token_ids)
        async with heavy():
            rows = await runner.run(model.unembed_rows(req.token_ids))
        return TensorResponse(tensor=encode_tensor(rows, req.output_dtype))

    @app.post("/v1/decode_residuals", response_model=None)
    async def decode_residuals(
        req: DecodeResidualsRequest, model: Any = Depends(ready)
    ) -> TensorResponse | TopKResponse:
        residuals = decode_tensor(req.residuals)
        if residuals.dim() != 2 or residuals.shape[1] != model.d_model:
            raise ValueError(f"residuals must be [n_rows, {model.d_model}], got {list(residuals.shape)}")
        if residuals.shape[0] > config.max_prompt_tokens:
            raise ValueError(f"{residuals.shape[0]} rows sent; this server accepts {config.max_prompt_tokens}")
        async with heavy():
            logits = await runner.run(model.decode_residuals(residuals))
        if req.top_k is None:
            return TensorResponse(tensor=encode_tensor(logits, req.output_dtype))
        values, indices = torch.topk(logits.float(), min(req.top_k, logits.shape[-1]), dim=-1)
        return TopKResponse(token_ids=indices.tolist(), logits=values.tolist())

    return app


def _line(body: dict[str, Any]) -> bytes:
    return (json.dumps(body) + "\n").encode()


def _untyped(body: dict[str, Any]) -> dict[str, Any]:
    """A stream line as a step of the whole response, which has no line types."""
    return {k: v for k, v in body.items() if k != "type"}
