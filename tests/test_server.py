"""The HTTP server against the in-process engine: each endpoint returns what the engine returns.

Runs on CPU against cached gpt2, through FastAPI's TestClient, so the lifespan (load on the
engine thread, then serve) runs as it does under uvicorn.
"""

from __future__ import annotations

import asyncio
import json
import time
import typing

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from harness import GPT2, QWEN_THINKING, load_model  # noqa: E402

from interp_engine import SteeringSpec, capture, steer  # noqa: E402
from interp_engine.api import DirectionSet, LensSpec  # noqa: E402
from interp_engine.server import ServerConfig, create_app  # noqa: E402
from interp_engine.server.__main__ import TOKEN_ENV, main  # noqa: E402
from interp_engine.server.lens import is_word_like  # noqa: E402
from interp_engine.server.runner import Busy, EngineRunner  # noqa: E402
from interp_engine.server.schemas import SteeringOpWire, decode_tensor, encode_tensor  # noqa: E402
from interp_engine.steer_specs import AddSpec, LayerSteeringSpec, SteerMethod  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PROMPT = "The capital of France is"


def _ready(client: TestClient) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        body = client.get("/health").json()
        if body["status"] == "ready":
            return
        assert body["status"] != "failed", body
        time.sleep(0.05)
    raise AssertionError("the server did not become ready")


def _client(model, **config) -> typing.Iterator[TestClient]:
    app = create_app(ServerConfig(hf_model_id=model.hf_model_id, token=TOKEN, **config), loader=lambda: model)
    with TestClient(app) as client:
        _ready(client)
        yield client


@pytest.fixture(scope="module")
def model():
    return load_model(GPT2, device="cpu")


@pytest.fixture(scope="module")
def client(model):
    yield from _client(model, max_prompt_tokens=64, max_new_tokens=16, max_directions=16)


@pytest.fixture(scope="module")
def ids(model) -> list[int]:
    return [int(t) for t in model.to_tokens(PROMPT)[0].tolist()]


def _decode(payload: dict) -> torch.Tensor:
    from interp_engine.server.schemas import TensorPayload

    return decode_tensor(TensorPayload(**payload))


def test_health_needs_no_token_and_describe_does(client, model):
    assert client.get("/health").json()["status"] == "ready"
    assert client.get("/v1/describe").status_code == 401
    assert client.get("/v1/describe", headers={"Authorization": "Bearer wrong"}).status_code == 401
    body = client.get("/v1/describe", headers=AUTH).json()
    desc = body["description"]
    assert desc["hf_model_id"] == model.hf_model_id
    assert desc["n_layers"] == model.n_layers and desc["d_model"] == model.d_model
    assert body["load"]["backend"] == "eager"
    assert body["limits"]["max_concurrency"] == 1
    assert body["manifest"]["digest"]


def test_tokenize_raw_text_matches_to_tokens(client, ids, model):
    body = client.post("/v1/tokenize", headers=AUTH, json={"text": PROMPT}).json()
    assert body["token_ids"] == ids
    assert body["str_tokens"] == model.to_str_tokens(torch.tensor(ids))


def test_tokenize_messages_on_a_model_without_a_chat_template_is_refused(client):
    r = client.post("/v1/tokenize", headers=AUTH, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422
    assert "chat template" in r.json()["detail"]


def test_tokenize_refuses_both_inputs(client):
    r = client.post("/v1/tokenize", headers=AUTH, json={"text": "a", "messages": []})
    assert r.status_code == 422


def test_serves_reports_the_engine_refusal(client, model):
    points = ["resid_post.3", f"resid_post.{model.n_layers}", "not a point!"]
    refusals = client.post("/v1/serves", headers=AUTH, json={"points": points}).json()["refusals"]
    assert refusals["resid_post.3"] is None
    assert refusals[f"resid_post.{model.n_layers}"] == model.refuses(f"resid_post.{model.n_layers}")
    assert refusals["not a point!"]


def test_capture_matches_the_in_process_capture(client, model, ids):
    points = ["resid_pre.0", "resid_post.5", "mlp_out.5"]
    body = client.post("/v1/capture", headers=AUTH, json={"prompt_token_ids": ids, "points": points}).json()
    expected = capture(model, torch.tensor([ids]), points)
    for point in points:
        assert torch.equal(_decode(body["activations"][point]), expected[point][0].float())


def test_capture_rows_picks_positions(client, ids):
    full = client.post("/v1/capture", headers=AUTH, json={"prompt_token_ids": ids, "points": ["resid_post.2"]})
    picked = client.post(
        "/v1/capture", headers=AUTH, json={"prompt_token_ids": ids, "points": ["resid_post.2"], "rows": [3, 0]}
    )
    whole = _decode(full.json()["activations"]["resid_post.2"])
    assert torch.equal(_decode(picked.json()["activations"]["resid_post.2"]), whole[[3, 0]])


def test_steered_capture_matches_an_in_process_steer(client, model, ids):
    vector = torch.randn(model.d_model, generator=torch.Generator().manual_seed(0))
    wire = [{"layers": {"3": [{"method": "additive", "vector": vector.tolist(), "scale": 4.0}]}}]
    body = client.post(
        "/v1/capture",
        headers=AUTH,
        json={"prompt_token_ids": ids, "points": ["resid_post.6"], "steering": wire},
    ).json()
    spec = SteeringSpec(layers={3: LayerSteeringSpec(operations=[AddSpec(vector=vector, scale=4.0)])})
    with steer(model, spec):
        expected = capture(model, torch.tensor([ids]), ["resid_post.6"])["resid_post.6"][0]
    assert torch.allclose(_decode(body["activations"]["resid_post.6"]), expected.float(), atol=1e-5)


def test_steering_at_two_points_with_normalize_matches_an_in_process_steer(client, model, ids):
    gen = torch.Generator().manual_seed(2)
    a, b = torch.randn(model.d_model, generator=gen), torch.randn(model.d_model, generator=gen)
    wire = [
        {"layers": {"2": [{"method": "additive", "vector": a.tolist(), "scale": 3.0}]}},
        {
            "point": "mlp_out",
            "layers": {"4": [{"method": "additive", "vector": b.tolist(), "scale": 2.0, "normalize": True}]},
        },
    ]
    body = client.post(
        "/v1/capture",
        headers=AUTH,
        json={"prompt_token_ids": ids, "points": ["resid_post.6"], "steering": wire},
    ).json()
    specs = [
        SteeringSpec.at("resid_post.2", AddSpec(vector=a, scale=3.0)),
        SteeringSpec.at("mlp_out.4", AddSpec(b, 2.0, normalize=True)),
    ]
    with steer(model, specs):
        expected = capture(model, torch.tensor([ids]), ["resid_post.6"])["resid_post.6"][0]
    assert torch.allclose(_decode(body["activations"]["resid_post.6"]), expected.float(), atol=1e-5)


def test_generate_stream_and_whole_agree_with_the_engine(client, model, ids):
    request = {"prompt_token_ids": ids, "max_tokens": 6, "temperature": 0.0}
    with client.stream("POST", "/v1/generate", headers=AUTH, json=request) as r:
        assert r.status_code == 200
        lines = [json.loads(line) for line in r.iter_lines() if line]
    assert lines[0]["type"] == "start" and lines[0]["sampling"]["temperature"] == 0.0
    assert lines[-1] == {"type": "done", "finish": "length", "n_tokens": 6}
    streamed = [line["token_id"] for line in lines if line["type"] == "token"]

    whole = client.post("/v1/generate", headers=AUTH, json={**request, "stream": False}).json()
    assert whole["token_ids"] == streamed and whole["finish"] == "length"

    async def engine() -> list[int]:
        return [s.token_id async for s in model.generate_steps(ids, max_tokens=6, temperature=0.0)]

    assert streamed == asyncio.run(engine())


def test_generate_logprobs_are_reported(client, ids):
    request = {"prompt_token_ids": ids, "max_tokens": 2, "temperature": 0.0, "n_logprobs": 3, "stream": False}
    steps = client.post("/v1/generate", headers=AUTH, json=request).json()["steps"]
    assert all(len(s["logprobs"]) == 3 for s in steps)
    assert all(s["token_id"] == s["logprobs"][0]["token_id"] for s in steps)


def test_steered_generation_differs_from_unsteered(client, model, ids):
    vector = torch.randn(model.d_model, generator=torch.Generator().manual_seed(1))
    base = {"prompt_token_ids": ids, "max_tokens": 6, "temperature": 0.0, "stream": False}
    steering = [{"layers": {"4": [{"method": "additive", "vector": vector.tolist(), "scale": 80.0}]}}]
    plain = client.post("/v1/generate", headers=AUTH, json=base).json()["token_ids"]
    steered = client.post("/v1/generate", headers=AUTH, json={**base, "steering": steering}).json()["token_ids"]
    assert plain != steered


def test_decode_residuals_top_k_matches_the_engine(client, model, ids):
    residuals = capture(model, torch.tensor([ids]), ["resid_post.8"])["resid_post.8"][0]
    body = client.post(
        "/v1/decode_residuals",
        headers=AUTH,
        json={"residuals": encode_tensor(residuals).model_dump(), "top_k": 5},
    ).json()
    logits = asyncio.run(model.decode_residuals(residuals))
    assert body["token_ids"] == torch.topk(logits, 5, dim=-1).indices.tolist()


def test_unembed_rows_matches_the_engine(client, model):
    body = client.post("/v1/unembed_rows", headers=AUTH, json={"token_ids": [1, 2, 3]}).json()
    expected = asyncio.run(model.unembed_rows([1, 2, 3]))
    assert torch.equal(_decode(body["tensor"]), expected.float())


@pytest.mark.parametrize(
    ("path", "body", "fragment"),
    [
        ("/v1/capture", {"prompt_token_ids": [10**9], "points": ["resid_post.0"]}, "outside this model's vocabulary"),
        ("/v1/capture", {"prompt_token_ids": [1] * 65, "points": ["resid_post.0"]}, "accepts 64"),
        ("/v1/capture", {"prompt_token_ids": [1, 2], "points": ["resid_post:3"]}, ""),
        (
            "/v1/capture",
            {
                "prompt_token_ids": [1],
                "points": ["resid_post.0"],
                "steering": [{"layers": {"0": [{"method": "ablate", "vector": [1.0]}]}}],
            },
            "d_model",
        ),
        ("/v1/generate", {"prompt_token_ids": [1], "max_tokens": 17}, "accepts 16"),
        ("/v1/decode_residuals", {"residuals": [[0.0, 1.0]]}, "residuals must be"),
        (
            "/v1/project",
            {"prompt_token_ids": [1, 2], "directions": [{"point": "resid_post.0", "vectors": [[0.0] * 768] * 17}]},
            "accepts 16",
        ),
        (
            "/v1/project",
            {"prompt_token_ids": [1, 2], "directions": [{"point": "resid_post.0", "vectors": [[0.0, 1.0]]}]},
            "basis of its point",
        ),
        ("/v1/lens", {"prompt_token_ids": [1, 2], "lenses": [{"layers": [3, 2]}]}, "must ascend"),
        ("/v1/lens", {"prompt_token_ids": [1, 2], "lenses": [{"layers": [2], "jacobian": True}]}, "J_bar"),
        ("/v1/lens", {"prompt_token_ids": [1, 2], "lenses": [{"layers": [2]}], "max_tokens": 17}, "accepts 16"),
    ],
)
def test_bad_requests_are_refused_with_a_reason(client, path, body, fragment):
    r = client.post(path, headers=AUTH, json=body)
    assert r.status_code == 422, r.text
    assert fragment in json.dumps(r.json())


def test_project_matches_the_engine(client, model, ids):
    gen = torch.Generator().manual_seed(3)
    probe = torch.randn(1, model.d_model, generator=gen)
    encoder, bias = torch.randn(8, model.d_model, generator=gen), torch.randn(8, generator=gen)
    wire = [
        {"point": "resid_post.4", "vectors": probe.tolist()},
        {
            "point": "resid_post.9",
            "vectors": encode_tensor(encoder).model_dump(),
            "bias": bias.tolist(),
            "nonlinearity": "relu",
        },
    ]
    body = client.post("/v1/project", headers=AUTH, json={"prompt_token_ids": ids, "directions": wire}).json()
    sets = [
        DirectionSet(point="resid_post.4", vectors=probe),
        DirectionSet(point="resid_post.9", vectors=encoder, bias=bias, nonlinearity="relu"),
    ]
    expected = asyncio.run(model.project(ids, sets))
    got = [_decode(v) for v in body["values"]]
    assert [tuple(g.shape) for g in got] == [(len(ids), 1), (len(ids), 8)]
    for g, e in zip(got, expected, strict=True):
        assert torch.equal(g, e)


def _lens_lines(client, request: dict) -> list[dict]:
    with client.stream("POST", "/v1/lens", headers=AUTH, json=request) as r:
        assert r.status_code == 200, r.read()
        return [json.loads(line) for line in r.iter_lines() if line]


def _engine_lens(model, ids, lenses: list[LensSpec], **kwargs) -> list:
    async def run() -> list:
        return [s async for s in model.generate_with_lens(ids, lenses, **kwargs)]

    return asyncio.run(run())


def test_lens_stream_matches_the_engine(client, model, ids):
    request = {"prompt_token_ids": ids, "lenses": [{"layers": [0, 6, 11]}], "top_n": 5, "max_tokens": 3}
    lines = _lens_lines(client, request)
    assert lines[0] == {"type": "start", "jacobian_layers": []}
    steps = [line for line in lines if line["type"] == "step"]
    assert lines[-1] == {"type": "done", "n_steps": len(steps)}
    expected = _engine_lens(model, ids, [LensSpec(layers=[0, 6, 11])], top_n=5, max_tokens=3)
    assert [s["position"] for s in steps] == [e.position for e in expected]
    assert [s["is_generated"] for s in steps] == [e.is_generated for e in expected]
    for s, e in zip(steps, expected, strict=True):
        read = s["lenses"][0]
        assert read["top_ids"] == e.top_ids[0].tolist()
        assert torch.allclose(torch.tensor(read["top_probs"]), e.top_probs[0])
        assert read["top_strs"] == [model.to_str_tokens(torch.tensor(row)) for row in read["top_ids"]]
    whole = client.post("/v1/lens", headers=AUTH, json={**request, "stream": False}).json()["steps"]
    assert whole == [{k: v for k, v in s.items() if k != "type"} for s in steps]


def test_lens_words_only_ranks_words_but_keeps_the_final_top1(client, ids):
    request = {"prompt_token_ids": ids, "lenses": [{"layers": [2, 11]}], "top_n": 5, "stream": False}
    plain = client.post("/v1/lens", headers=AUTH, json=request).json()["steps"]
    words = client.post("/v1/lens", headers=AUTH, json={**request, "words_only": True}).json()["steps"]
    for p, w in zip(plain, words, strict=True):
        mid_strs, final_strs = w["lenses"][0]["top_strs"]
        assert all(is_word_like(t) for t in mid_strs)
        assert w["lenses"][0]["top_ids"][1][0] == p["lenses"][0]["top_ids"][1][0]
        assert all(is_word_like(t) for t in final_strs[1:])


def test_lens_jacobians_load_at_startup_and_match_the_engine(model, ids, tmp_path):
    gen = torch.Generator().manual_seed(4)
    eye = torch.eye(model.d_model)
    jacobians = {layer: eye + 0.05 * torch.randn(model.d_model, model.d_model, generator=gen) for layer in (1, 6)}
    path = tmp_path / "gpt2_jacobian_lens.pt"
    torch.save({"J": jacobians, "d_model": model.d_model}, path)
    try:
        for c in _client(model, lens_jacobians=str(path), lens_jacobian_dtype="float32"):
            assert c.get("/v1/describe", headers=AUTH).json()["lens"]["jacobian_layers"] == [1, 6]
            request = {"prompt_token_ids": ids, "lenses": [{"layers": [1, 6, 11], "jacobian": True}], "top_n": 4}
            steps = [line for line in _lens_lines(c, request) if line["type"] == "step"]
        lenses = [LensSpec(layers=[1, 6, 11], jacobian=True)]
        expected = _engine_lens(model, ids, lenses, top_n=4, jacobians=jacobians)
        plain = _engine_lens(model, ids, [LensSpec(layers=[1, 6, 11])], top_n=4)
        assert [s["lenses"][0]["top_ids"] for s in steps] == [e.top_ids[0].tolist() for e in expected]
        assert [s["lenses"][0]["top_ids"] for s in steps] != [p.top_ids[0].tolist() for p in plain]
    finally:
        asyncio.run(model.set_lens_jacobians(None))


def test_a_jpp_lens_file_loads_like_a_jacobian_lens_file(tmp_path):
    from interp_engine.server.lens import load_jacobians

    jacobians = {0: torch.eye(4), 2: 2 * torch.eye(4)}
    np_path, jpp_path = tmp_path / "np.pt", tmp_path / "jpp.pt"
    torch.save({"J": jacobians, "d_model": 4}, np_path)
    torch.save({"parameters": {"jacobians": jacobians}, "source_layers": [0, 2], "config": {"d_model": 4}}, jpp_path)
    kw = {"n_layers": 3, "d_model": 4, "dtype": torch.float32}
    a, b = load_jacobians(str(np_path), **kw), load_jacobians(str(jpp_path), **kw)
    assert a.keys() == b.keys() == {0, 2}
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_lens_for_another_model_fails_the_load(model, tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"J": {0: torch.eye(4)}}, path)
    app = create_app(ServerConfig(hf_model_id=model.hf_model_id, lens_jacobians=str(path)), loader=lambda: model)
    with TestClient(app) as c:
        deadline = time.monotonic() + 60
        while (body := c.get("/health").json())["status"] == "loading" and time.monotonic() < deadline:
            time.sleep(0.05)
    assert body["status"] == "failed" and "J_bar" in body["error"]


def test_chat_spans_match_message_spans():
    chat = load_model(QWEN_THINKING, device="cpu")
    messages = [{"role": "user", "content": "Name a colour."}, {"role": "assistant", "content": "Blue."}]
    for c in _client(chat):
        body = c.post("/v1/tokenize", headers=AUTH, json={"messages": messages, "spans": True}).json()
        expected_ids = chat.tok.apply_chat_template(messages, tokenize=True)
        assert body["token_ids"] == list(expected_ids)
        spans = chat.tok.message_spans(messages)
        assert [s["role"] for s in body["spans"]] == [s.role for s in spans]
        assert len(body["spans"]) == len(body["token_ids"])


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_tensor_wire_round_trips(dtype):
    t = torch.randn(3, 4, generator=torch.Generator().manual_seed(2))
    back = decode_tensor(encode_tensor(t, dtype))
    assert back.shape == t.shape
    assert torch.equal(back, t.to(getattr(torch, dtype)).float())


def test_every_steer_method_has_a_wire_form():
    members = typing.get_args(typing.get_args(SteeringOpWire)[0])
    wire = {typing.get_args(m.model_fields["method"].annotation)[0] for m in members}
    assert wire == {m.value for m in SteerMethod}


def test_runner_refuses_past_the_queue():
    async def run() -> None:
        runner = EngineRunner()
        runner.set_limits(concurrency=1, max_queue=0)
        await runner.acquire()
        with pytest.raises(Busy):
            await runner.acquire()
        runner.release()
        await runner.acquire()
        runner.release()
        runner.close()

    asyncio.run(run())


def test_cli_refuses_an_open_host_without_a_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match="no token is set"):
        main(["--model", "openai-community/gpt2", "--host", "0.0.0.0"])
