"""Read, write and run the manifest's fixtures, for the Python engine.

A fixture is one manifest case on one checkpoint, as the reference computed it::

    <root>/<org--name>/meta.json                 checkpoint, dtype, manifest digest, engine commit
    <root>/<org--name>/<op>/<case>.json          inputs and expected result; tensors by reference
    <root>/<org--name>/<op>/<case>.safetensors   those tensors

``<root>`` is ``IE_FIXTURES_DIR`` or ``~/.cache/interp-engine/fixtures``, outside git, as the golden
tensors are. ``api/make_fixtures.py`` writes it from ``EagerModel`` on the CPU in float32; the tests
here read it. A tensor in JSON is ``{"$tensor": key}``.

A case's inputs are written resolved, so neither reader needs a tokenizer: ``{prompt: text}`` is
the ids ``to_tokens`` gives, ``{tokens: [...]}`` the id of each single-token string,
``{capture: {prompt, point}}`` the captured tensor, ``{near_identity: {seed, scale}}`` a
``[d_model, d_model]`` identity plus seeded noise, ``{randn: {shape, seed}}`` seeded normal noise,
``{vocab_mask: {every}}`` a bool mask true at every ``every``-th id, and ``{mid}``, ``{last}``,
``{n_layers}`` and ``{d_model}`` in a string or a key are those numbers (a string that is only a
placeholder becomes the int). A ``steering`` input is a list of ops, ``{point, method, vector}``
and the method's fields; each is one ``SteeringSpec.at(point, op)``.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import Case, Manifest, Op, T  # noqa: E402

DEFAULT_ROOT = Path(os.environ.get("IE_FIXTURES_DIR", "~/.cache/interp-engine/fixtures")).expanduser()


def fixture_dir(hf_id: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / hf_id.replace("/", "--")


def case_path(hf_id: str, op: str, case: str, root: Path = DEFAULT_ROOT) -> Path:
    return fixture_dir(hf_id, root) / op / f"{case}.json"


# ------------------------------------------------------------------------------ resolving --


def _layers(model: Any) -> dict[str, int]:
    n = int(model.n_layers)
    return {"mid": n // 2, "last": n - 1, "n_layers": n, "d_model": int(model.d_model)}


def resolve(value: Any, model: Any, tensors: dict[str, torch.Tensor], key: str) -> Any:
    """One manifest input as the reference will pass it; a made tensor lands in ``tensors``."""
    if isinstance(value, str):
        text = value.format(**_layers(model))
        return int(text) if re.fullmatch(r"\{\w+\}", value) else text
    if isinstance(value, list):
        return [resolve(v, model, tensors, f"{key}.{i}") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        if set(value) == {"prompt"}:
            return [int(t) for t in model.to_tokens(value["prompt"])[0]]
        if set(value) == {"tokens"}:
            ids = []
            for text in value["tokens"]:
                got = model.tokenizer.encode(text, add_special_tokens=False)
                if len(got) != 1:
                    raise ValueError(f"{text!r} is {len(got)} tokens on {model.hf_model_id}; a case needs one")
                ids.append(int(got[0]))
            return ids
        if set(value) == {"capture"}:
            from interp_engine import capture, to_address

            spec = value["capture"]
            ids = [int(t) for t in model.to_tokens(spec["prompt"])[0]]
            address = to_address(spec["point"].format(**_layers(model)))
            got = capture(model, ids, [address]).get(address.name, address.layer)[0]
            tensors[f"inputs/{key}"] = got.detach().float().contiguous()
            return {"$tensor": f"inputs/{key}"}
        if set(value) == {"near_identity"}:
            spec = value["near_identity"]
            d = int(model.d_model)
            gen = torch.Generator().manual_seed(int(spec["seed"]))
            noise = torch.randn(d, d, generator=gen) * (float(spec["scale"]) / d**0.5)
            tensors[f"inputs/{key}"] = (torch.eye(d) + noise).contiguous()
            return {"$tensor": f"inputs/{key}"}
        if set(value) == {"randn"}:
            spec = value["randn"]
            shape = [int(resolve(d, model, tensors, key)) for d in spec["shape"]]
            gen = torch.Generator().manual_seed(int(spec["seed"]))
            tensors[f"inputs/{key}"] = torch.randn(*shape, generator=gen).contiguous()
            return {"$tensor": f"inputs/{key}"}
        if set(value) == {"vocab_mask"}:
            every = int(value["vocab_mask"]["every"])
            tensors[f"inputs/{key}"] = torch.arange(len(model.tokenizer)) % every == 0
            return {"$tensor": f"inputs/{key}"}
        out = {}
        for k, v in value.items():
            name = str(resolve(k, model, tensors, key))
            out[name] = resolve(v, model, tensors, f"{key}.{name}")
        return out
    return value


# -------------------------------------------------------------------------------- calling --


def _decode(value: Any, t: T, tensors: Mapping[str, torch.Tensor], manifest: Manifest) -> Any:
    if value is None:
        return None
    if t.name == "tensor":
        return tensors[value["$tensor"]]
    if t.name == "address":
        from interp_engine import to_address

        return to_address(value)
    if t.name == "steering":
        return [_steering_spec(op, tensors) for op in value]
    if t.name == "list":
        return [_decode(v, t.args[0], tensors, manifest) for v in value]
    if t.name == "map":
        key = int if t.args[0].name == "int" else str
        return {key(k): _decode(v, t.args[1], tensors, manifest) for k, v in value.items()}
    if t.name in manifest.records:
        from interp_engine import api

        record = manifest.records[t.name]
        fields = {f.name: _decode(value[f.name], f.type, tensors, manifest) for f in record.fields if f.name in value}
        return getattr(api, t.name)(**fields)
    return value


def _steering_spec(op: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> Any:
    """One steering op of a case as a spec at its point. Field names are the op class's own."""
    from interp_engine import steer_specs as s

    ops = {
        "additive": s.AddSpec,
        "orthogonal": s.OrthogonalDecompSpec,
        "projection_cap": s.ProjectionCapSpec,
        "norm_scaled_add": s.NormScaledAddSpec,
        "ablate": s.AblateSpec,
        "swap": s.SwapSpec,
    }
    fields = {
        k: tensors[v["$tensor"]] if isinstance(v, Mapping) else v for k, v in op.items() if k not in ("point", "method")
    }
    return s.SteeringSpec.at(op["point"], ops[op["method"]](**fields))


async def call(
    model: Any, op: Op, inputs: Mapping[str, Any], tensors: Mapping[str, torch.Tensor], manifest: Manifest
) -> Any:
    """Call ``op`` on ``model`` with a case's resolved inputs. A parameter the case leaves out takes
    the backend's own default, which ``tests/test_api_generated.py`` holds to the manifest's."""
    if op.kind == "property":
        return getattr(model, op.name)
    args, kwargs = [], {}
    for p in op.params:
        if p.name not in inputs:
            if p.positional and not p.has_default:
                raise ValueError(f"{op.name}: the case gives no {p.name}")
            continue
        value = _decode(inputs[p.name], p.type, tensors, manifest)
        if p.positional:
            args.append(value)
        else:
            kwargs[p.python_name] = value
    fn = getattr(model, op.name)
    if op.kind == "stream":
        return [item async for item in fn(*args, **kwargs)]
    result = fn(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


# --------------------------------------------------------------------------------- trees --


def to_tree(value: Any, t: T, manifest: Manifest) -> Any:
    """A result as nested None/bool/int/float/str/Tensor/list/dict, keyed by field name."""
    if value is None:
        return None
    if t.name == "stream":
        return [to_tree(v, t.args[0], manifest) for v in value]
    if t.name == "tensor":
        return value.detach().to("cpu", torch.float32).contiguous()
    if t.name == "address":
        return str(value)
    if t.name in ("list", "token_ids"):
        inner = t.args[0] if t.args else T("int")
        return [to_tree(v, inner, manifest) for v in value]
    if t.name == "map":
        return {str(k): to_tree(v, t.args[1], manifest) for k, v in value.items()}
    if t.name in manifest.records:
        return {f.name: to_tree(getattr(value, f.name), f.type, manifest) for f in manifest.records[t.name].fields}
    if t.name == "float":
        return float(value)
    if t.name in ("int", "uint64"):
        return int(value)
    return value


def encode(tree: Any, tensors: dict[str, torch.Tensor], key: str = "expected") -> Any:
    """``tree`` as JSON, each tensor moved into ``tensors`` and left as a reference."""
    if isinstance(tree, torch.Tensor):
        tensors[key] = tree
        return {"$tensor": key}
    if isinstance(tree, list):
        return [encode(v, tensors, f"{key}.{i}") for i, v in enumerate(tree)]
    if isinstance(tree, dict):
        return {k: encode(v, tensors, f"{key}.{k}") for k, v in tree.items()}
    return tree


def _agreement(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Max-abs relative error and cosine, in float64."""
    x, y = a.double().flatten(), b.double().flatten()
    rel = float((x - y).abs().max() / max(float(y.abs().max()), 1e-12))
    cos = float(x @ y / max(float(x.norm() * y.norm()), 1e-30))
    return rel, cos


def compare(
    actual: Any,
    expected: Any,
    tensors: Mapping[str, torch.Tensor],
    tolerance: Mapping[str, Any],
    fields: list[str] | None,
    path: str = "$",
) -> list[str]:
    """Every place ``actual`` differs from ``expected``, as ``path: why``; empty when they agree."""
    if isinstance(actual, torch.Tensor):
        if not (isinstance(expected, dict) and "$tensor" in expected):
            return [f"{path}: expected no tensor here"]
        want = tensors[expected["$tensor"]]
        if tuple(actual.shape) != tuple(want.shape):
            return [f"{path}: shape {tuple(actual.shape)} != {tuple(want.shape)}"]
        if tolerance["kind"] == "exact":
            return [] if torch.equal(actual.float(), want.float()) else [f"{path}: tensors differ"]
        rel, cos = _agreement(actual, want)
        ok = rel <= tolerance["rel"] and cos >= tolerance["cos"]
        return (
            []
            if ok
            else [f"{path}: relative {rel:.3g} (<= {tolerance['rel']}), cosine {cos:.8f} (>= {tolerance['cos']})"]
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        return [] if actual is expected else [f"{path}: {actual!r} != {expected!r}"]
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return [] if abs(actual - expected) <= 1e-6 * max(1.0, abs(expected)) else [f"{path}: {actual} != {expected}"]
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: {len(actual)} items != {len(expected)}"]
        out: list[str] = []
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            out += compare(a, e, tensors, tolerance, fields, f"{path}[{i}]")
        return out
    if isinstance(actual, dict) and isinstance(expected, dict):
        keys = set(fields) if fields else set(actual) | set(expected)
        out = []
        for k in sorted(keys):
            if k not in actual or k not in expected:
                out.append(f"{path}.{k}: present on one side only")
            else:
                out += compare(actual[k], expected[k], tensors, tolerance, fields, f"{path}.{k}")
        return out
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


# ------------------------------------------------------------------------------- on disk --


def write_case(
    hf_id: str,
    op: Op,
    case: Case,
    inputs: dict[str, Any],
    input_tensors: dict[str, torch.Tensor],
    expected: Any,
    root: Path = DEFAULT_ROOT,
) -> Path:
    tensors = dict(input_tensors)
    payload = {
        "op": op.name,
        "case": case.name,
        "digest": case.digest,
        "tolerance": case.compare,
        "fields": case.fields,
        "inputs": inputs,
        "expected": encode(expected, tensors),
    }
    path = case_path(hf_id, op.name, case.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    st = path.with_suffix(".safetensors")
    if tensors:
        save_file(tensors, str(st))
    elif st.exists():
        st.unlink()
    return path


def read_case(
    hf_id: str, op: str, case: str, root: Path = DEFAULT_ROOT
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = case_path(hf_id, op, case, root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    st = path.with_suffix(".safetensors")
    return payload, (load_file(str(st)) if st.exists() else {})


async def run_case(
    model: Any, manifest: Manifest, op: Op, case: Case, hf_id: str, root: Path = DEFAULT_ROOT
) -> list[str]:
    """Run one case on ``model`` and compare with the fixture on disk; the problems, or empty."""
    path = case_path(hf_id, op.name, case.name, root)
    if not path.exists():
        return [f"no fixture at {path}; run `uv run python api/make_fixtures.py {hf_id}`"]
    payload, tensors = read_case(hf_id, op.name, case.name, root)
    if payload["digest"] != case.digest:
        return [f"{path} was written for an older case; rerun api/make_fixtures.py {hf_id}"]
    assert op.returns is not None
    actual = to_tree(await call(model, op, payload["inputs"], tensors, manifest), op.returns, manifest)
    return compare(actual, payload["expected"], tensors, manifest.tolerances[case.compare], case.fields)
