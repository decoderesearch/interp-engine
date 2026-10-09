"""Write the manifest's fixtures for one checkpoint, from the Python reference.

The reference is ``EagerModel`` on the CPU in float32.
Every case in ``api/engine.yaml`` is run once and its result written under ``IE_FIXTURES_DIR`` (or
``~/.cache/interp-engine/fixtures``); see ``api/fixtures.py`` for the layout. Run in the engine's
venv, on the machine whose other backends will be checked::

    uv run python api/make_fixtures.py HuggingFaceTB/SmolLM2-135M

Fixtures are only ever regenerated from here. A backend that disagrees is fixed, or the manifest's
tolerance is changed on purpose; the expected values are never edited to make a test pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fixtures  # noqa: E402
from generate import load  # noqa: E402

from interp_engine import EagerModel, __version__  # noqa: E402


def engine_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def write_all(model: EagerModel, hf_id: str, root: Path) -> int:
    manifest = load()
    written = 0
    for op in manifest.ops:
        for case in op.cases:
            tensors: dict = {}
            inputs = {k: fixtures.resolve(v, model, tensors, k) for k, v in case.inputs.items()}
            result = await fixtures.call(model, op, inputs, tensors, manifest)
            assert op.returns is not None
            expected = fixtures.to_tree(result, op.returns, manifest)
            path = fixtures.write_case(hf_id, op, case, inputs, tensors, expected, root)
            print(f"  {path.relative_to(root)}", file=sys.stderr)
            written += 1
    meta = {
        "hf_id": hf_id,
        "dtype": "float32",
        "device": "cpu",
        "manifest_digest": manifest.digest,
        "engine_version": __version__,
        "engine_commit": engine_commit(),
        "n_layers": model.n_layers,
    }
    out = fixtures.fixture_dir(hf_id, root)
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_id")
    parser.add_argument("--out", type=Path, default=fixtures.DEFAULT_ROOT)
    args = parser.parse_args()
    model = EagerModel(args.hf_id, device="cpu", dtype="float32", attn_implementation="eager")
    n = asyncio.run(write_all(model, args.hf_id, args.out))
    print(f"wrote {n} cases for {args.hf_id} under {fixtures.fixture_dir(args.hf_id, args.out)}", file=sys.stderr)


if __name__ == "__main__":
    main()
