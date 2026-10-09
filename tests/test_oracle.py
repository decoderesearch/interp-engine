"""LoRA reads: the read contract, the adapter hooks, and a read on a small Qwen3.5.

The CPU tests pin the pieces a wrong read would hide in: the marker scan, the injected row, the
bullet cap, and the hook-applied LoRA against the merged weights. The GPU test reads with a
synthetic adapter on ``Qwen/Qwen3.5-0.8B`` and checks the batched read against one layer at a time.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from interp_engine.oracle import (
    ALL_LAYERS,
    FAST_LAYERS,
    OracleAdapter,
    OracleContract,
    OraclePartial,
    OraclePrompt,
    OracleRead,
    _vllm_reads,
    bullets_done,
    bullets_of,
    eager_lora,
    find_marker,
    inject,
    oracle_prompts,
    read_oracle,
)

RUN_JSON = {
    "config": {
        "transform": "unit",
        "alpha": 16000.0,
        "layers": list(ALL_LAYERS),
        "extra": {"prompt": "concepts_raw"},
    }
}


def _adapter_dir(tmp: Path, tensors: dict[str, torch.Tensor], *, r: int, alpha: float, layers: list[int]) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tmp / "adapter_model.safetensors"))
    (tmp / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "target_modules": ["down_proj"], "peft_type": "LORA"})
    )
    run = json.loads(json.dumps(RUN_JSON))
    run["config"]["layers"] = layers
    (tmp / "run.json").write_text(json.dumps(run))
    return tmp


def test_the_contract_comes_from_run_json_and_refuses_layers_off_the_band() -> None:
    contract = OracleContract.from_run(RUN_JSON)
    assert contract == OracleContract()
    assert set(FAST_LAYERS) <= set(contract.layers)
    with pytest.raises(ValueError, match="outside this adapter's trained band"):
        contract.check_layers([20, 63])
    with pytest.raises(ValueError, match="only 'unit'"):
        OracleContract(transform="whiten")


def test_the_injected_row_is_alpha_times_the_unit_activation() -> None:
    rows = torch.randn(6, 8, dtype=torch.bfloat16)
    h = torch.randn(8) * 37
    out = inject(rows, 2, h, 16000.0)
    assert torch.equal(out[[0, 1, 3, 4, 5]], rows[[0, 1, 3, 4, 5]])
    torch.testing.assert_close(out[2].float(), (16000.0 * h / h.norm()).bfloat16().float())
    assert torch.equal(rows, rows.clone())
    with pytest.raises(ValueError, match="zero or non-finite"):
        inject(rows, 0, torch.zeros(8), 1.0)


def test_a_bullet_is_done_at_its_newline() -> None:
    assert not bullets_done("- Paris", 1)
    assert bullets_done("- Paris\n", 1)
    assert not bullets_done("- Paris\n- France", 2)
    assert bullets_done("intro\n- Paris\n- France\n- ", 2)
    assert bullets_of("- Paris\n  * France \nnot a bullet\n") == ["Paris", "France"]


def test_the_hooked_adapter_matches_the_merged_weights_and_comes_off(tmp_path: Path) -> None:
    torch.manual_seed(0)
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.down_proj = torch.nn.Linear(16, 12, bias=False)
    model = SimpleNamespace(arch=SimpleNamespace(decoder_layers=[torch.nn.Identity(), layer]))
    a, b = torch.randn(4, 16), torch.randn(12, 4)
    adapter = OracleAdapter.from_directory(
        _adapter_dir(
            tmp_path / "ao",
            {
                "base_model.model.model.layers.1.mlp.down_proj.lora_A.weight": a,
                "base_model.model.model.layers.1.mlp.down_proj.lora_B.weight": b,
            },
            r=4,
            alpha=8,
            layers=[1],
        )
    )
    x = torch.randn(3, 16)
    base = layer.mlp.down_proj(x)
    merged = x @ (layer.mlp.down_proj.weight + adapter.scale * b @ a).T
    with eager_lora(model, adapter):
        torch.testing.assert_close(layer.mlp.down_proj(x), merged)
    torch.testing.assert_close(layer.mlp.down_proj(x), base)


def _base_dir(tmp: Path, trunk: str) -> Path:
    """A base checkpoint that is only its index, naming its layers ``{trunk}N``."""
    tmp.mkdir(parents=True, exist_ok=True)
    names = {f"{trunk}3.mlp.down_proj.weight": "model.safetensors", "lm_head.weight": "model.safetensors"}
    (tmp / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
    return tmp


def test_the_vllm_copy_keys_the_trunk_as_the_checkpoint_does(tmp_path: Path) -> None:
    from safetensors.torch import load_file

    key = "base_model.model.model.layers.3.linear_attn.in_proj_qkv.lora_A.weight"
    adapter = OracleAdapter.from_directory(
        _adapter_dir(tmp_path / "ao", {key: torch.randn(2, 4)}, r=2, alpha=4, layers=[3])
    )
    out = adapter.vllm_dir(str(_base_dir(tmp_path / "vl", "model.language_model.layers.")))
    keys = set(load_file(str(out / "adapter_model.safetensors")))
    assert keys == {"base_model.model.model.language_model.layers.3.linear_attn.in_proj_qkv.lora_A.weight"}
    assert (out / "adapter_config.json").exists()


def test_the_vllm_copy_of_a_text_only_checkpoint_is_the_adapter_as_saved(tmp_path: Path) -> None:
    key = "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight"
    directory = _adapter_dir(tmp_path / "ao", {key: torch.randn(2, 4)}, r=2, alpha=4, layers=[3])
    adapter = OracleAdapter.from_directory(directory)
    assert adapter.vllm_dir(str(_base_dir(tmp_path / "text", "model.layers."))) == directory
    assert not (tmp_path / "ao.vllm").exists()


def test_the_vllm_copy_refuses_a_checkpoint_with_no_known_trunk(tmp_path: Path) -> None:
    key = "base_model.model.model.layers.3.mlp.down_proj.lora_A.weight"
    adapter = OracleAdapter.from_directory(
        _adapter_dir(tmp_path / "ao", {key: torch.randn(2, 4)}, r=2, alpha=4, layers=[3])
    )
    with pytest.raises(ValueError, match="no decoder layers"):
        adapter.vllm_dir(str(_base_dir(tmp_path / "other", "language_model.model.layers.")))


def test_the_inline_prompt_is_the_same_words_on_one_line() -> None:
    inline = OracleContract(prompt_kind="concepts_raw_inline").template
    assert "\n" not in inline
    assert inline == OracleContract().template.replace("\n", " ")


class _FakeVLLM:
    """Writes ``texts[layer]`` one character per step, then EOS; a layer whose text is None fails.

    A prompt's first token id is its layer, and row 0 of its embeds carries it to the generator.
    """

    max_lora_rank = 8

    def __init__(self, texts: dict[int, str | None]) -> None:
        self.texts = texts
        self.tokenizer = SimpleNamespace(eos_token_id=0, unk_token_id=None, convert_tokens_to_ids=lambda _n: None)

    async def embed_rows(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(token_ids, dtype=torch.float32)[:, None].expand(-1, 4).clone()

    async def generate_steps_from_embeds(self, rows: torch.Tensor, **_kwargs: object) -> AsyncIterator[object]:
        layer = int(rows[0, 0])
        text = self.texts[layer]
        if text is None:
            raise RuntimeError(f"layer {layer} failed")
        for char in text:
            await asyncio.sleep(0)
            yield SimpleNamespace(token_id=ord(char), token_str=char)
        yield SimpleNamespace(token_id=0, token_str="")


def _vllm_run(texts: dict[int, str | None], *, partial: bool) -> list[OracleRead | OraclePartial]:
    adapter = SimpleNamespace(rank=2, vllm_dir=lambda: Path("/unused"), contract=SimpleNamespace(alpha=1.0))
    prompts = {layer: OraclePrompt(layer, (layer, 5), 1, "x", 5) for layer in texts}
    acts = {layer: torch.ones(4) for layer in texts}

    async def run() -> list[OracleRead | OraclePartial]:
        return [item async for item in _vllm_reads(_FakeVLLM(texts), adapter, prompts, acts, 2, 64, partial)]

    return asyncio.run(run())


def test_a_vllm_read_streams_each_layers_text_before_its_read() -> None:
    texts: dict[int, str | None] = {20: "- Paris\n- France\n- Europe\n", 24: "- city\n"}
    items = _vllm_run(texts, partial=True)
    reads = {i.layer: i for i in items if isinstance(i, OracleRead)}
    assert reads[20].bullets == ["Paris", "France"] and reads[20].finish == "bullets"
    assert reads[24].bullets == ["city"] and reads[24].finish == "eos"
    for layer, read in reads.items():
        seen = [i.text for i in items if isinstance(i, OraclePartial) and i.layer == layer]
        assert seen and all(read.text.startswith(t) for t in seen)
        assert items.index(read) > max(
            i for i, x in enumerate(items) if isinstance(x, OraclePartial) and x.layer == layer
        )

    assert not any(isinstance(i, OraclePartial) for i in _vllm_run(texts, partial=False))
    with pytest.raises(RuntimeError, match="layer 24 failed"):
        _vllm_run({20: "- Paris\n", 24: None}, partial=True)


@pytest.mark.hub
def test_the_marker_scan_lands_on_the_model_cards_character() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    contract = OracleContract()
    marker = find_marker(tok, contract)
    prompts = oracle_prompts(tok, contract, ALL_LAYERS, marker)
    assert (marker, {p.char_id for p in prompts.values()}) == ("㈜", {158983})
    assert len({len(p.token_ids) for p in prompts.values()}) == 1


def _synthetic_adapter(model, directory: Path, layers: list[int]) -> OracleAdapter:
    """A rank-8 adapter on every MLP of ``model``, large enough to change what it writes."""
    g = torch.Generator().manual_seed(0)
    d, f = model.d_model, model.hf_model.config.get_text_config().intermediate_size
    tensors = {}
    for i in range(model.n_layers):
        pre = f"base_model.model.model.layers.{i}.mlp"
        tensors[f"{pre}.down_proj.lora_A.weight"] = torch.randn(8, f, generator=g) / f**0.5
        tensors[f"{pre}.down_proj.lora_B.weight"] = torch.randn(d, 8, generator=g) * 0.02
    return OracleAdapter.from_directory(_adapter_dir(directory, tensors, r=8, alpha=16, layers=layers))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_batched_eager_read_matches_one_layer_at_a_time(tmp_path: Path) -> None:
    from interp_engine import load_model

    model = load_model("Qwen/Qwen3.5-0.8B", backend="eager", dtype="float32")
    layers = [4, 12, 20]
    adapter = _synthetic_adapter(model, tmp_path / "ao", layers)
    ids = model.tokenizer("The capital of France is Paris.", add_special_tokens=False).input_ids
    resid = asyncio.run(model.capture(ids, [("resid_post", layer) for layer in layers]))
    acts = {addr.layer: rows[-1] for addr, rows in resid.items()}

    batched = asyncio.run(read_oracle(model, adapter, acts, max_bullets=2, max_tokens=48))
    assert list(batched) == layers
    for layer in layers:
        alone = asyncio.run(read_oracle(model, adapter, {layer: acts[layer]}, max_bullets=2, max_tokens=48))[layer]
        assert alone.text == batched[layer].text, layer
        assert batched[layer].token_ids
        assert len(batched[layer].bullets) <= 2
