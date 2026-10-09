"""Request and response bodies, and the conversions between them and engine objects.

A tensor crosses the wire as :class:`TensorPayload`: little-endian bytes in base64, with dtype
and shape. An input tensor can also be a plain nested list. Steering is a list of specs, one per
point; each names its operations by their :class:`~interp_engine.steer_specs.SteerMethod` value.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from interp_engine.address import to_address
from interp_engine.api import DirectionSet
from interp_engine.steer_specs import (
    AblateSpec,
    AddSpec,
    LayerSteeringSpec,
    NormScaledAddSpec,
    OrthogonalDecompSpec,
    ProjectionCapSpec,
    SteeringOp,
    SteeringSpec,
    SwapSpec,
)

WireDtype = Literal["float32", "float16", "bfloat16"]

_NUMPY = {"float32": np.float32, "float16": np.float16}


class TensorPayload(BaseModel):
    """A dense tensor. ``data`` is base64 of the row-major, little-endian bytes."""

    dtype: WireDtype
    shape: list[int]
    data: str


def encode_tensor(tensor: torch.Tensor, dtype: WireDtype = "float32") -> TensorPayload:
    """``tensor`` as a payload in ``dtype``. bfloat16 is sent as its raw 16-bit words."""
    t = tensor.detach().to("cpu").contiguous()
    if dtype == "bfloat16":
        raw = t.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()
    else:
        raw = t.to(getattr(torch, dtype)).numpy().astype(np.dtype(_NUMPY[dtype]).newbyteorder("<")).tobytes()
    return TensorPayload(dtype=dtype, shape=list(t.shape), data=base64.b64encode(raw).decode("ascii"))


def decode_tensor(value: TensorPayload | list[Any]) -> torch.Tensor:
    """A float32 tensor from a payload or a nested list."""
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float32)
    raw = base64.b64decode(value.data)
    if value.dtype == "bfloat16":
        words = np.frombuffer(raw, dtype="<i2").copy()
        t = torch.from_numpy(words).view(torch.bfloat16)
    else:
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.dtype(_NUMPY[value.dtype]).newbyteorder("<")).copy())
    expected = int(np.prod(value.shape)) if value.shape else 1
    if t.numel() != expected:
        raise ValueError(f"tensor data holds {t.numel()} values, but shape {value.shape} needs {expected}")
    return t.reshape(value.shape).float()


Vector = TensorPayload | list[float]


class _Op(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vector: Vector


class AdditiveOp(_Op):
    method: Literal["additive"]
    scale: float
    normalize: bool = False


class OrthogonalOp(_Op):
    method: Literal["orthogonal"]
    coeff: float = 1.0


class ProjectionCapOp(_Op):
    method: Literal["projection_cap"]
    min: float | None = None
    max: float | None = None


class NormScaledAddOp(_Op):
    method: Literal["norm_scaled_add"]
    strength: float
    max_fraction: float = 1.0
    normalize: bool = False


class AblateOp(_Op):
    method: Literal["ablate"]


class SwapOp(_Op):
    method: Literal["swap"]
    target: Vector


SteeringOpWire = Annotated[
    AdditiveOp | OrthogonalOp | ProjectionCapOp | NormScaledAddOp | AblateOp | SwapOp,
    Field(discriminator="method"),
]


class SteeringWire(BaseModel):
    """Steering ops by decoder layer, all written at one point. A request takes a list, in order."""

    model_config = ConfigDict(extra="forbid")
    point: str = "resid_post"
    stream: int | None = None
    layers: dict[int, list[SteeringOpWire]]

    def to_spec(self, d_model: int) -> SteeringSpec:
        """The engine spec. Refuses a vector whose width is not ``d_model``."""
        layers: dict[int, LayerSteeringSpec] = {}
        for layer, ops in self.layers.items():
            layers[layer] = LayerSteeringSpec(operations=[_to_op(op, layer, d_model) for op in ops])
        return SteeringSpec(layers=layers, point=self.point, stream=self.stream)


def _vector(value: Vector, what: str, d_model: int) -> torch.Tensor:
    v = decode_tensor(value).flatten()
    if v.numel() != d_model:
        raise ValueError(f"{what} has {v.numel()} values; this model's d_model is {d_model}")
    return v


def _to_op(op: Any, layer: int, d_model: int) -> SteeringOp:
    what = f"layer {layer} {op.method} vector"
    v = _vector(op.vector, what, d_model)
    if isinstance(op, AdditiveOp):
        return AddSpec(vector=v, scale=op.scale, normalize=op.normalize)
    if isinstance(op, OrthogonalOp):
        return OrthogonalDecompSpec(vector=v, coeff=op.coeff)
    if isinstance(op, ProjectionCapOp):
        return ProjectionCapSpec(vector=v, min=op.min, max=op.max)
    if isinstance(op, NormScaledAddOp):
        return NormScaledAddSpec(vector=v, strength=op.strength, max_fraction=op.max_fraction, normalize=op.normalize)
    if isinstance(op, AblateOp):
        return AblateSpec(vector=v)
    if isinstance(op, SwapOp):
        return SwapSpec(vector=v, target=_vector(op.target, f"layer {layer} swap target", d_model))
    raise ValueError(f"unknown steering method {op.method!r}")


class ChatMessage(BaseModel):
    """One chat turn. Extra fields go to the chat template as given."""

    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class TokenizeRequest(BaseModel):
    text: str | None = None
    messages: list[ChatMessage] | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False
    prepend_bos: bool | None = None
    """Raw text only. None uses the model's default."""
    template_kwargs: dict[str, Any] = Field(default_factory=dict)
    spans: bool = False
    """Chat only: add per-token role, section and message index."""

    @model_validator(mode="after")
    def _one_input(self) -> TokenizeRequest:
        if (self.text is None) == (self.messages is None):
            raise ValueError("send exactly one of text or messages")
        if self.spans and self.messages is None:
            raise ValueError("spans needs messages; raw text has no turns")
        return self


class TokenizeResponse(BaseModel):
    token_ids: list[int]
    str_tokens: list[str]
    spans: list[dict[str, Any]] | None = None


class ServesRequest(BaseModel):
    points: list[str] = Field(min_length=1)


class ServesResponse(BaseModel):
    refusals: dict[str, str | None]
    """Point to the reason it cannot be served, or null when it can."""


class CaptureRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1)
    points: list[str] = Field(min_length=1)
    rows: list[int] | None = None
    steering: list[SteeringWire] | None = None
    output_dtype: WireDtype = "float32"


class CaptureResponse(BaseModel):
    activations: dict[str, TensorPayload]
    """Canonical address to ``[n_rows, width]``."""


class GenerateRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1)
    max_tokens: int = Field(default=64, ge=1)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop_at_eos: bool = True
    n_logprobs: int = Field(default=0, ge=0, le=20)
    steering: list[SteeringWire] | None = None
    steer_generated: bool = True
    """False steers the prompt only."""
    stream: bool = True


class GenerateResponse(BaseModel):
    """The whole generation, for ``stream=false``."""

    text: str
    token_ids: list[int]
    steps: list[dict[str, Any]]
    sampling: dict[str, Any]
    finish: Literal["eos", "length"]


class UnembedRowsRequest(BaseModel):
    token_ids: list[int] = Field(min_length=1)
    output_dtype: WireDtype = "float32"


class DecodeResidualsRequest(BaseModel):
    residuals: TensorPayload | list[list[float]]
    """``[n_rows, d_model]``."""
    top_k: int | None = Field(default=None, ge=1)
    """Return only the top ``k`` logits per row, with their token ids."""
    output_dtype: WireDtype = "float32"


class TensorResponse(BaseModel):
    tensor: TensorPayload


class TopKResponse(BaseModel):
    token_ids: list[list[int]]
    logits: list[list[float]]


class DirectionSetWire(BaseModel):
    """Directions read at one point: a probe is one row; an SAE encoder is many, with a bias and relu."""

    model_config = ConfigDict(extra="forbid")
    point: str
    vectors: TensorPayload | list[list[float]]
    """``[k, width]``, in the basis of the point's rows."""
    bias: TensorPayload | list[float] | None = None
    nonlinearity: Literal["none", "relu"] = "none"

    def to_engine(self) -> DirectionSet:
        bias = None if self.bias is None else decode_tensor(self.bias)
        return DirectionSet(
            point=to_address(self.point),
            vectors=decode_tensor(self.vectors),
            bias=bias,
            nonlinearity=self.nonlinearity,
        )


class ProjectRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1)
    directions: list[DirectionSetWire] = Field(min_length=1)
    steering: list[SteeringWire] | None = None
    output_dtype: WireDtype = "float32"


class ProjectResponse(BaseModel):
    values: list[TensorPayload]
    """Per direction set, in order: ``[n_prompt_tokens, k]``."""


class LensSpecWire(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layers: list[int] = Field(min_length=1)
    """Ascending. The last is decoded as the final row of each position."""
    jacobian: bool = False
    """Carry each layer through the server's J_bar first. A layer with no J_bar is read as it is."""


class LensRequest(BaseModel):
    prompt_token_ids: list[int] = Field(min_length=1)
    lenses: list[LensSpecWire] = Field(min_length=1, max_length=8)
    point: str = "resid_post"
    top_n: int = Field(default=10, ge=1, le=100)
    max_tokens: int = Field(default=0, ge=0)
    """Tokens to generate after the prompt, each read as it lands."""
    temperature: float = 0.0
    seed: int | None = None
    words_only: bool = False
    """Rank word-like tokens only. The last layer keeps its true top-1."""
    skip_before: int = Field(default=0, ge=0)
    """No steps for positions before this one."""
    stream_reduce: str = "none"
    stream_index: int | None = None
    steering: list[SteeringWire] | None = None
    stream: bool = True


class LensResponse(BaseModel):
    """The whole read-out, for ``stream=false``: one step per position."""

    steps: list[dict[str, Any]]
