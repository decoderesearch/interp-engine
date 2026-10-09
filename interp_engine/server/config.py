"""What the server loads, and the limits it applies to each request."""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any

#: Backends that hold one forward at a time. Steering on these installs hooks on the shared
#: modules, so two concurrent requests would steer each other.
SERIAL_BACKENDS = ("eager",)


@dataclass(frozen=True)
class ServerConfig:
    """One model, one process. ``load_model`` receives the load fields as given."""

    hf_model_id: str
    backend: str = "auto"
    device: str | None = None
    dtype: str = "auto"
    quantization: str = ""
    kv_cache_dtype: str = "auto"
    num_gpus: int = 1
    load_kwargs: dict[str, Any] = field(default_factory=dict)

    token: str | None = None
    """The bearer token each request must send. None turns auth off."""

    max_prompt_tokens: int = 8192
    max_new_tokens: int = 2048
    max_capture_points: int = 256
    max_directions: int = 65536
    """Direction rows across every set of one project request."""
    max_concurrency: int | None = None
    """Heavy requests (capture, generate, lens) that run at once. None: 1 on a serial backend, 16
    on vLLM."""
    max_queue: int = 64
    """Heavy requests that wait for a slot. More than this gets 503."""
    warmup: bool = True

    lens_jacobians: str | None = None
    """A Jacobian lens file, a local path or ``hf://<owner>/<repo>/<path>``, loaded after the model."""
    lens_jacobian_dtype: str = "bfloat16"

    def concurrency_for(self, backend: str) -> int:
        """The slot count for the backend that actually loaded."""
        if self.max_concurrency is not None:
            return max(1, self.max_concurrency)
        return 1 if backend in SERIAL_BACKENDS else 16

    def load_args(self) -> dict[str, Any]:
        """Keyword arguments for :func:`interp_engine.load_model`."""
        args: dict[str, Any] = dict(self.load_kwargs)
        args.update(
            backend=self.backend,
            dtype=self.dtype,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
            num_gpus=self.num_gpus,
        )
        if self.device is not None:
            args["device"] = self.device
        return args


def is_loopback(host: str) -> bool:
    """Whether ``host`` only accepts connections from this machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
