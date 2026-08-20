"""Chat formats that live in Python instead of in a Jinja ``chat_template``.

Almost every chat model describes its own prompt format in the tokenizer's ``chat_template``,
which is why :meth:`Tokenize.message_spans` needs no per-model knowledge: it renders the real
template and diffs. A few checkpoints ship no template at all and define the format in code
instead, because the thing they need to express is not a render. DeepSeek-V4 is the case this
module exists for: its ``encoding/`` folder carries an encoder *and* a parser
(``parse_message_from_completion_text``), and a template can only ever do the first half.

**The reference implementation ships inside the checkpoint**, at
``encoding/encoding_dsv4.py``, beside the weights and versioned with them. So this module
downloads and imports it rather than carrying a copy: a vendored fork is a second source of
truth for a format whose first source of truth we already have on disk. That the risk is real
rather than theoretical is easy to check -- vLLM's fork of the same file
(``vllm/tokenizers/deepseek_v4_encoding.py``) renamed the assistant field
``reasoning_content`` to ``reasoning``, so feeding it upstream-shaped messages silently drops
every thinking block. :func:`_reasoning_field` reads the field name back out of whichever copy
got loaded rather than assuming either spelling.

Loading it is remote code execution, so it is gated on the same ``trust_remote_code`` flag the
weights are. When the file cannot be fetched the engine keeps loading and simply reports no
chat support, which lands the caller on the existing raw-text refusal rather than on a failed
load -- chat is one endpoint, not the model.

Adding a family means one entry in :data:`CODE_CHAT_FORMATS` and a class satisfying
:class:`ChatFormatter`. Generation-side structure (reasoning delimiters, harmony channels)
stays in ``chat_conventions``; this module is only about rendering the prompt.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

logger = logging.getLogger(__name__)

Message = Mapping[str, Any]


class ChatFormatterUnavailable(RuntimeError):
    """The architecture needs a code formatter and the engine could not load one.

    Raised by the loaders, caught by :func:`resolve_chat_formatter`, which downgrades it to a
    warning: a model whose chat format is unreachable still captures, steers and completes raw
    text, so this must not gate loading.
    """


@dataclass(frozen=True)
class RenderedChat:
    """A chat render, kept split at the boundaries span metadata is derived from.

    ``blocks`` is 1:1 with the input messages, in order, so ``blocks[k]`` is exactly what
    message ``k`` contributed to the prompt. ``prefix`` holds what precedes message 0 (a BOS
    token, an injected tools preamble) and ``suffix`` the trailing generation scaffold.

    The split is the point. A Jinja template can only be rendered whole, so ``message_spans``
    has to recover boundaries by re-rendering growing message prefixes and diffing tokens --
    which assumes that growing the message list only appends, and DeepSeek-V4 breaks that
    assumption (whether a turn is the *last* user turn changes how earlier turns render). A
    formatter that reports its own boundaries sidesteps the diff entirely: ``upto`` returns a
    genuine string prefix of ``text``, so the token prefix is exact.
    """

    prefix: str
    blocks: tuple[str, ...]
    suffix: str

    @property
    def text(self) -> str:
        """The whole prompt, exactly as it would be tokenized."""
        return self.prefix + "".join(self.blocks) + self.suffix

    def upto(self, count: int) -> str:
        """``text`` truncated to the first ``count`` messages, generation scaffold excluded."""
        return self.prefix + "".join(self.blocks[:count])


class ChatFormatter(Protocol):
    """What :class:`~interp_engine.tokenize.Tokenize` needs from a code-defined chat format.

    ``template_kwargs`` is the set of keyword arguments the format actually reads. Callers use
    it the way they used to grep the Jinja source for a variable name: to pass
    ``enable_thinking`` only where it means something. A formatter refuses an unknown kwarg
    rather than ignoring it, since a silently dropped ``enable_thinking`` renders the wrong
    prompt and returns 200.
    """

    name: str
    template_kwargs: frozenset[str]

    def render(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **template_kwargs: Any,
    ) -> RenderedChat: ...


# --------------------------------------------------------------------------- #
# DeepSeek-V4
# --------------------------------------------------------------------------- #

# Where the reference encoder sits in every DeepSeek-V4 repo (V4-Flash, V4-Pro). The folder
# also carries README.md (the format spec) and tests/ (input/expected-output fixtures, which
# validator/ uses as golden cases).
DEEPSEEK_V4_ENCODER_FILE = "encoding/encoding_dsv4.py"

# vLLM vendors the same file. Only consulted when the checkpoint's own copy cannot be fetched,
# and it is a fork rather than a mirror -- see the module docstring.
DEEPSEEK_V4_FALLBACK_MODULE = "vllm.tokenizers.deepseek_v4_encoding"


class DeepseekV4Formatter:
    """Renders DeepSeek-V4 prompts through the encoder shipped with the checkpoint.

    Three of the encoder's conventions differ from the ``apply_chat_template`` vocabulary the
    rest of the engine speaks, and each is translated here rather than at the call sites:

    - **There is no ``add_generation_prompt`` flag.** The encoder appends
      ``<｜Assistant｜>`` plus a thinking delimiter whenever the conversation ends on a user or
      developer turn, unconditionally. So the scaffold is *identified* after the fact and
      reported as :attr:`RenderedChat.suffix`, which callers keep or drop. Note this makes
      ``add_generation_prompt=True`` a no-op after a closed assistant turn, where a Jinja
      template would open a fresh one: the encoder's own answer is that a transcript ending in
      a completed answer is a transcript, not a prompt, and synthesizing the opener anyway
      would put two tokens in front of the model that the checkpoint's own reference encoder
      never emits. The fixtures in ``encoding/tests/`` are exactly such transcripts.
    - **There is no ``continue_final_message`` flag.** A prefill is spelled per message, as
      ``wo_eos``, which suppresses that turn's end-of-sentence token. That is a better fit
      than the flag it replaces: transformers implements ``continue_final_message`` by
      rendering a sentinel into the content and cutting the string, and here nothing is cut.
    - **Historical reasoning is dropped by default** (``drop_thinking``), which is what makes
      the render non-monotonic and is why this class reports message blocks itself.
    """

    name = "deepseek_v4"
    # `thinking` is the encoder's own spelling and `enable_thinking` the one every other family
    # in this engine uses; both are accepted so a caller need not know which model it has.
    template_kwargs = frozenset({"enable_thinking", "thinking", "reasoning_effort", "drop_thinking", "tools"})

    def __init__(self, module: ModuleType):
        self._module = module
        self._encode: Callable[..., str] = getattr(module, "encode_messages")  # noqa: B009 - module attr
        self._reasoning_key = _reasoning_field(module)
        self._assistant_token = str(getattr(module, "ASSISTANT_SP_TOKEN", "<｜Assistant｜>"))
        self._think_open = str(getattr(module, "thinking_start_token", "<think>"))
        self._think_close = str(getattr(module, "thinking_end_token", "</think>"))
        self._bos = str(getattr(module, "bos_token", ""))
        self._encode_params = _accepted_parameters(self._encode)

    # --- rendering ----------------------------------------------------------
    def render(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **template_kwargs: Any,
    ) -> RenderedChat:
        config, tools = self._encode_config(template_kwargs)
        turns = [self._normalize(m) for m in messages]

        if continue_final_message:
            if not turns or turns[-1].get("role") != "assistant":
                raise ValueError(
                    "continue_final_message=True keeps a trailing ASSISTANT turn open, but the last "
                    f"message is {turns[-1].get('role') if turns else 'absent'!r}."
                )
            turns[-1] = {**turns[-1], "wo_eos": True}

        # Tools ride on a system message in this format. Injecting one shifts every index, so
        # its render is folded into `prefix` and `blocks` stays 1:1 with the input messages.
        lead: list[dict[str, Any]] = [{"role": "system", "tools": list(tools)}] if tools else []
        conversation = lead + turns

        tails = self._tails(conversation, config)
        blocks = [tails[k][: len(tails[k]) - len(tails[k + 1])] for k in range(len(conversation))]
        prefix = self._bos + "".join(blocks[: len(lead)])
        blocks = blocks[len(lead) :]

        self._reassign_interior_openers(blocks, turns)
        scaffold = self._split_scaffold(blocks, turns)
        return RenderedChat(prefix=prefix, blocks=tuple(blocks), suffix=scaffold if add_generation_prompt else "")

    def _tails(self, conversation: Sequence[Message], config: dict[str, Any]) -> list[str]:
        """``tails[k]`` = messages ``k..end`` rendered *in the context of the whole conversation*.

        The encoder's own ``context`` argument is what makes per-message blocks exact. Passing
        ``messages[k:]`` with ``context=messages[:k]`` renders only the tail, but computes
        ``last_user_index`` -- the fact that makes this format non-monotonic -- over the whole
        list, so ``tails[k] == block_k + tails[k + 1]`` holds by construction and subtracting
        one from the next recovers each block exactly.
        """
        tails = [""] * (len(conversation) + 1)
        for k in range(len(conversation) - 1, -1, -1):
            tails[k] = self._call_encoder(
                list(conversation[k:]),
                context=list(conversation[:k]),
                **config,
            )
            if not tails[k].endswith(tails[k + 1]):
                # Would mean the encoder's `context` no longer decomposes the render this way,
                # which is the assumption every span position rests on.
                raise ChatFormatterUnavailable(
                    f"{DEEPSEEK_V4_ENCODER_FILE} did not render message {k} as a prefix of the "
                    "messages after it, so message boundaries cannot be located. The encoder's "
                    "`context` argument may have changed meaning."
                )
        return tails

    def _call_encoder(self, messages: list[Any], **kwargs: Any) -> str:
        """Call ``encode_messages``, passing only arguments this copy of it declares.

        The checkpoint's encoder and vLLM's fork of it do not have identical signatures, and a
        future revision may add or drop a knob. Filtering here means an unknown argument
        degrades to the encoder's own default instead of raising ``TypeError`` at render time.
        """
        kwargs.setdefault("add_default_bos_token", False)
        if self._encode_params is not None:
            kwargs = {k: v for k, v in kwargs.items() if k in self._encode_params}
        return str(self._encode(messages, **kwargs))

    # --- translation --------------------------------------------------------
    def _encode_config(self, template_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], Sequence[Any]]:
        """Map the engine's template kwargs onto ``encode_messages`` arguments."""
        unknown = sorted(set(template_kwargs) - self.template_kwargs)
        if unknown:
            raise ValueError(
                f"{self.name} does not read {', '.join(unknown)}. It accepts: "
                f"{', '.join(sorted(self.template_kwargs))}."
            )
        thinking = bool(template_kwargs.get("enable_thinking") or template_kwargs.get("thinking"))
        effort = template_kwargs.get("reasoning_effort")
        effort = effort if isinstance(effort, str) else None
        if effort == "none":
            thinking, effort = False, None
        elif effort in ("max", "xhigh"):
            effort = "max"
        elif effort == "low":
            # The encoder only branches on "max"; it asserts on anything outside
            # {"max", "high", None}, and treats "high" and None identically. So "low" is None
            # rather than "high" -- same render either way, but it does not claim otherwise.
            effort = None
        elif effort is not None:
            effort = "high"
        return (
            {
                "thinking_mode": "thinking" if thinking else "chat",
                "drop_thinking": bool(template_kwargs.get("drop_thinking", True)),
                "reasoning_effort": effort,
            },
            template_kwargs.get("tools") or (),
        )

    def _normalize(self, message: Message) -> dict[str, Any]:
        """A message dict spelling reasoning the way the loaded encoder reads it."""
        turn = dict(message)
        alias = "reasoning" if self._reasoning_key == "reasoning_content" else "reasoning_content"
        if alias in turn:
            value = turn.pop(alias)
            turn.setdefault(self._reasoning_key, value)
        return turn

    def _reassign_interior_openers(self, blocks: list[str], turns: Sequence[Message]) -> None:
        """Move each assistant opener onto the assistant turn it opens, in place.

        The encoder writes ``<｜Assistant｜>`` plus a thinking delimiter at the *tail* of every
        user/developer turn -- including interior ones that a completed assistant answer follows,
        not just the last turn whose opener is the generation scaffold. Because ``_tails``
        decomposes the render by common suffix, that opener lands at the end of the *user* block,
        so span metadata attributes it to the user turn: the opener renders inside the user's
        bubble and the assistant's bubble starts on bare content. This peels it off the user turn
        and prepends it to the assistant turn it belongs to, leaving ``RenderedChat.text``
        unchanged (a substring only moves across the block boundary). The final turn's opener has
        no assistant turn after it and is handled by :meth:`_split_scaffold` instead.
        """
        for k in range(len(blocks) - 1):
            turn = turns[k]
            if turn.get("role") not in ("user", "developer") or turn.get("task") is not None:
                continue
            if turns[k + 1].get("role") != "assistant":
                continue
            for delimiter in (self._think_open, self._think_close):
                opener = self._assistant_token + delimiter
                if blocks[k].endswith(opener):
                    blocks[k] = blocks[k][: -len(opener)]
                    blocks[k + 1] = opener + blocks[k + 1]
                    break

    def _split_scaffold(self, blocks: list[str], turns: Sequence[Message]) -> str:
        """Remove and return the trailing generation scaffold from ``blocks``, or ``""``.

        The encoder emits it only after a user or developer turn that carries no ``task``, so
        that is the only case checked. Narrowing it matters twice over: assistant content that
        happened to end in these two tokens would otherwise be mistaken for scaffolding, and a
        ``task`` turn puts its own task token *after* the scaffold, so the suffix match would
        miss it and the strip would cut in the wrong place.
        """
        if not blocks or not turns:
            return ""
        last = turns[-1]
        if last.get("role") not in ("user", "developer") or last.get("task") is not None:
            return ""
        for delimiter in (self._think_open, self._think_close):
            scaffold = self._assistant_token + delimiter
            if blocks[-1].endswith(scaffold):
                blocks[-1] = blocks[-1][: -len(scaffold)]
                return scaffold
        return ""


def _reasoning_field(module: ModuleType) -> str:
    """The message key this copy of the encoder reads a thinking block from.

    Read out of ``thinking_template`` -- ``"{reasoning_content}"`` upstream,
    ``"{reasoning}"`` in vLLM's fork -- rather than assumed, because guessing wrong drops
    every thinking block without raising.
    """
    template = str(getattr(module, "thinking_template", "") or "")
    match = re.fullmatch(r"\s*\{(\w+)\}\s*", template)
    if match is None:
        logger.warning("Could not read the reasoning field name from %r; assuming reasoning_content", template)
        return "reasoning_content"
    return match.group(1)


def _accepted_parameters(func: Callable[..., Any]) -> frozenset[str] | None:
    """The keyword arguments ``func`` declares. ``None`` means "anything" (it takes ``**kwargs``)."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return frozenset(parameters)


# --------------------------------------------------------------------------- #
# Loading the checkpoint's own encoder
# --------------------------------------------------------------------------- #


def _download_repo_file(hf_model_id: str, filename: str) -> str | None:
    """Local path to ``filename`` from the checkpoint repo, cache first, then the hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover - transformers depends on it, so this is belt-and-braces
        logger.warning("huggingface_hub is not installed, so %s cannot be fetched", filename)
        return None
    last: Exception | None = None
    for local_files_only in (True, False):
        try:
            return hf_hub_download(hf_model_id, filename, local_files_only=local_files_only)
        except Exception as exc:  # noqa: BLE001 - offline, gated, or absent: all mean "no file"
            last = exc
    logger.info("Could not fetch %s from %s (%s)", filename, hf_model_id, last)
    return None


def _import_file(path: str, module_name: str) -> ModuleType | None:
    """Import a standalone ``.py`` file under ``module_name``."""
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so the module can import itself by name if it ever needs to; also
    # what makes the check above a cache rather than a repeated exec.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _module_alias(hf_model_id: str, suffix: str) -> str:
    """A ``sys.modules`` key unique to this checkpoint, so two revisions cannot collide."""
    return f"interp_engine._chat_encoders.{re.sub(r'[^0-9A-Za-z_]+', '_', hf_model_id)}_{suffix}"


def load_deepseek_v4_formatter(hf_model_id: str, *, trust_remote_code: bool = True) -> DeepseekV4Formatter:
    """Load DeepSeek-V4's reference encoder and wrap it as a :class:`ChatFormatter`.

    Prefers the copy inside the checkpoint, which is the format's source of truth and travels
    with the weights. Falls back to vLLM's vendored fork only when that file cannot be
    fetched -- it is a fork, and the module docstring lists how it has diverged.
    """
    if not trust_remote_code:
        raise ChatFormatterUnavailable(
            f"{hf_model_id} defines its chat format in {DEEPSEEK_V4_ENCODER_FILE} rather than in a "
            "chat template, and importing it is remote code execution. Load with "
            "trust_remote_code=True to render chat messages for this model."
        )

    path = _download_repo_file(hf_model_id, DEEPSEEK_V4_ENCODER_FILE)
    if path is not None:
        module = _import_file(path, _module_alias(hf_model_id, "dsv4"))
        if module is not None and hasattr(module, "encode_messages"):
            logger.info("Loaded the DeepSeek-V4 chat encoder from %s", path)
            return DeepseekV4Formatter(module)

    try:
        module = importlib.import_module(DEEPSEEK_V4_FALLBACK_MODULE)
    except ImportError:
        module = None
    if module is not None:
        logger.warning(
            "Using vLLM's vendored copy of the DeepSeek-V4 encoder: %s could not be fetched from %s. "
            "It is a fork of the checkpoint's own encoder, not a mirror.",
            DEEPSEEK_V4_ENCODER_FILE,
            hf_model_id,
        )
        return DeepseekV4Formatter(module)

    raise ChatFormatterUnavailable(
        f"{hf_model_id} ships no chat template; its format is defined by "
        f"{DEEPSEEK_V4_ENCODER_FILE}, which could not be fetched from the repo, and "
        f"{DEEPSEEK_V4_FALLBACK_MODULE} is not importable either."
    )


# Architecture *prefixes* (as they appear in ``config.architectures``) whose chat format is
# code rather than a template, mapped to the loader that builds it. Prefixes and families
# rather than checkpoints, matching `facts.MANDATORY_KV_CACHE_DTYPES`: this follows from the
# format the family was post-trained on, so every DeepSeek-V4 checkpoint shares it.
CODE_CHAT_FORMATS: dict[str, Callable[..., ChatFormatter]] = {
    "DeepseekV4": load_deepseek_v4_formatter,
}


_CACHE: dict[tuple[str, str, bool], ChatFormatter | None] = {}


def resolve_chat_formatter(
    architectures: Sequence[str] | None,
    hf_model_id: str,
    *,
    trust_remote_code: bool = True,
) -> ChatFormatter | None:
    """The code formatter these architectures need, or ``None`` when a template is enough.

    ``None`` is the answer for the great majority of models, and also the answer when a
    formatter is needed but unreachable -- that case logs a warning naming the file. Loading a
    model must not fail because one endpoint is unavailable, and the caller already has a
    refusal for "this model cannot take chat input" (``NoChatTemplateError``) that says to send
    raw text instead.

    Takes the config's whole ``architectures`` list for the reason
    :func:`facts.mandatory_kv_cache_dtype` does: it is the shape callers already hold, and a
    composite config can name more than one.
    """
    for name in architectures or ():
        for prefix, loader in CODE_CHAT_FORMATS.items():
            if not str(name).startswith(prefix):
                continue
            key = (str(name), hf_model_id, bool(trust_remote_code))
            if key not in _CACHE:
                try:
                    _CACHE[key] = loader(hf_model_id, trust_remote_code=trust_remote_code)
                except ChatFormatterUnavailable as exc:
                    logger.warning("No chat formatter for %s: %s", hf_model_id, exc)
                    _CACHE[key] = None
            return _CACHE[key]
    return None
