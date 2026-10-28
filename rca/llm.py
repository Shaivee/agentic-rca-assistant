"""
A thin LLM wrapper used by every agent.

Why our own wrapper instead of calling the SDK everywhere?
  * One place to switch provider (Groq / OpenAI) or to plug in a fake LLM for tests.
  * One place to record cost and latency for every call (used in evaluation).
  * One place to enforce structured output: the LLM must return JSON that passes
    a Pydantic schema, or it gets one chance to fix it.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from rca.config import PROVIDERS, llm_provider

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger("rca")

OUTPUT_RETRIES = 3          # attempts when the model's output cannot be parsed
RETRY_TEMPERATURE = 0.4


class LLMOutputError(RuntimeError):
    """The model kept producing output (tool call or JSON) that could not be parsed."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_message(self) -> dict:
        """The assistant turn in OpenAI message format, to append to the history."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                                 for tc in self.tool_calls]
        return msg


@dataclass
class CallRecord:
    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class BaseLLM:
    model = "base"

    def __init__(self) -> None:
        self.calls: list[CallRecord] = []

    # ---- implemented by subclasses -----------------------------------------
    def _chat(self, messages: list[dict], tools: list[dict] | None, json_mode: bool) -> ChatResult:
        raise NotImplementedError

    # ---- public API ---------------------------------------------------------
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False, node: str = "?") -> ChatResult:
        t0 = time.perf_counter()
        result = self._chat(messages, tools, json_mode)
        self.calls.append(CallRecord(node, self.model, result.prompt_tokens, result.completion_tokens,
                                     round(time.perf_counter() - t0, 3)))
        return result

    def structured(self, messages: list[dict], schema: Type[T], node: str = "?") -> T:
        """Ask for JSON matching `schema`. If validation fails, show the model its
        error once and let it correct itself (a cheap, very effective retry)."""
        instruction = {"role": "user", "content":
                       "Respond with ONLY a JSON object (no prose, no markdown) that matches this JSON schema:\n"
                       + json.dumps(schema.model_json_schema())}
        convo = messages + [instruction]
        for attempt in range(2):
            raw = self.chat(convo, json_mode=True, node=node).content
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError) as err:
                if attempt == 1:
                    raise
                convo = convo + [{"role": "assistant", "content": raw},
                                 {"role": "user", "content": f"That JSON was invalid: {err}. Return corrected JSON only."}]
        raise RuntimeError("unreachable")

    def usage_summary(self) -> dict:
        return {"llm_calls": len(self.calls),
                "prompt_tokens": sum(c.prompt_tokens for c in self.calls),
                "completion_tokens": sum(c.completion_tokens for c in self.calls),
                "llm_seconds": round(sum(c.latency_s for c in self.calls), 2)}


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences or add a sentence; take the object."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in the model output.")
    return match.group(0)


class OpenAICompatibleLLM(BaseLLM):
    """Works for OpenAI and Groq (Groq's API is OpenAI-compatible)."""

    def __init__(self, provider: str | None = None, model: str | None = None, temperature: float = 0.0):
        super().__init__()
        from openai import OpenAI
        provider = provider or llm_provider()
        if provider not in PROVIDERS:
            raise RuntimeError("No LLM configured. Put GROQ_API_KEY=... (or OPENAI_API_KEY=...) in .env")
        base_url, key_var, default_model = PROVIDERS[provider]
        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL") or default_model
        self.temperature = temperature        # 0 = as repeatable as possible
        # max_retries handles rate limits (HTTP 429) with exponential back-off - common on free tiers.
        # Accept-Encoding: ask for gzip (built into Python) instead of brotli/zstd, whose decoders
        # depend on optional packages that may be outdated in some environments (e.g. Anaconda).
        self.client = OpenAI(api_key=os.getenv(key_var), base_url=base_url, max_retries=6,
                             default_headers={"Accept-Encoding": "gzip, deflate"})

    def _chat(self, messages, tools, json_mode):
        from openai import BadRequestError, NotFoundError
        kwargs: dict = {"model": self.model, "messages": messages}
        if "gpt-oss" in self.model:
            # gpt-oss is a reasoning model; its hidden "thinking" tokens count against the free tier's
            # tokens-per-minute limit. Low effort is plenty for tool selection and report writing.
            kwargs["reasoning_effort"] = os.getenv("LLM_REASONING_EFFORT", "low")
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = False   # one call per turn: open models format these more reliably
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = ""
        for attempt in range(OUTPUT_RETRIES):
            # Attempt 1 is (near-)deterministic. A retry of the SAME request at temperature 0 tends to
            # reproduce the SAME broken output, so later attempts add a little randomness.
            kwargs["temperature"] = self.temperature if attempt == 0 else RETRY_TEMPERATURE
            try:
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                calls = [ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
                         for tc in (msg.tool_calls or [])]
            except NotFoundError as err:
                # Providers retire models; make the fix obvious instead of showing a bare 404.
                try:
                    available = sorted(m.id for m in self.client.models.list().data)
                except Exception:                         # noqa: BLE001
                    available = ["(could not list models)"]
                raise RuntimeError(f"Model '{self.model}' is not available from {self.provider}. "
                                   f"Set LLM_MODEL in .env to one of: {', '.join(available)}") from err
            except BadRequestError as err:
                # Groq returns 400 when the model's output cannot be parsed (malformed tool call or
                # JSON: "tool_use_failed", "json_validate_failed", "Parsing failed"). Anything else is
                # a genuine request problem and is raised immediately.
                body = err.body if isinstance(err.body, dict) else {}
                info = body.get("error", body) if isinstance(body.get("error", body), dict) else {}
                if not ("failed" in str(err) or info.get("failed_generation")):
                    raise
                last_error = str(info.get("failed_generation") or err)[:300]
                log.warning("Model output could not be parsed (attempt %d/%d). Start of output: %r",
                            attempt + 1, OUTPUT_RETRIES, last_error[:160])
                continue
            except json.JSONDecodeError as err:            # tool arguments that are not valid JSON
                last_error = f"invalid tool arguments: {err}"
                log.warning("Model produced invalid tool arguments (attempt %d/%d)", attempt + 1, OUTPUT_RETRIES)
                continue
            usage = resp.usage
            return ChatResult(msg.content or "", calls,
                              usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        raise LLMOutputError(last_error)


class ScriptedLLM(BaseLLM):
    """A fake LLM for tests: returns pre-written responses in order. This lets us
    test the graph wiring and the real tools without an API key or randomness."""
    model = "scripted"

    def __init__(self, responses: list[ChatResult | str | Exception]):
        super().__init__()
        self.responses = list(responses)
        self.seen: list[list[dict]] = []

    def _chat(self, messages, tools, json_mode):
        self.seen.append(messages)
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return ChatResult(content=r) if isinstance(r, str) else r


def get_llm() -> BaseLLM:
    return OpenAICompatibleLLM()
