from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI
from pydantic import BaseModel

from lemurian.streaming import StreamChunk, ToolCallFragment

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ModelProvider:
    """Base class for model providers.

    Subclass this to implement a provider for a specific LLM API.
    Providers receive pre-built message and tool schema dicts from
    the Runner — they do not interact with lemurian types directly.
    """

    def __init__(self):
        pass

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        """Send a chat completion request to the model.

        Args:
            model: The model identifier.
            messages: List of message dicts in OpenAI format.
            tools: List of tool schema dicts, or None if no tools.

        Returns:
            The provider's response message object.
        """
        pass

    async def stream_complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion as StreamChunk objects.

        Subclasses should override this. The default raises
        NotImplementedError.
        """
        raise NotImplementedError
        yield  # make this a generator  # pragma: no cover

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        """Send a completion request expecting a structured response.

        Args:
            model: The model identifier.
            messages: List of message dicts in OpenAI format.
            response_model: A Pydantic model defining the response schema.

        Returns:
            The parsed response matching the response_model.
        """
        pass


async def _openai_stream_complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    **extra_kwargs,
) -> AsyncIterator[StreamChunk]:
    """Shared streaming helper for all OpenAI-SDK-based providers."""
    kwargs: dict = {"model": model, "messages": messages, "stream": True}
    if tools:
        kwargs["tools"] = tools
    kwargs.update(extra_kwargs)

    stream = await client.chat.completions.create(**kwargs)
    async for chunk in stream:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        delta = choice.delta
        fragments = None
        if delta.tool_calls:
            fragments = [
                ToolCallFragment(
                    index=tc.index,
                    call_id=tc.id,
                    name=tc.function.name if tc.function else None,
                    arguments_delta=(
                        tc.function.arguments if tc.function else None
                    ),
                )
                for tc in delta.tool_calls
            ]

        yield StreamChunk(
            content_delta=delta.content,
            tool_call_fragments=fragments,
            finish_reason=choice.finish_reason,
        )


class OpenAIProvider(ModelProvider):
    """Provider for the OpenAI API."""

    def __init__(self, api_key: str | None = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            api_key=api_key, max_retries=5, timeout=600.0,
        )

    async def complete(self, model, messages, tools=None):
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def stream_complete(self, model, messages, tools=None):
        async for chunk in _openai_stream_complete(
            self.client, model, messages, tools,
        ):
            yield chunk

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.responses.parse(
            model=model,
            input=messages,  # type: ignore[arg-type]  # OpenAI SDK overloaded unions
            text_format=response_model,  # type: ignore[arg-type]
        )
        return response.output_parsed


class OpenRouter(ModelProvider):
    """Provider for the OpenRouter API."""

    def __init__(self, api_key: str | None = None):
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key, max_retries=5, timeout=180.0,
        )

    async def complete(self, model, messages, tools=None):
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def stream_complete(self, model, messages, tools=None):
        async for chunk in _openai_stream_complete(
            self.client, model, messages, tools,
        ):
            yield chunk

    async def structured_completion(self, model, messages, response_model):
        response = await self.client.responses.parse(
            model=model,
            input=messages,
            text_format=response_model,
        )
        return response.output_parsed


class VLLMProvider(ModelProvider):
    """Provider for a locally-served vLLM instance."""

    def __init__(self, url: str, port: int):
        self.base_url = f"http://{url}:{port}/v1"
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="DUMMY")

    async def complete(self, model, messages, tools=None):
        kwargs: dict = {"model": model, "messages": messages, "tool_choice": "auto"}
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def stream_complete(self, model, messages, tools=None):
        async for chunk in _openai_stream_complete(
            self.client, model, messages, tools, tool_choice="auto",
        ):
            yield chunk

    async def structured_completion(self, model, messages, response_model):
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed  # type: ignore[attr-defined]


class ModalVLLMProvider(ModelProvider):
    """Provider for vLLM deployed on Modal."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        if not self.endpoint_url.endswith("/v1"):
            self.base_url = f"{self.endpoint_url}/v1"
        else:
            self.base_url = self.endpoint_url

        if not api_key:
            api_key = os.getenv("MODAL_VLLM_API_KEY", "DUMMY")

        self.client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )

    async def complete(self, model, messages, tools=None):
        kwargs: dict = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def stream_complete(self, model, messages, tools=None):
        extra = {"tool_choice": "auto"} if tools else {}
        async for chunk in _openai_stream_complete(
            self.client, model, messages, tools, **extra,
        ):
            yield chunk

    async def structured_completion(self, model, messages, response_model):
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed  # type: ignore[attr-defined]
