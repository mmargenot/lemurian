from dataclasses import dataclass

from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class CompletionResult:
    """Wraps a provider response with optional usage metadata.

    Args:
        content: The text content of the response.
        tool_calls: List of tool call objects, or None.
        usage: Token usage object from the provider, or None.
        response_model: The actual model string returned by the
            provider, or None.
    """

    content: str | None = None
    tool_calls: list | None = None
    usage: object | None = None
    response_model: str | None = None


class ModelProvider:
    """Base class for model providers.

    Subclass this to implement a provider for a specific LLM API.
    Providers receive pre-built message and tool schema dicts from
    the Runner — they do not interact with lemurian types directly.
    """

    system_name: str = "unknown"

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


class OpenAIProvider(ModelProvider):
    """Provider for the OpenAI API.

    Args:
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
            environment variable if not provided.
    """

    system_name: str = "openai"

    def __init__(self, api_key: str | None = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            api_key=api_key,
            max_retries=5,
            timeout=600.0
        )
        self.client = client

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return CompletionResult(
            content=msg.content,
            tool_calls=msg.tool_calls,
            usage=getattr(response, "usage", None),
            response_model=getattr(response, "model", None),
        )

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
    """Provider for the OpenRouter API.

    Args:
        api_key: OpenRouter API key. Falls back to the
            ``OPENROUTER_API_KEY`` environment variable if not provided.
    """

    system_name: str = "openrouter"

    def __init__(self, api_key: str | None = None):
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=5,
            timeout=180.0
        )
        self.client = client

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return CompletionResult(
            content=msg.content,
            tool_calls=msg.tool_calls,
            usage=getattr(response, "usage", None),
            response_model=getattr(response, "model", None),
        )

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


class VLLMProvider(ModelProvider):
    """Provider for any vLLM endpoint, local or remote.

    Args:
        url: Address of the vLLM endpoint. Normalized automatically:
            no scheme → ``http://`` prepended, ``/v1`` appended if
            missing, trailing slashes stripped.
        timeout: Request timeout in seconds.
        max_retries: Maximum retries for failed requests.
    """

    system_name: str = "vllm"

    def __init__(
        self,
        url: str,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        normalized = url.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        self.base_url = normalized

        api_key = os.getenv("VLLM_API_KEY", "DUMMY")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return CompletionResult(
            content=msg.content,
            tool_calls=msg.tool_calls,
            usage=getattr(response, "usage", None),
            response_model=getattr(response, "model", None),
        )

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            response_format=response_model,  # type: ignore[arg-type]
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed  # type: ignore[attr-defined]
