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
        response = await self.client.responses.parse(
            model=model,
            input=messages,  # type: ignore[arg-type]  # OpenAI SDK overloaded unions
            text_format=response_model,  # type: ignore[arg-type]
        )
        return response.output_parsed


class VLLMProvider(ModelProvider):
    """Provider for a locally-served vLLM instance.

    Args:
        url: Hostname or IP of the vLLM server.
        port: Port the vLLM server is listening on.
    """

    system_name: str = "vllm"

    def __init__(self, url: str, port: int):
        self.base_url = f"http://{url}:{port}/v1"
        self.client = AsyncOpenAI(base_url=self.base_url, api_key="DUMMY")

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
            messages=messages,  # type: ignore[arg-type]  # OpenAI SDK overloaded unions
            response_format=response_model,  # type: ignore[arg-type]
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed  # type: ignore[attr-defined]


class ModalVLLMProvider(ModelProvider):
    """Provider for vLLM deployed on Modal.

    Connects to a vLLM instance hosted on Modal's serverless
    infrastructure. Deploy with ``modal deploy examples/modal_vllm.py``.

    Args:
        endpoint_url: The Modal deployment URL
            (e.g. ``https://your-workspace--lemurian-vllm-vllmserver-serve.modal.run``).
        api_key: Optional API key. Falls back to the
            ``MODAL_VLLM_API_KEY`` environment variable, or ``"DUMMY"``.
        timeout: Request timeout in seconds.
        max_retries: Maximum retries for failed requests.
    """

    system_name: str = "vllm"

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
            base_url=self.base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.info(f"Initialized ModalVLLMProvider with endpoint: {self.base_url}")

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
            messages=messages,  # type: ignore[arg-type]  # OpenAI SDK overloaded unions
            response_format=response_model,  # type: ignore[arg-type]
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed  # type: ignore[attr-defined]


class OpenAICompatibleProvider(ModelProvider):
    """Provider for any OpenAI-compatible endpoint.

    Works with Ollama, Together, Groq, Fireworks, LM Studio, vLLM,
    TGI, and any server that implements the ``/chat/completions``
    endpoint.

    Args:
        base_url: Full base URL of the endpoint
            (e.g. ``"http://localhost:11434/v1"``).
        api_key: Optional API key. Defaults to ``"DUMMY"`` for
            local servers that don't require auth.
        timeout: Request timeout in seconds.
        max_retries: Maximum retries for failed requests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key or "DUMMY",
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
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": response_model.model_json_schema(),
                "name": response_model.__name__,  # type: ignore[attr-defined]  # always a class at runtime
                "strict": True,
            },
        }
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]  # OpenAI SDK overloaded unions
            response_format=response_format,  # type: ignore[arg-type]  # raw dict accepted at runtime
        )
        return response_model.model_validate_json(
            response.choices[0].message.content
        )


class LiteLLMProvider(ModelProvider):
    """Provider that delegates to LiteLLM for 100+ LLM APIs.

    Supports Anthropic, Gemini, Cohere, Bedrock, Mistral, Azure,
    and any other provider that LiteLLM supports. Uses
    ``litellm.acompletion()`` under the hood.

    LiteLLM is **not** a required dependency — it is imported lazily
    when this provider is instantiated. Install it separately::

        pip install litellm

    Model names use LiteLLM's ``"provider/model"`` format, e.g.
    ``"anthropic/claude-sonnet-4-20250514"`` or
    ``"gemini/gemini-2.0-flash"``.

    Args:
        api_key: Optional API key passed to every ``acompletion``
            call. Provider-specific env vars (e.g.
            ``ANTHROPIC_API_KEY``) also work.
        **kwargs: Extra keyword arguments forwarded to every
            ``litellm.acompletion`` call (e.g. ``api_base``,
            ``timeout``, ``max_retries``).
    """

    def __init__(self, api_key: str | None = None, **kwargs):
        try:
            import litellm  # type: ignore[unresolved-import]  # optional dependency
        except ImportError:
            raise ImportError(
                "LiteLLMProvider requires 'litellm'. "
                "Install: pip install litellm"
            )
        self._litellm = litellm
        self.api_key = api_key
        self._kwargs = kwargs

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        kwargs: dict = {
            "model": model,
            "messages": messages,
            **self._kwargs,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = await self._litellm.acompletion(**kwargs)
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": response_model.model_json_schema(),
                "name": response_model.__name__,  # type: ignore[attr-defined]  # always a class at runtime
                "strict": True,
            },
        }
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "response_format": response_format,
            **self._kwargs,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = await self._litellm.acompletion(**kwargs)
        return response_model.model_validate_json(
            response.choices[0].message.content
        )
