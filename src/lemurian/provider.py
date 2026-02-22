from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import logging

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
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.responses.parse(
            model=model,
            input=messages,
            text_format=response_model,
        )
        return response.output_parsed


class OpenRouter(ModelProvider):
    """Provider for the OpenRouter API.

    Args:
        api_key: OpenRouter API key. Falls back to the
            ``OPENROUTER_API_KEY`` environment variable if not provided.
    """

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
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.responses.parse(
            model=model,
            input=messages,
            text_format=response_model,
        )
        return response.output_parsed


class VLLMProvider(ModelProvider):
    """Provider for a locally-served vLLM instance.

    Args:
        url: Hostname or IP of the vLLM server.
        port: Port the vLLM server is listening on.
    """

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
            "tool_choice": "auto",
        }
        if tools:
            kwargs["tools"] = tools
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed


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
        return response.choices[0].message

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_model,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed
