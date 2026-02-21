from pydantic import BaseModel
from openai import AsyncOpenAI
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('lemurian.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ModelProvider:
    def __init__(self):
        pass

    async def complete(
            self,
            model: str,
            messages: list[dict],
            tools: list[dict] | None = None,
    ):
        pass

    async def structured_completion(
            self,
            model: str,
            messages: list[dict],
            response_model: BaseModel,
    ):
        pass


class OpenAIProvider(ModelProvider):

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
    """
    Provider for vLLM deployed on Modal.

    This provider connects to a vLLM instance hosted on Modal's serverless
    infrastructure. Deploy vLLM to Modal using the modal_vllm.py script:

        modal deploy src/scripts/modal_vllm.py

    The deployment URL will be displayed after deployment completes.

    Example:
        provider = ModalVLLMProvider(
            endpoint_url="https://your-workspace--lemurian-vllm-vllmserver-serve.modal.run"
        )
        agent = Agent(name="my_agent", model="Qwen/Qwen3-8B", provider=provider, system_prompt="...")
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
