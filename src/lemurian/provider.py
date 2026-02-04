from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
import os
import logging

from lemurian.message import Message
from lemurian.tools import Tool

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
            messages: list[Message],
            all_tools: list[Tool] = []
    ):
        pass

    async def structured_completion(
            self,
            model: str,
            messages: list[Message],
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
            messages: list[Message],
            all_tools: list[Tool] = []
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.chat.completions.create(
            model=model,
            tools=all_tools,
            messages=message_dump,
        )
        response = response.choices[0].message
        return response

    async def structured_completion(
            self,
            model: str,
            messages: list[Message],
            response_model: BaseModel,
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.responses.parse(
            model=model,
            input=message_dump,
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
            messages: list[Message],
            all_tools: list[Tool] = []
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.chat.completions.create(
            model=model,
            tools=all_tools,
            messages=message_dump,
        )
        response = response.choices[0].message
        return response

    async def structured_completion(
            self,
            model: str,
            messages: list[Message],
            response_model: BaseModel,
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.responses.parse(
            model=model,
            input=message_dump,
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
            messages: list[Message],
            all_tools: list[Tool] = []
    ):
        message_dump = [m.model_dump() for m in messages]
        tool_call_schemas = [
            tool.model_dump()
            for tool in all_tools
        ]
        response = await self.client.chat.completions.create(
            model=model,
            messages=message_dump,
            tools=all_tools,
            tool_choice="auto",
        )
        response = response.choices[0].message
        return response

    async def structured_completion(
            self,
            model: str,
            messages: list[Message],
            response_model: BaseModel,
    ):
        # TODO: check for async or not and handle it with client
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=message_dump,
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
        agent = MyAgent(model="Qwen/Qwen3-8B", provider=provider)
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 3,
    ):
        """
        Initialize the Modal vLLM provider.

        Args:
            endpoint_url: The Modal deployment URL (e.g., https://...modal.run)
            api_key: Optional API key if Modal authentication is configured.
                     Can also be set via MODAL_VLLM_API_KEY environment variable.
            timeout: Request timeout in seconds (default: 300.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
        """
        # Ensure URL ends with /v1 for OpenAI compatibility
        self.endpoint_url = endpoint_url.rstrip("/")
        if not self.endpoint_url.endswith("/v1"):
            self.base_url = f"{self.endpoint_url}/v1"
        else:
            self.base_url = self.endpoint_url

        # API key from parameter or environment variable
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
            messages: list[Message],
            all_tools: list[Tool] = []
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.chat.completions.create(
            model=model,
            messages=message_dump,
            tools=all_tools if all_tools else None,
            tool_choice="auto" if all_tools else None,
        )
        response = response.choices[0].message
        return response

    async def structured_completion(
            self,
            model: str,
            messages: list[Message],
            response_model: BaseModel,
    ):
        message_dump = [m.model_dump() for m in messages]
        response = await self.client.beta.chat.completions.parse(
            model=model,
            messages=message_dump,
            response_format=response_model,
            extra_body=dict(guided_decoding_backend="outlines"),
        )
        return response.output_parsed
