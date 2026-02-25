from dataclasses import dataclass, field
from types import ModuleType
from unittest.mock import AsyncMock

import pytest

from lemurian.provider import (
    LiteLLMProvider,
    ModalVLLMProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    OpenRouter,
    VLLMProvider,
)


# ---------------------------------------------------------------------------
# Fake OpenAI client response objects
# ---------------------------------------------------------------------------

@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list[FakeChoice] = field(default_factory=list)


def _fake_completion(content="hi", tool_calls=None):
    """Build a FakeCompletion matching the OpenAI client shape."""
    return FakeCompletion(
        choices=[FakeChoice(message=FakeMessage(
            content=content, tool_calls=tool_calls,
        ))]
    )


# ---------------------------------------------------------------------------
# URL normalisation (existing tests)
# ---------------------------------------------------------------------------

def test_modal_provider_appends_v1():
    p = ModalVLLMProvider(
        endpoint_url="https://workspace--app.modal.run",
        api_key="test",
    )
    assert p.base_url == "https://workspace--app.modal.run/v1"


def test_modal_provider_strips_trailing_slash():
    p = ModalVLLMProvider(
        endpoint_url="https://workspace--app.modal.run/",
        api_key="test",
    )
    assert p.base_url == "https://workspace--app.modal.run/v1"


def test_modal_provider_no_double_v1():
    p = ModalVLLMProvider(
        endpoint_url="https://workspace--app.modal.run/v1",
        api_key="test",
    )
    assert p.base_url == "https://workspace--app.modal.run/v1"


def test_openai_provider_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    p = OpenAIProvider()
    assert p.client.api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# OpenAIProvider.complete — kwargs forwarding
# ---------------------------------------------------------------------------

class TestOpenAIProviderComplete:
    @pytest.mark.asyncio
    async def test_forwards_tools_with_tool_choice(
        self, monkeypatch
    ):
        """complete() passes tools and tool_choice='auto' together."""
        provider = OpenAIProvider(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("hello")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        result = await provider.complete(
            "gpt-4o", messages, tools=tools,
        )

        mock_create.assert_called_once_with(
            model="gpt-4o", messages=messages,
            tools=tools, tool_choice="auto",
        )
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_omits_tools_and_tool_choice_when_none(
        self, monkeypatch
    ):
        """When tools is None, neither 'tools' nor 'tool_choice' is
        sent."""
        provider = OpenAIProvider(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("hi")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        await provider.complete(
            "gpt-4o", [{"role": "user", "content": "hi"}],
            tools=None,
        )

        _, kwargs = mock_create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    @pytest.mark.asyncio
    async def test_returns_message_from_first_choice(
        self, monkeypatch
    ):
        """complete() returns choices[0].message."""
        provider = OpenAIProvider(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("answer")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        result = await provider.complete("gpt-4o", [])
        assert result.content == "answer"


# ---------------------------------------------------------------------------
# VLLMProvider.complete — conditional tool_choice (bug fix)
# ---------------------------------------------------------------------------

class TestVLLMProviderComplete:
    @pytest.mark.asyncio
    async def test_omits_tool_choice_without_tools(
        self, monkeypatch
    ):
        """VLLMProvider omits tool_choice when no tools are provided."""
        provider = VLLMProvider(url="localhost", port=8000)
        mock_create = AsyncMock(
            return_value=_fake_completion("ok")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        await provider.complete("llama", [], tools=None)

        _, kwargs = mock_create.call_args
        assert "tool_choice" not in kwargs
        assert "tools" not in kwargs

    @pytest.mark.asyncio
    async def test_includes_tool_choice_with_tools(
        self, monkeypatch
    ):
        """VLLMProvider sends tool_choice='auto' when tools present."""
        provider = VLLMProvider(url="localhost", port=8000)
        mock_create = AsyncMock(
            return_value=_fake_completion("ok")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        tools = [{"type": "function", "function": {"name": "f"}}]
        await provider.complete("llama", [], tools=tools)

        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# ModalVLLMProvider.complete — conditional tool_choice
# ---------------------------------------------------------------------------

class TestModalVLLMProviderComplete:
    @pytest.mark.asyncio
    async def test_includes_tool_choice_only_with_tools(
        self, monkeypatch
    ):
        """ModalVLLMProvider only sends tool_choice when tools present."""
        provider = ModalVLLMProvider(
            endpoint_url="https://x.modal.run", api_key="k",
        )
        mock_create = AsyncMock(
            return_value=_fake_completion("ok")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        # Without tools — no tool_choice
        await provider.complete("m", [], tools=None)
        _, kwargs = mock_create.call_args
        assert "tool_choice" not in kwargs

        mock_create.reset_mock()

        # With tools — tool_choice present
        tools = [{"type": "function", "function": {"name": "f"}}]
        await provider.complete("m", [], tools=tools)
        _, kwargs = mock_create.call_args
        assert kwargs["tool_choice"] == "auto"


# ---------------------------------------------------------------------------
# OpenRouter.complete — same contract as OpenAIProvider
# ---------------------------------------------------------------------------

class TestOpenRouterComplete:
    @pytest.mark.asyncio
    async def test_forwards_tools_with_tool_choice(
        self, monkeypatch
    ):
        """OpenRouter sends tool_choice='auto' when tools present."""
        provider = OpenRouter(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("hi")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        tools = [{"type": "function", "function": {"name": "f"}}]
        await provider.complete("model", [], tools=tools)
        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_omits_tools_and_tool_choice_when_none(
        self, monkeypatch
    ):
        """OpenRouter omits tools and tool_choice when tools=None."""
        provider = OpenRouter(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("hi")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        await provider.complete("model", [], tools=None)
        _, kwargs = mock_create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs


# ---------------------------------------------------------------------------
# OpenAICompatibleProvider
# ---------------------------------------------------------------------------

class TestOpenAICompatibleProvider:
    def test_strips_trailing_slash(self):
        p = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1/"
        )
        assert p.base_url == "http://localhost:11434/v1"

    def test_defaults_api_key_to_dummy(self):
        p = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1"
        )
        assert p.client.api_key == "DUMMY"

    def test_uses_provided_api_key(self):
        p = OpenAICompatibleProvider(
            base_url="http://localhost/v1",
            api_key="real-key",
        )
        assert p.client.api_key == "real-key"

    @pytest.mark.asyncio
    async def test_complete_with_tools(self, monkeypatch):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1"
        )
        mock_create = AsyncMock(
            return_value=_fake_completion("ok")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        tools = [{"type": "function", "function": {"name": "f"}}]
        result = await provider.complete("llama3", [], tools=tools)

        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_complete_without_tools(self, monkeypatch):
        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1"
        )
        mock_create = AsyncMock(
            return_value=_fake_completion("ok")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        await provider.complete("llama3", [], tools=None)

        _, kwargs = mock_create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    @pytest.mark.asyncio
    async def test_structured_completion(self, monkeypatch):
        from pydantic import BaseModel

        class Weather(BaseModel):
            temperature: int
            unit: str

        provider = OpenAICompatibleProvider(
            base_url="http://localhost:11434/v1"
        )
        mock_create = AsyncMock(
            return_value=_fake_completion(
                '{"temperature": 72, "unit": "F"}'
            )
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        result = await provider.structured_completion(
            "llama3", [], Weather,
        )

        assert isinstance(result, Weather)
        assert result.temperature == 72
        assert result.unit == "F"

        _, kwargs = mock_create.call_args
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Weather"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == Weather.model_json_schema()


# ---------------------------------------------------------------------------
# LiteLLMProvider
# ---------------------------------------------------------------------------

class TestLiteLLMProvider:
    def _make_provider(self, mock_acompletion, **kwargs):
        """Create a LiteLLMProvider with a fake litellm module."""
        fake_litellm = ModuleType("litellm")
        fake_litellm.acompletion = mock_acompletion
        provider = object.__new__(LiteLLMProvider)
        provider._litellm = fake_litellm
        provider.api_key = kwargs.get("api_key")
        provider._kwargs = {
            k: v for k, v in kwargs.items() if k != "api_key"
        }
        return provider

    def test_import_error_without_litellm(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="LiteLLMProvider"):
            LiteLLMProvider()

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        mock_acompletion = AsyncMock(
            return_value=_fake_completion("ok")
        )
        provider = self._make_provider(mock_acompletion)

        tools = [{"type": "function", "function": {"name": "f"}}]
        result = await provider.complete(
            "anthropic/claude-sonnet-4-20250514", [], tools=tools,
        )

        mock_acompletion.assert_called_once()
        _, kwargs = mock_acompletion.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_complete_without_tools(self):
        mock_acompletion = AsyncMock(
            return_value=_fake_completion("ok")
        )
        provider = self._make_provider(mock_acompletion)

        await provider.complete("gemini/gemini-2.0-flash", [])

        _, kwargs = mock_acompletion.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        assert "api_key" not in kwargs

    @pytest.mark.asyncio
    async def test_api_key_forwarded(self):
        mock_acompletion = AsyncMock(
            return_value=_fake_completion("ok")
        )
        provider = self._make_provider(
            mock_acompletion, api_key="sk-test",
        )

        await provider.complete("anthropic/claude-sonnet-4-20250514", [])

        _, kwargs = mock_acompletion.call_args
        assert kwargs["api_key"] == "sk-test"

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        mock_acompletion = AsyncMock(
            return_value=_fake_completion("ok")
        )
        provider = self._make_provider(
            mock_acompletion, timeout=30, max_retries=2,
        )

        await provider.complete("model", [])

        _, kwargs = mock_acompletion.call_args
        assert kwargs["timeout"] == 30
        assert kwargs["max_retries"] == 2

    @pytest.mark.asyncio
    async def test_structured_completion(self):
        from pydantic import BaseModel

        class Sentiment(BaseModel):
            label: str
            score: float

        mock_acompletion = AsyncMock(
            return_value=_fake_completion(
                '{"label": "positive", "score": 0.95}'
            )
        )
        provider = self._make_provider(
            mock_acompletion, api_key="sk-test", timeout=60,
        )

        result = await provider.structured_completion(
            "anthropic/claude-sonnet-4-20250514", [], Sentiment,
        )

        assert isinstance(result, Sentiment)
        assert result.label == "positive"
        assert result.score == 0.95

        _, kwargs = mock_acompletion.call_args
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Sentiment"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == (
            Sentiment.model_json_schema()
        )
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["timeout"] == 60
