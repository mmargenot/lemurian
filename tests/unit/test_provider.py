from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from lemurian.provider import (
    ModalVLLMProvider,
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
    async def test_forwards_model_messages_and_tools(
        self, monkeypatch
    ):
        """complete() passes model, messages, and tools through."""
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
            model="gpt-4o", messages=messages, tools=tools,
        )
        assert result.content == "hello"

    @pytest.mark.asyncio
    async def test_omits_tools_key_when_none(self, monkeypatch):
        """When tools is None, the 'tools' key is not sent."""
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
# VLLMProvider.complete — always sends tool_choice
# ---------------------------------------------------------------------------

class TestVLLMProviderComplete:
    @pytest.mark.asyncio
    async def test_always_includes_tool_choice_auto(
        self, monkeypatch
    ):
        """VLLMProvider always sends tool_choice='auto'."""
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
    async def test_forwards_tools_and_omits_when_none(
        self, monkeypatch
    ):
        """OpenRouter follows the same tools-forwarding contract."""
        provider = OpenRouter(api_key="test-key")
        mock_create = AsyncMock(
            return_value=_fake_completion("hi")
        )
        monkeypatch.setattr(
            provider.client.chat.completions, "create",
            mock_create,
        )

        # With tools
        tools = [{"type": "function", "function": {"name": "f"}}]
        await provider.complete("model", [], tools=tools)
        _, kwargs = mock_create.call_args
        assert kwargs["tools"] == tools

        mock_create.reset_mock()

        # Without tools
        await provider.complete("model", [], tools=None)
        _, kwargs = mock_create.call_args
        assert "tools" not in kwargs
