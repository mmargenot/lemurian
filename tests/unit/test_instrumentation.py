"""Unit tests for the instrumentation module.

Tests use unittest.mock for OTel interactions.  ``opentelemetry-api``
is a dev dependency so we can import ``SpanKind`` / ``StatusCode``
directly for assertion accuracy.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.trace import SpanKind, StatusCode

import lemurian.instrumentation as inst
from lemurian.instrumentation import (
    agent_span,
    completion_span,
    record_error,
    record_usage,
    tool_span,
    uninstrument,
)


@pytest.fixture(autouse=True)
def _reset_tracer():
    """Ensure _tracer is reset to None before and after each test."""
    inst._tracer = None
    yield
    inst._tracer = None


# -------------------------------------------------------------------
# instrument()
# -------------------------------------------------------------------


class TestInstrument:
    def test_raises_without_otel_installed(self):
        with patch(
            "importlib.util.find_spec", return_value=None
        ):
            with pytest.raises(
                ImportError, match="pip install"
            ):
                inst.instrument()

    def _mock_otel(self, mock_trace):
        """Patch find_spec + sys.modules for a mock OTel env."""
        return (
            patch(
                "importlib.util.find_spec",
                return_value=MagicMock(),
            ),
            patch.dict(
                "sys.modules",
                {
                    "opentelemetry": MagicMock(
                        trace=mock_trace
                    ),
                    "opentelemetry.trace": mock_trace,
                },
            ),
        )

    def test_sets_global_tracer(self):
        mock_tracer = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.NoOpTracer = type("NoOpTracer", (), {})

        p1, p2 = self._mock_otel(mock_trace)
        with p1, p2:
            inst.instrument()

        assert inst._tracer is mock_tracer
        mock_trace.get_tracer.assert_called_once_with(
            "lemurian"
        )

    def test_custom_tracer_name(self):
        mock_tracer = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = mock_tracer
        mock_trace.NoOpTracer = type("NoOpTracer", (), {})

        p1, p2 = self._mock_otel(mock_trace)
        with p1, p2:
            inst.instrument(tracer_name="my-app")

        mock_trace.get_tracer.assert_called_once_with(
            "my-app"
        )

    def test_logs_warning_for_noop_tracer(self, caplog):
        """When no TracerProvider is configured the returned tracer
        is a NoOpTracer — instrument() should log a helpful message."""
        NoOpTracer = type("NoOpTracer", (), {})
        noop = NoOpTracer()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = noop
        mock_trace.NoOpTracer = NoOpTracer

        p1, p2 = self._mock_otel(mock_trace)
        with p1, p2:
            with caplog.at_level(
                logging.INFO,
                logger="lemurian.instrumentation",
            ):
                inst.instrument()

        assert any(
            "No TracerProvider configured" in r.message
            for r in caplog.records
        )


# -------------------------------------------------------------------
# uninstrument()
# -------------------------------------------------------------------


class TestUninstrument:
    def test_clears_tracer(self):
        inst._tracer = MagicMock()
        uninstrument()
        assert inst._tracer is None

    def test_noop_when_already_none(self):
        uninstrument()  # should not raise
        assert inst._tracer is None


# -------------------------------------------------------------------
# Span helpers — uninstrumented (default)
# -------------------------------------------------------------------


class TestSpansUninstrumented:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "span_fn,args",
        [
            (agent_span, ("a", "m")),
            (completion_span, ("openai", "m")),
            (tool_span, ("t", "call_1")),
        ],
        ids=["agent_span", "completion_span", "tool_span"],
    )
    async def test_span_yields_none_without_tracer(
        self, span_fn, args
    ):
        async with span_fn(*args) as s:
            assert s is None


# -------------------------------------------------------------------
# Span helpers — instrumented
# Each test verifies the GenAI semantic convention attributes that
# form the public contract for that span type.
# -------------------------------------------------------------------


class TestSpansInstrumented:
    @pytest.fixture(autouse=True)
    def _set_mock_tracer(self):
        self.mock_span = MagicMock()
        self.mock_tracer = MagicMock()
        self.mock_tracer.start_as_current_span.return_value.__enter__ = (
            MagicMock(return_value=self.mock_span)
        )
        self.mock_tracer.start_as_current_span.return_value.__exit__ = (
            MagicMock(return_value=False)
        )
        inst._tracer = self.mock_tracer

    @pytest.mark.asyncio
    async def test_agent_span_creates_span(self):
        async with agent_span("support", "gpt-4o") as s:
            assert s is self.mock_span

        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "invoke_agent support",
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": "support",
                "gen_ai.request.model": "gpt-4o",
            },
        )

    @pytest.mark.asyncio
    async def test_completion_span_creates_span(self):
        async with completion_span("openai", "gpt-4o") as s:
            assert s is self.mock_span

        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "chat gpt-4o",
            kind=SpanKind.CLIENT,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o",
            },
        )

    @pytest.mark.asyncio
    async def test_tool_span_creates_span(self):
        async with tool_span("lookup", "call_42") as s:
            assert s is self.mock_span

        self.mock_tracer.start_as_current_span.assert_called_once_with(
            "execute_tool lookup",
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "lookup",
                "gen_ai.tool.call.id": "call_42",
            },
        )


# -------------------------------------------------------------------
# record_usage
# -------------------------------------------------------------------


class TestRecordUsage:
    def test_noop_on_none_span(self):
        """Silently returns when span is None (tracing disabled)."""
        usage = MagicMock(
            prompt_tokens=10, completion_tokens=5
        )
        record_usage(None, usage)  # should not raise

    def test_sets_token_counts_and_response_model(self):
        """Happy path: sets input tokens, output tokens, and
        response model attributes on the span."""
        span = MagicMock()
        usage = MagicMock(
            prompt_tokens=100, completion_tokens=50
        )
        record_usage(
            span, usage, response_model="gpt-4o-2024-08-06"
        )

        span.set_attribute.assert_any_call(
            "gen_ai.usage.input_tokens", 100
        )
        span.set_attribute.assert_any_call(
            "gen_ai.usage.output_tokens", 50
        )
        span.set_attribute.assert_any_call(
            "gen_ai.response.model", "gpt-4o-2024-08-06"
        )

    def test_handles_missing_token_fields(self):
        """Gracefully handles usage objects that lack token attrs
        (e.g. from a provider that doesn't report usage)."""
        span = MagicMock()
        usage = MagicMock(spec=[])  # no attributes at all
        record_usage(span, usage)  # should not raise
        span.set_attribute.assert_not_called()


# -------------------------------------------------------------------
# record_error
# -------------------------------------------------------------------


class TestRecordError:
    def test_sets_status_and_records_exception(self):
        span = MagicMock()
        exc = RuntimeError("boom")
        record_error(span, exc)

        span.set_status.assert_called_once_with(
            StatusCode.ERROR, "boom"
        )
        span.record_exception.assert_called_once_with(exc)
        span.set_attribute.assert_called_once_with(
            "error.type", "RuntimeError"
        )

    def test_noop_on_none_span(self):
        record_error(None, RuntimeError("boom"))  # no raise

    def test_error_type_uses_qualname(self):
        """Nested exception classes use __qualname__."""

        class Outer:
            class InnerError(Exception):
                pass

        span = MagicMock()
        exc = Outer.InnerError("nested")
        record_error(span, exc)

        span.set_attribute.assert_called_once_with(
            "error.type", "TestRecordError.test_error_type_uses_qualname.<locals>.Outer.InnerError"
        )
