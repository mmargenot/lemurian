"""Optional OpenTelemetry instrumentation for lemurian.

Call ``lemurian.instrument()`` once at startup to enable tracing.
Requires ``opentelemetry-api`` to be installed; the framework
works identically without it.
"""

import importlib.util
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

_tracer = None


def instrument(*, tracer_name: str = "lemurian") -> None:
    """Enable OpenTelemetry tracing for all lemurian operations.

    Call once at startup, after configuring your TracerProvider.
    Requires ``opentelemetry-api``: ``pip install lemurian[otel]``

    Example — explicit SDK setup::

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry import trace

        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter())
        )
        trace.set_tracer_provider(provider)

        import lemurian
        lemurian.instrument()

    Example — zero-code via ``opentelemetry-instrument``::

        $ pip install opentelemetry-distro opentelemetry-exporter-otlp
        $ opentelemetry-bootstrap -a install
        $ OTEL_SERVICE_NAME=my-agent \\
          OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
          opentelemetry-instrument python my_app.py

        # In my_app.py, just call:
        import lemurian
        lemurian.instrument()

    See also:
        - `OTel Python SDK <https://opentelemetry.io/docs/languages/python/>`_
        - `GenAI Semantic Conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
        - `Zero-code instrumentation <https://opentelemetry.io/docs/zero-code/python/>`_

    Args:
        tracer_name: Name passed to ``trace.get_tracer()``.

    Raises:
        ImportError: If ``opentelemetry-api`` is not installed.
    """
    global _tracer
    if importlib.util.find_spec("opentelemetry.trace") is None:
        raise ImportError(
            "opentelemetry-api is required for instrumentation. "
            "Install it with: pip install lemurian[otel]"
        )
    from opentelemetry import trace
    _tracer = trace.get_tracer(tracer_name)
    if isinstance(_tracer, trace.NoOpTracer):
        logger.info(
            "No TracerProvider configured — spans will be "
            "discarded. Set up a TracerProvider to export "
            "traces."
        )
    else:
        logger.info("Lemurian instrumentation enabled")


def uninstrument() -> None:
    """Disable OpenTelemetry tracing.

    Subsequent operations will not emit spans.
    """
    global _tracer
    _tracer = None


@asynccontextmanager
async def agent_span(agent_name: str, model: str):
    """Wrap a Runner.run() invocation in an ``invoke_agent`` span."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(
        f"invoke_agent {agent_name}",
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.agent.name": agent_name,
            "gen_ai.request.model": model,
        },
    ) as span:
        yield span


@asynccontextmanager
async def completion_span(system: str, model: str):
    """Wrap a provider.complete() call in a ``chat`` span."""
    if _tracer is None:
        yield None
        return
    from opentelemetry.trace import SpanKind

    with _tracer.start_as_current_span(
        f"chat {model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": system,
            "gen_ai.request.model": model,
        },
    ) as span:
        yield span


@asynccontextmanager
async def tool_span(tool_name: str, call_id: str):
    """Wrap a tool execution in an ``execute_tool`` span."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(
        f"execute_tool {tool_name}",
        attributes={
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "gen_ai.tool.call.id": call_id,
        },
    ) as span:
        yield span


def record_usage(
    span, usage, response_model: str | None = None
):
    """Set token-usage and response-model attributes on a span."""
    if span is None or usage is None:
        return
    if (
        hasattr(usage, "prompt_tokens")
        and usage.prompt_tokens is not None
    ):
        span.set_attribute(
            "gen_ai.usage.input_tokens",
            usage.prompt_tokens,
        )
    if (
        hasattr(usage, "completion_tokens")
        and usage.completion_tokens is not None
    ):
        span.set_attribute(
            "gen_ai.usage.output_tokens",
            usage.completion_tokens,
        )
    if response_model:
        span.set_attribute(
            "gen_ai.response.model", response_model
        )


def record_error(span, exception: BaseException) -> None:
    """Record an exception and set ERROR status on a span.

    Sets ``error.type`` per GenAI semantic conventions.
    No-ops when *span* is ``None`` (tracing disabled).
    """
    if span is None:
        return
    from opentelemetry.trace import StatusCode

    span.set_status(StatusCode.ERROR, str(exception))
    span.record_exception(exception)
    span.set_attribute(
        "error.type", type(exception).__qualname__
    )
