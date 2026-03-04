"""Unit tests for streaming primitives."""

from lemurian.streaming import (
    StreamChunk,
    ToolCall,
    ToolCallAccumulator,
    ToolCallFragment,
)


class TestToolCallAccumulator:
    def test_single_tool_call_in_one_fragment(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(
            index=0, call_id="call_1", name="greet",
            arguments_delta='{"name": "world"}',
        ))
        calls = acc.finalize()
        assert len(calls) == 1
        assert calls[0].id == "call_1"
        assert calls[0].name == "greet"
        assert calls[0].arguments == '{"name": "world"}'

    def test_arguments_accumulated_across_fragments(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(
            index=0, call_id="call_1", name="greet",
        ))
        acc.feed(ToolCallFragment(
            index=0, arguments_delta='{"na',
        ))
        acc.feed(ToolCallFragment(
            index=0, arguments_delta='me": "world"}',
        ))
        calls = acc.finalize()
        assert calls[0].arguments == '{"name": "world"}'

    def test_multiple_tool_calls_by_index(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(
            index=0, call_id="c1", name="foo",
            arguments_delta="{}",
        ))
        acc.feed(ToolCallFragment(
            index=1, call_id="c2", name="bar",
            arguments_delta="{}",
        ))
        calls = acc.finalize()
        assert len(calls) == 2
        assert calls[0].name == "foo"
        assert calls[1].name == "bar"

    def test_finalize_empty(self):
        acc = ToolCallAccumulator()
        assert acc.finalize() == []

    def test_stream_chunk_defaults(self):
        chunk = StreamChunk()
        assert chunk.content_delta is None
        assert chunk.tool_call_fragments is None
        assert chunk.finish_reason is None
        assert chunk.usage is None
        assert chunk.response_model is None

    def test_tool_call_defaults(self):
        tc = ToolCall()
        assert tc.id == ""
        assert tc.name == ""
        assert tc.arguments == ""
