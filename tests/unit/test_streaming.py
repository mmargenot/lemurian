"""Unit tests for streaming primitives."""

from lemurian.streaming import ToolCall, ToolCallAccumulator, ToolCallFragment


class TestToolCallAccumulator:
    def test_single_tool_call_single_fragment(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(index=0, call_id="c1", name="echo", arguments_delta='{"text": "hi"}'))
        result = acc.finalize()

        assert len(result) == 1
        assert result[0] == ToolCall(id="c1", name="echo", arguments='{"text": "hi"}')

    def test_arguments_accumulated_across_fragments(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(index=0, call_id="c1", name="echo", arguments_delta='{"te'))
        acc.feed(ToolCallFragment(index=0, arguments_delta='xt": "hi"}'))
        result = acc.finalize()

        assert result[0].arguments == '{"text": "hi"}'

    def test_multiple_concurrent_tool_calls(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(index=0, call_id="c1", name="foo", arguments_delta='{"a":'))
        acc.feed(ToolCallFragment(index=1, call_id="c2", name="bar", arguments_delta='{"b":'))
        acc.feed(ToolCallFragment(index=0, arguments_delta=' 1}'))
        acc.feed(ToolCallFragment(index=1, arguments_delta=' 2}'))
        result = acc.finalize()

        assert len(result) == 2
        assert result[0] == ToolCall(id="c1", name="foo", arguments='{"a": 1}')
        assert result[1] == ToolCall(id="c2", name="bar", arguments='{"b": 2}')

    def test_finalize_returns_index_order(self):
        acc = ToolCallAccumulator()
        acc.feed(ToolCallFragment(index=2, call_id="c3", name="c"))
        acc.feed(ToolCallFragment(index=0, call_id="c1", name="a"))
        acc.feed(ToolCallFragment(index=1, call_id="c2", name="b"))
        result = acc.finalize()

        assert [tc.name for tc in result] == ["a", "b", "c"]

    def test_empty_accumulator(self):
        acc = ToolCallAccumulator()
        assert acc.finalize() == []
