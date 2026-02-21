from lemurian.tools import tool


def test_tool_registry_maps_names(make_agent, sample_tool):
    agent = make_agent(tools=[sample_tool])
    assert "greet" in agent.tool_registry
    assert agent.tool_registry["greet"] is sample_tool


def test_model_copy_is_independent(make_agent, sample_tool):
    """Swarm uses model_copy to augment agents. Copy must be independent."""
    original = make_agent(tools=[sample_tool])
    copy = original.model_copy()

    @tool
    def extra():
        """Extra tool."""
        return "extra"

    copy.tools = list(copy.tools) + [extra]

    assert len(original.tools) == 1
    assert len(copy.tools) == 2
