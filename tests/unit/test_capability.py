from lemurian.tools import tool

from tests.conftest import MockCapability


# ---------------------------------------------------------------------------
# Agent.tool_registry with capabilities
# ---------------------------------------------------------------------------

class TestAgentToolRegistryWithCapabilities:
    def test_registry_includes_capability_tools(
        self, make_agent, cap_tool_raven
    ):
        cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        agent = make_agent(capabilities=[cap])
        assert "raven" in agent.tool_registry

    def test_registry_merges_tools_and_capabilities(
        self, make_agent, sample_tool, cap_tool_raven
    ):
        cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        agent = make_agent(tools=[sample_tool], capabilities=[cap])
        registry = agent.tool_registry
        assert "greet" in registry
        assert "raven" in registry

    def test_registry_with_multiple_capabilities(
        self, make_agent, cap_tool_raven, cap_tool_annabel
    ):
        raven_cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        annabel_cap = MockCapability(name="annabel_lee", tool_list=[cap_tool_annabel])
        agent = make_agent(capabilities=[raven_cap, annabel_cap])
        registry = agent.tool_registry
        assert "raven" in registry
        assert "annabel" in registry

    def test_registry_with_capabilities_only(
        self, make_agent, cap_tool_raven
    ):
        cap = MockCapability(name="raven", tool_list=[cap_tool_raven])
        agent = make_agent(tools=[], capabilities=[cap])
        assert list(agent.tool_registry.keys()) == ["raven"]

    def test_capability_tool_shadows_agent_tool(
        self, make_agent, sample_tool
    ):
        """Agent-level tool_registry does not validate duplicates.
        A capability tool with the same name silently overwrites."""
        @tool
        def greet():
            """Quoth the Raven."""
            return "Quoth the Raven, 'Nevermore.'"
        cap = MockCapability(name="raven", tool_list=[greet])
        agent = make_agent(tools=[sample_tool], capabilities=[cap])
        assert agent.tool_registry["greet"] is not sample_tool
