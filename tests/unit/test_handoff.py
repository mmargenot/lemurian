from lemurian.handoff import (
    Handoff,
    HandoffResult,
    _normalize_tool_name,
    handoff,
)


class TestNormalizeToolName:
    def test_lowercase(self):
        assert _normalize_tool_name("Billing") == "billing"

    def test_spaces_to_underscores(self):
        assert (
            _normalize_tool_name("Billing Support")
            == "billing_support"
        )

    def test_hyphens_to_underscores(self):
        assert (
            _normalize_tool_name("billing-support")
            == "billing_support"
        )

    def test_strips_special_chars(self):
        assert (
            _normalize_tool_name("billing@support!")
            == "billingsupport"
        )

    def test_mixed_case_spaces_and_special(self):
        assert (
            _normalize_tool_name("  My Agent-Name! ")
            == "my_agent_name"
        )

    def test_already_normalized(self):
        assert _normalize_tool_name("billing") == "billing"

    def test_multiple_spaces_collapse(self):
        assert (
            _normalize_tool_name("billing   support")
            == "billing_support"
        )


class TestHandoffFactory:
    def test_basic_creation(self):
        h = handoff("billing", "Handles billing")

        assert h.tool_name == "transfer_to_billing"
        assert h.target_agent == "billing"
        assert "billing" in h.tool_description
        assert "Handles billing" in h.tool_description

    def test_normalizes_name(self):
        h = handoff("Billing Support", "Bills customers")

        assert h.tool_name == "transfer_to_billing_support"
        assert h.target_agent == "Billing Support"

    def test_empty_description(self):
        h = handoff("billing")

        assert h.tool_description == "Hand off to billing."
        assert "billing" in h.tool_description

    def test_schema_has_message_param(self):
        h = handoff("billing", "Bills")

        schema = h.input_json_schema
        assert "message" in schema["properties"]
        assert "message" in schema["required"]


class TestHandoffToolSchema:
    def test_openai_format(self):
        h = handoff("billing", "Bills customers")
        schema = h.tool_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "transfer_to_billing"
        assert "Bills customers" in schema["function"]["description"]
        assert "properties" in schema["function"]["parameters"]

    def test_custom_handoff_schema(self):
        h = Handoff(
            tool_name="route_to_vip",
            tool_description="Route to VIP support",
            target_agent="vip",
            input_json_schema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Context",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "Priority level",
                    },
                },
                "required": ["message", "priority"],
            },
        )
        schema = h.tool_schema()

        assert schema["function"]["name"] == "route_to_vip"
        params = schema["function"]["parameters"]
        assert "priority" in params["properties"]


class TestHandoffResult:
    def test_basic_creation(self):
        r = HandoffResult(
            target_agent="billing", message="help"
        )
        assert r.target_agent == "billing"
        assert r.message == "help"
