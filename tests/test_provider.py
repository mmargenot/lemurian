from lemurian.provider import ModalVLLMProvider, OpenAIProvider


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
