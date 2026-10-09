from __future__ import annotations

import json
from types import SimpleNamespace
import pytest

from terralens import assistant
from terralens.workflow import WorkflowStep


@pytest.fixture(autouse=True)
def isolate_credentials(monkeypatch):
    monkeypatch.setattr(assistant, "load_configuration", lambda: None)
    for name in ("LLM_PROVIDER", "GEMINI_MODEL", "OPENAI_MODEL", "XAI_MODEL", "GEMINI_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class FakeCompletionClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        name = kwargs["tools"][0]["function"]["name"]
        arguments = self.payloads.pop(0)
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
        message = SimpleNamespace(tool_calls=[call])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice])


def test_provider_defaults_to_free_tier_gemini_and_keeps_keys_server_side(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "local-test-key")
    status = assistant.status()
    assert status["configured"] is True
    assert status["provider"] == "gemini"
    assert status["model"] == "gemini-3.7-flash"
    assert "GEMINI_API_KEY" not in json.dumps(status)
    assert "free-tier" in status["privacy_note"]


def test_gemini_function_call_is_validated_then_semantically_reviewed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "local-test-key")
    step = {key: None for key in WorkflowStep.model_fields}
    step.update(id="inventory", operation="coverage", label="Inspect available datasets", dataset_id="spacenet")
    workflow = {"title": "Inspect available evidence", "status": "ready",
        "requirements": [{"text": "List available evidence", "status": "planned", "step_ids": ["inventory"]}],
        "missing_evidence": [], "steps": [step]}
    client = FakeCompletionClient([workflow, {"valid": True, "issues": []}])
    catalog = {"spacenet": {"manifest": {"area_name": "SpaceNet Rio sample", "vector_layers": {}, "coverage": {}}}}
    parsed, audit = assistant.plan("What data can I search?", "spacenet", [], catalog, client=client)
    assert parsed.steps[0].operation == "coverage"
    assert audit["provider"] == "gemini"
    assert audit["semantic_review"] == "model_checked"
    assert [call["tools"][0]["function"]["name"] for call in client.calls] == ["propose_workflow", "review_workflow"]
    assert all(call["tool_choice"] == "required" for call in client.calls)
    assert all("parallel_tool_calls" not in call for call in client.calls)
    assert all("strict" not in call["tools"][0]["function"] for call in client.calls)
    assert "$ref" not in json.dumps(client.calls[0]["tools"][0]["function"]["parameters"])


def test_provider_schema_adapter_preserves_nullability_and_local_validation():
    schema = assistant._gemini_schema(assistant.strict_schema())
    step = schema["properties"]["steps"]["items"]
    assert step["properties"]["plan"]["nullable"] is True
    assert step["properties"]["operation"]["enum"] == ["search", "spatial_filter", "aggregate", "coverage"]
    assert set(step["required"]) == set(WorkflowStep.model_fields)


def test_openai_uses_reasoning_model_completion_limit():
    client = FakeCompletionClient([{"valid": True, "issues": []}])
    assistant._call(client, "gpt-5-mini", "review", assistant.REVIEW_SCHEMA, [], "review", "openai")
    assert "max_tokens" not in client.calls[0]
    assert client.calls[0]["max_completion_tokens"] == 6000
    assert client.calls[0]["tools"][0]["function"]["strict"] is True


def test_grok_and_openai_can_be_selected_without_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("XAI_API_KEY", "local-test-key")
    assert assistant.status()["model"] == "grok-4.7"
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-key")
    assert assistant.status()["provider"] == "openai"
    assert assistant.status()["model"] == "gpt-5-mini"
