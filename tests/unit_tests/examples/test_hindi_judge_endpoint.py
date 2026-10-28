"""Check Azure judge endpoint routing without calling external services."""

import importlib.util
import io
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
SCRIPTS = Path(__file__).resolve().parents[3] / "examples" / "models" / "speech_cascades"


@pytest.fixture
def judge_module(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "hindi_judge_endpoint", SCRIPTS / "judge_voiceagentbench_hindi_tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "endpoint,url,v1",
    [
        (
            "https://example.cognitiveservices.azure.com/",
            "https://example.cognitiveservices.azure.com/openai/deployments/gpt-5.2/chat/completions"
            "?api-version=2025-01-01-preview",
            False,
        ),
        (
            "https://example.services.ai.azure.com/openai/v1/",
            "https://example.services.ai.azure.com/openai/v1/chat/completions",
            True,
        ),
    ],
)
def test_endpoint_routes_and_preserves_judge_payload(judge_module, monkeypatch, endpoint, url, v1):
    requests = []
    monkeypatch.setattr(judge_module, "azure_token", lambda: "test-token")

    def respond(request, timeout):
        requests.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "gpt-5.2-2025-12-11",
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"Score":1}'}}],
                }
            ).encode()
        )

    monkeypatch.setattr(judge_module.urllib.request, "urlopen", respond)
    judge = judge_module.AzureJudge(endpoint)
    assert judge.complete("Exact upstream rubric").text == '{"Score":1}'
    assert requests[0].full_url == url
    payload = json.loads(requests[0].data)
    assert payload["messages"] == [{"role": "user", "content": "Exact upstream rubric"}]
    assert payload["max_completion_tokens"] == 2048
    assert payload.get("model") == ("gpt-5.2" if v1 else None)
    assert judge.responses[0]["judge_model"] == "gpt-5.2-2025-12-11"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.services.ai.azure.com/openai/v1",
        "https://example.com/openai/v1",
        "https://example.services.ai.azure.com/openai/v1/responses",
        "https://example.services.ai.azure.com/openai/v1?token=private",
        "https://user:password@example.services.ai.azure.com/openai/v1",
    ],
)
def test_endpoint_rejects_non_base_or_credential_urls(judge_module, endpoint):
    with pytest.raises(ValueError, match="Azure HTTPS"):
        judge_module.AzureJudge(endpoint)
