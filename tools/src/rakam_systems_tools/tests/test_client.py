from rakam_systems_tools.evaluation.schema import EvalConfig, ToxicityConfig, TextInputItem
from types import SimpleNamespace
from typing import Dict, Tuple
from unittest.mock import Mock

import pytest
import requests

from rakam_systems_tools.evaluation.client import DeepEvalClient


@pytest.fixture
def fake_response() -> Mock:
    resp = Mock()
    resp.json.return_value = {"ok": True}
    resp.raise_for_status.return_value = None
    resp.text = "raw"
    return resp


@pytest.fixture
def mock_request(monkeypatch: pytest.MonkeyPatch, fake_response: Mock) -> Mock:
    mock = Mock(return_value=fake_response)
    monkeypatch.setattr(requests, "request", mock)
    return mock


def test_init_uses_base_url_and_token_args() -> None:
    client = DeepEvalClient(
        base_url="http://example.com/",
        api_token="token123",
    )

    assert client.base_url == "http://example.com"
    assert client.api_token == "token123"
    assert client.timeout == 30


def test_init_uses_settings_module() -> None:
    settings = SimpleNamespace(
        EVALFRAMEWORK_URL="http://settings-url",
        EVALFRAMWORK_API_KEY="settings-token",
    )

    client = DeepEvalClient(settings_module=settings)

    assert client.base_url == "http://settings-url"
    assert client.api_token == "settings-token"


def test_init_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALFRAMEWORK_URL", "http://env-url")
    monkeypatch.setenv("EVALFRAMEWORK_API_KEY", "env-token")

    client = DeepEvalClient()

    assert client.base_url == "http://env-url"
    assert client.api_token == "env-token"


def test_init_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVALFRAMEWORK_URL", raising=False)

    client = DeepEvalClient()

    assert client.base_url == "http://localhost:8080"


def test_request_success_json(mock_request: Mock) -> None:
    client = DeepEvalClient(
        base_url="http://api",
        api_token="abc",
        timeout=10,
    )

    resp = client._request(
        "POST",
        "/test",
        json={"a": 1},
        params={"q": "x"},
    )

    assert resp == {"ok": True}

    mock_request.assert_called_once_with(
        method="POST",
        url="http://api/test",
        headers={
            "accept": "application/json",
            "X-API-Token": "abc",
            "Content-Type": "application/json",
        },
        json={"a": 1},
        params={"q": "x"},
        timeout=10,
    )


def test_request_exception_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Tuple, **__: Dict) -> None:
        raise requests.RequestException("network down")

    monkeypatch.setattr(requests, "request", boom)

    client = DeepEvalClient()

    resp = client._request("GET", "/x")
    assert resp is not None
    assert "error" in resp
    assert "network down" in resp["error"]


def test_request_exception_raises_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: Tuple, **__: Dict) -> None:
        raise requests.RequestException("fail")

    monkeypatch.setattr(requests, "request", boom)

    client = DeepEvalClient()

    with pytest.raises(requests.RequestException):
        client._request("GET", "/x", raise_exception=True)


def test_invalid_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = Mock()
    resp.json.side_effect = ValueError("no json")
    resp.text = "not-json"
    resp.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "request", Mock(return_value=resp))

    client = DeepEvalClient()

    out = client._request("GET", "/bad")

    assert out == {
        "error": "Invalid JSON response",
        "raw": "not-json",
    }


@pytest.mark.parametrize(
    "method,call",
    [
        ("GET", "_get"),
        ("POST", "_post"),
        ("PATCH", "_patch"),
        ("DELETE", "_delete"),
    ],
)
def test_http_helpers_delegate_correctly(
    method: str,
    call: str,
    monkeypatch: pytest.MonkeyPatch,
    fake_response: Mock,
) -> None:
    mock = Mock(return_value=fake_response)
    monkeypatch.setattr(requests, "request", mock)

    client = DeepEvalClient(base_url="http://api")

    if method == "GET":
        getattr(client, call)("/x", params={"a": 1})
    else:
        getattr(client, call)("/x", payload={"a": 1})

    args, kwargs = mock.call_args
    assert kwargs["method"] == method
    assert kwargs["url"] == "http://api/x"


def test_update_tag_calls_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    patch_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_patch", patch_mock)

    resp = client.update_evaluation_testcase_tag(
        testcase_id=42,
        tag="smoke",
    )

    assert resp == {"ok": True}

    patch_mock.assert_called_once_with(
        endpoint="/deepeval/42/tag",
        payload={"tag": "smoke"},
        raise_exception=False,
    )


def test_update_tag_patch_with_none_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    patch_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_patch", patch_mock)

    client.update_evaluation_testcase_tag(
        testcase_id=10,
        tag=None,
    )

    patch_mock.assert_called_once_with(
        endpoint="/deepeval/10/tag",
        payload={"tag": None},
        raise_exception=False,
    )


def test_update_tag_calls_delete_when_testcase_id_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DeepEvalClient()

    delete_mock = Mock(return_value={"deleted": True})
    monkeypatch.setattr(client, "_delete", delete_mock)

    resp = client.update_evaluation_testcase_tag(
        testcase_id=None,  # type: ignore[arg-type]
        tag="smoke",
    )

    assert resp == {"deleted": True}

    delete_mock.assert_called_once_with(
        endpoint="/deepeval/tag/smoke",
        payload={},
        raise_exception=False,
    )


def test_update_tag_propagates_raise_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    patch_mock = Mock()
    monkeypatch.setattr(client, "_patch", patch_mock)

    client.update_evaluation_testcase_tag(
        testcase_id=99,
        tag="critical",
        raise_exception=True,
    )

    patch_mock.assert_called_once_with(
        endpoint="/deepeval/99/tag",
        payload={"tag": "critical"},
        raise_exception=True,
    )


def test_list_evaluation_testcases_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    get_mock = Mock(return_value={"items": [], "total": 0})
    monkeypatch.setattr(client, "_get", get_mock)

    resp = client.list_evaluation_testcases()

    assert resp == {"items": [], "total": 0}

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/token",
        params={
            "limit": 10,
            "offset": 0,
        },
        raise_exception=False,
    )


def test_list_evaluation_testcases_with_custom_pagination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DeepEvalClient()

    get_mock = Mock(return_value={"items": ["a", "b"], "total": 2})
    monkeypatch.setattr(client, "_get", get_mock)

    resp = client.list_evaluation_testcases(
        limit=25,
        offset=50,
    )
    assert resp is not None
    assert resp["total"] == 2

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/token",
        params={
            "limit": 25,
            "offset": 50,
        },
        raise_exception=False,
    )


def test_list_evaluation_testcases_propagates_raise_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DeepEvalClient()

    get_mock = Mock()
    monkeypatch.setattr(client, "_get", get_mock)

    client.list_evaluation_testcases(
        raise_exception=True,
    )

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/token",
        params={
            "limit": 10,
            "offset": 0,
        },
        raise_exception=True,
    )


def test_compare_testcases_with_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    get_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_get", get_mock)

    resp = client.compare_testcases(
        testcase_a_id=1,
        testcase_b_id=2,
    )

    assert resp == {"ok": True}

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/compare",
        params={
            "testcase_a_id": 1,
            "testcase_b_id": 2,
        },
        raise_exception=False,
    )


def test_compare_testcases_with_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    get_mock = Mock(return_value={"diff": []})
    monkeypatch.setattr(client, "_get", get_mock)

    resp = client.compare_testcases(
        testcase_a_tag="baseline",
        testcase_b_tag="candidate",
    )

    assert resp == {"diff": []}

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/compare",
        params={
            "testcase_a_tag": "baseline",
            "testcase_b_tag": "candidate",
        },
        raise_exception=False,
    )


def test_compare_testcases_id_vs_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    get_mock = Mock(return_value={"result": "ok"})
    monkeypatch.setattr(client, "_get", get_mock)

    client.compare_testcases(
        testcase_a_id=10,
        testcase_b_tag="latest",
        raise_exception=True,
    )

    get_mock.assert_called_once_with(
        endpoint="/eval-framework/deepeval/evaluation-testcases/compare",
        params={
            "testcase_a_id": 10,
            "testcase_b_tag": "latest",
        },
        raise_exception=True,
    )


def test_compare_testcases_missing_identifier_a() -> None:
    client = DeepEvalClient()

    with pytest.raises(ValueError, match="testcase_a"):
        client.compare_testcases(
            testcase_b_id=2,
        )


def test_compare_testcases_both_id_and_tag_a() -> None:
    client = DeepEvalClient()

    with pytest.raises(ValueError, match="testcase_a"):
        client.compare_testcases(
            testcase_a_id=1,
            testcase_a_tag="oops",
            testcase_b_id=2,
        )


def test_compare_testcases_missing_identifier_b() -> None:
    client = DeepEvalClient()

    with pytest.raises(ValueError, match="testcase_b"):
        client.compare_testcases(
            testcase_a_id=1,
        )


@pytest.mark.parametrize("chance", [0, 0.5, 1])
def test_validate_chance_valid(chance: float) -> None:
    DeepEvalClient._validate_chance(chance)


@pytest.mark.parametrize("chance", [-0.1, 1.01, 2, -1])
def test_validate_chance_invalid(chance: float) -> None:
    with pytest.raises(ValueError, match="chance must be between 0 and 1"):
        DeepEvalClient._validate_chance(chance)


def test_text_eval_with_explicit_config(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    config = EvalConfig(
        data=[TextInputItem(input="hello", output="world")],
        metrics=[ToxicityConfig()],
        component="test",
        label="run-1",
    )

    post_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post", post_mock)

    resp = client.text_eval(config)

    assert resp == {"ok": True}

    post_mock.assert_called_once_with(
        endpoint="/deepeval/text-eval",
        payload=config.model_dump(),
        raise_exception=False,
    )


def test_text_eval_builds_config_from_args(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    data = [TextInputItem(input="foo", output="bar")]
    metrics = [ToxicityConfig(name="bleu-1")]

    post_mock = Mock(return_value={"result": "created"})
    monkeypatch.setattr(client, "_post", post_mock)

    resp = client.text_eval(
        data=data,
        metrics=metrics,
        component="api",
        label="experiment-42",
    )

    assert resp == {"result": "created"}

    called_payload = post_mock.call_args.kwargs["payload"]

    assert called_payload["data"] == [item.model_dump() for item in data]
    assert called_payload["metrics"] == [m.model_dump() for m in metrics]
    assert called_payload["component"] == "api"
    assert called_payload["label"] == "experiment-42"


def test_text_eval_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    post_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post", post_mock)

    client.text_eval(
        data=[],
        metrics=[],
    )

    payload = post_mock.call_args.kwargs["payload"]

    assert payload["component"] == "unknown"
    assert payload["label"] is None


def test_text_eval_raise_exception_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    post_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post", post_mock)

    client.text_eval(
        data=[],
        metrics=[],
        raise_exception=True,
    )

    post_mock.assert_called_once()
    assert post_mock.call_args.kwargs["raise_exception"] is True


def test_text_eval_config_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DeepEvalClient()

    config = EvalConfig(
        data=[],
        metrics=[],
        component="from-config",
        label="config-label",
    )

    post_mock = Mock(return_value={"ok": True})
    monkeypatch.setattr(client, "_post", post_mock)

    client.text_eval(  # type: ignore[call-overload]
        config,
        data=[TextInputItem(input="ignored", output="ignored")],
        metrics=[ToxicityConfig()],
        component="ignored",
        label="ignored",
    )

    payload = post_mock.call_args.kwargs["payload"]

    assert payload["component"] == "from-config"
    assert payload["label"] == "config-label"
