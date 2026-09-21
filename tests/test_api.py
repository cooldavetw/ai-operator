import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi.testclient import TestClient

from app.errors import ServiceError
from app.main import create_app
from app.schemas import Decision
from tests.conftest import FakeRuntime, KEY, clarify, matched


def payload(message="我登不進去", previous=None, **kwargs):
    value = {"request_id": str(uuid4()), "message": message}
    if previous:
        value.update(conversation_id=previous["conversation_id"], expected_version=previous["version"])
    value.update(kwargs)
    return value


def post(client, message="我登不進去", previous=None):
    response = client.post("/api/v1/classify", json=payload(message, previous))
    assert response.status_code == 200, response.text
    return response.json()


def test_exact_rule_classifies_without_model(client, runtime):
    result = post(client, "我要續借書")
    assert result["status"] == "CLASSIFIED"
    assert result["domain"] == "LIBRARY_SYSTEM"
    assert result["reason_code"] == "RULE_MATCH"
    assert result["is_final"] and result["clarification"] is None
    assert runtime.calls == []


def test_two_clarifications_then_unknown(client, runtime):
    first = post(client)
    second = post(client, "不知道", first)
    final = post(client, "還是不知道", second)
    assert first["clarification_count"] == 1
    assert second["clarification_count"] == 2
    assert first["clarification"]["id"] != second["clarification"]["id"]
    assert final["status"] == "UNKNOWN" and final["domain"] == "UNKNOWN"
    assert final["reason_code"] == "CLARIFICATION_LIMIT"
    assert final["clarification_count"] == 2 and final["is_final"]
    assert final["clarification"] is None
    assert len(runtime.calls[-1][0]) == 5


def test_second_clarification_answer_can_succeed(client, runtime):
    runtime.decisions = [clarify(), clarify(), matched()]
    first = post(client)
    second = post(client, "不清楚", first)
    final = post(client, "我要續借書", second)
    assert final["status"] == "CLASSIFIED"
    assert final["clarification_count"] == 2
    assert len(runtime.calls) == 3  # Follow-up exact matches cannot ignore history.


def test_out_of_scope_is_immediately_final(client, runtime):
    runtime.decisions = [Decision(decision="UNKNOWN", domain=None, clarification_id=None, reason="OUT_OF_SCOPE")]
    result = post(client, "停車證要去哪申請")
    assert result["status"] == "UNKNOWN"
    assert result["clarification_count"] == 0
    assert result["reason_code"] == "OUT_OF_SCOPE"


def test_multiple_domains_requires_priority(client, runtime):
    runtime.decisions = [clarify("choose_priority", "MULTIPLE_DOMAINS")]
    result = post(client, "選課失敗，而且書不能續借")
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["reason_code"] == "MULTIPLE_DOMAINS"
    assert result["clarification"]["id"] == "choose_priority"


def test_negated_keyword_does_not_trigger_rule(client, runtime):
    post(client, "不是我要續借書，是登入不了")
    assert len(runtime.calls) == 1


def test_replay_is_identical_and_does_not_infer_twice(client, runtime):
    request = payload()
    first = client.post("/api/v1/classify", json=request)
    retry = client.post("/api/v1/classify", json=request)
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json()
    assert len(runtime.calls) == 1
    changed = client.post("/api/v1/classify", json={**request, "message": "其他問題"})
    assert changed.status_code == 409
    assert changed.json()["error"]["code"] == "REQUEST_ID_REUSED"


def test_replay_after_later_turn_returns_original(client):
    request = payload()
    original = client.post("/api/v1/classify", json=request).json()
    post(client, "不清楚", original)
    assert client.post("/api/v1/classify", json=request).json() == original


def test_conflicting_updates_are_serialized(client):
    first = post(client)
    def send():
        return client.post("/api/v1/classify", json=payload("不知道", first)).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: send(), range(2))) == [200, 409]


def test_finished_and_missing_conversations(client):
    result = post(client, "我要續借書")
    response = client.post("/api/v1/classify", json=payload("新問題", result))
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONVERSATION_FINISHED"
    response = client.post("/api/v1/classify", json=payload(conversation_id=str(uuid4()), expected_version=1))
    assert response.status_code == 404


def test_failure_preserves_version_and_retry_id(client, runtime):
    first = post(client)
    request = payload("圖書館那個", first)
    runtime.decisions = [ServiceError(504, "INFERENCE_TIMEOUT", "timeout"), matched()]
    failed = client.post("/api/v1/classify", json=request)
    assert failed.status_code == 504
    retry = client.post("/api/v1/classify", json=request)
    assert retry.status_code == 200
    assert retry.json()["version"] == 2
    assert retry.json()["clarification_count"] == 1


def test_invalid_model_output_is_not_unknown(client, runtime):
    runtime.decisions = [matched("INVENTED_SYSTEM")]
    result = client.post("/api/v1/classify", json=payload())
    assert result.status_code == 502
    assert result.json()["error"]["code"] == "INVALID_MODEL_OUTPUT"


def test_auth_validation_and_body_limit(client):
    assert client.get("/health/live", headers={"X-API-Key": "wrong"}).status_code == 200
    assert client.get("/health/ready").status_code == 200
    assert client.get("/openapi.json", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/v1/classify", json=payload(), headers={"X-API-Key": "wrong"}).status_code == 401
    for message in [" ", "敏感資料" * 1000]:
        result = client.post("/api/v1/classify", json=payload(message))
        assert result.status_code == 422
        assert "敏感資料" not in result.text
    assert client.post("/api/v1/classify", json=payload(expected_version=1)).status_code == 422
    assert client.post("/api/v1/classify", content=b"x" * 17000).status_code == 413
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"


def test_streaming_body_limit(client):
    response = client.post("/api/v1/classify", content=iter([b"x" * 9000, b"y" * 9000]))
    assert response.status_code == 413


def test_restart_preserves_replay_and_config_snapshot(settings):
    request = payload()
    with TestClient(create_app(settings, FakeRuntime()), headers={"X-API-Key": KEY}) as first_client:
        original = first_client.post("/api/v1/classify", json=request).json()
    content = settings.config_path.read_text()
    settings.config_path.write_text(content.replace('version: "1"', 'version: "2"').replace("您原本想辦理什麼事情", "更新後的問句"))
    runtime = FakeRuntime()
    with TestClient(create_app(settings, runtime), headers={"X-API-Key": KEY}) as second_client:
        assert second_client.post("/api/v1/classify", json=request).json() == original
        continued = post(second_client, "不知道", original)
        assert continued["config_version"] == "1"
        assert "更新後的問句" not in continued["clarification"]["question"]
        assert post(second_client)["config_version"] == "2"


def test_expired_conversation_and_replay(client, settings):
    request = payload()
    original = client.post("/api/v1/classify", json=request).json()
    with sqlite3.connect(settings.database_path) as db:
        db.execute("UPDATE conversations SET expires_at=?", (time.time() - 1,))
        db.execute("UPDATE requests SET expires_at=?", (time.time() - 1,))
    assert client.post("/api/v1/classify", json=request).status_code == 410
    assert client.post("/api/v1/classify", json=payload("不知道", original)).status_code == 410


def test_purge_removes_state_and_replay(client, settings):
    post(client)
    with sqlite3.connect(settings.database_path) as db:
        db.execute("UPDATE conversations SET expires_at=0")
    post(client)
    with sqlite3.connect(settings.database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1


def test_logs_do_not_contain_question(client, caplog):
    secret = "這是不可出現在日誌的學生個資"
    with caplog.at_level("INFO", logger="classifier"):
        post(client, secret)
    assert secret not in caplog.text
    assert "classification request_id=" in caplog.text


def test_storage_failure_rolls_back_state_and_replay(client, settings):
    first = post(client)
    request = payload("不知道", first)
    with sqlite3.connect(settings.database_path) as db:
        db.execute("CREATE TRIGGER fail_request BEFORE INSERT ON requests BEGIN SELECT RAISE(ABORT, 'test'); END")
    assert client.post("/api/v1/classify", json=request).status_code == 503
    with sqlite3.connect(settings.database_path) as db:
        assert db.execute("SELECT version FROM conversations").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
        db.execute("DROP TRIGGER fail_request")
    assert client.post("/api/v1/classify", json=request).json()["version"] == 2


def test_expiry_during_inference_is_not_committed(client, runtime, settings, monkeypatch):
    first = post(client)
    original_infer = runtime.infer
    def expire(history, config):
        result = original_infer(history, config)
        monkeypatch.setattr("app.service.time.time", lambda: 9999999999)
        return result
    runtime.infer = expire
    assert client.post("/api/v1/classify", json=payload("不知道", first)).status_code == 410
    with sqlite3.connect(settings.database_path) as db:
        assert db.execute("SELECT version FROM conversations").fetchone()[0] == 1


def test_missing_model_keeps_liveness_but_fails_readiness(settings):
    class UnavailableRuntime(FakeRuntime):
        def start(self):
            raise ServiceError(503, "MODEL_NOT_READY", "missing")

        def infer(self, history, config):
            self.start()

    with TestClient(create_app(settings, UnavailableRuntime()), headers={"X-API-Key": KEY}) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/api/v1/classify", json=payload()).status_code == 503

