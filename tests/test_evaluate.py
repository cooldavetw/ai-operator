import io
import json
import urllib.error
from pathlib import Path

import pytest

from scripts.evaluate import evaluate, normalize_case, summarize


def test_metrics_do_not_hide_technical_errors_as_unknown():
    rows = [
        {"expected": "UNKNOWN", "predicted": "LIBRARY_SYSTEM", "clarifications": 0, "error": None},
        {"expected": "LIBRARY_SYSTEM", "predicted": "LIBRARY_SYSTEM", "clarifications": 1, "error": None},
        {"expected": "ACADEMIC_SYSTEM", "predicted": "ERROR", "clarifications": 0, "error": "HTTP_503"},
        {"expected": "ACADEMIC_SYSTEM", "predicted": "UNRESOLVED", "clarifications": 2, "error": None},
    ]
    report = summarize(rows, [10, 20, 30, 40])
    assert report["auto_classification_coverage"] == 0.5
    assert report["auto_classification_accuracy"] == 0.5
    assert report["out_of_scope_misroute_rate"] == 1
    assert report["technical_errors"] == 1
    assert report["per_domain"]["ACADEMIC_SYSTEM"]["recall"] == 0
    assert report["latency_ms"] == {"p50": 20, "p95": 40}



def test_out_of_scope_clarification_is_reported_as_unresolved():
    rows = [
        {"expected": "UNKNOWN", "predicted": "UNRESOLVED", "clarifications": 1, "error": None},
        {"expected": "UNKNOWN", "predicted": "UNKNOWN", "clarifications": 0, "error": None},
    ]
    report = summarize(rows, [])
    assert report["out_of_scope_misroute_rate"] == 0
    assert report["out_of_scope_unresolved_rate"] == 0.5
    assert report["per_domain"]["UNKNOWN"]["recall"] == 0.5




def dialogue():
    return {"id": "clarify", "turns": [
        {"message": "哪個系統", "expected": {"status": "NEEDS_CLARIFICATION", "is_final": False}},
        {"message": "圖書館", "expected": {"status": "CLASSIFIED", "domain": "LIBRARY_SYSTEM", "is_final": True}},
    ]}


def reply(final=False, domain=None, version=1):
    return {"conversation_id": "conversation", "version": version,
            "status": "CLASSIFIED" if final else "NEEDS_CLARIFICATION",
            "domain": domain, "is_final": final, "clarification_count": 1,
            "reason_code": "SEMANTIC_MATCH" if final else "INSUFFICIENT_INFORMATION"}


def transport(monkeypatch, responses):
    requests = []
    values = iter(responses)
    def send(request, timeout):
        requests.append(json.loads(request.data))
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(json.dumps(value).encode())
    monkeypatch.setattr("urllib.request.urlopen", send)
    return requests


def test_dialogue_passes_and_propagates_version(monkeypatch):
    sent = transport(monkeypatch, [reply(), reply(True, "LIBRARY_SYSTEM", 2)])
    report = evaluate("http://localhost", "secret", [dialogue()], 1)
    assert report["dialogue_pass_rate"] == report["turn_pass_rate"] == 1
    assert sent[0]["expected_version"] == 0
    assert sent[1]["expected_version"] == 1
    assert sent[1]["conversation_id"] == "conversation"
    assert sent[0]["request_id"] != sent[1]["request_id"]


def test_premature_correct_domain_is_not_dialogue_success(monkeypatch):
    sent = transport(monkeypatch, [reply(True, "LIBRARY_SYSTEM")])
    report = evaluate("http://localhost", "secret", [dialogue()], 1)
    assert len(sent) == 1
    assert report["per_domain"]["LIBRARY_SYSTEM"]["recall"] == 1
    assert report["dialogue_pass_rate"] == report["turn_pass_rate"] == 0
    assert report["premature_completions"] == report["skipped_turns"] == 1
    assert report["results"][0]["turns"][0]["mismatches"]["is_final"] == {"expected": False, "actual": True}


def test_wrong_reason_fails_despite_correct_domain(monkeypatch):
    case = {"id": "limit", "turns": [{"message": "不知道", "expected": {
        "domain": "UNKNOWN", "reason_code": "CLARIFICATION_LIMIT", "clarification_count": 2}}]}
    value = reply(True, "UNKNOWN")
    value.update(status="UNKNOWN", reason_code="OUT_OF_SCOPE")
    transport(monkeypatch, [value])
    report = evaluate("http://localhost", "secret", [case], 1)
    assert report["dialogue_pass_rate"] == 0
    assert set(report["results"][0]["turns"][0]["mismatches"]) == {"reason_code", "clarification_count"}


@pytest.mark.parametrize("failure,code", [
    (urllib.error.HTTPError("http://localhost", 503, "unavailable", {}, None), "HTTP_503"),
    (TimeoutError(), "TRANSPORT_ERROR"),
    ({"unexpected": "response"}, "INVALID_RESPONSE"),
])
def test_error_and_skipped_turns_count_as_failures(monkeypatch, failure, code):
    transport(monkeypatch, [failure])
    report = evaluate("http://localhost", "secret", [dialogue()], 1)
    assert report["technical_errors"] == 1
    assert report["turn_pass_rate"] == 0
    assert report["skipped_turns"] == 1
    assert report["results"][0]["predicted"] == "ERROR"
    assert report["results"][0]["turns"][0]["error"] == code


def test_legacy_dataset_supported(monkeypatch):
    transport(monkeypatch, [reply(True, "LIBRARY_SYSTEM")])
    case = {"id": "legacy", "messages": ["續借"], "expected_domain": "LIBRARY_SYSTEM"}
    assert evaluate("http://localhost", "secret", [case], 1)["dialogue_pass_rate"] == 1


def test_sample_dataset_and_validation():
    cases = [json.loads(line) for line in Path("examples/evaluation.jsonl").read_text().splitlines()]
    assert len(cases) == 12
    for case in cases:
        normalize_case(case)
    case = dialogue()
    case["turns"][0]["expected"] = {"typo": True}
    with pytest.raises(ValueError, match="supported"):
        normalize_case(case)
