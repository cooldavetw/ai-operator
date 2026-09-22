from scripts.evaluate import summarize


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
