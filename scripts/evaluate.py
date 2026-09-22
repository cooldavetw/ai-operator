"""Evaluate held-out JSONL dialogues against the deployed API; no external deps."""
import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from uuid import uuid4


EXPECTED_FIELDS = {"status", "domain", "is_final", "reason_code", "clarification_count"}


def normalize_case(case):
    """Accept explicit turn expectations and legacy final-domain datasets."""
    if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
        raise ValueError("Each case needs a nonempty id")
    domain = case.get("expected_domain")
    if "turns" in case:
        if "messages" in case:
            raise ValueError("Use turns or messages, not both")
        turns = case["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError("turns must be a nonempty list")
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("message"), str) or not turn["message"].strip():
                raise ValueError("Each turn needs a nonempty message")
            expected = turn.get("expected")
            if not isinstance(expected, dict) or not expected or expected.keys() - EXPECTED_FIELDS:
                raise ValueError("Each turn needs supported expected response fields")
            for key, value in expected.items():
                valid = (type(value) is bool if key == "is_final" else
                         type(value) is int and value >= 0 if key == "clarification_count" else
                         value is None or isinstance(value, str) if key == "domain" else
                         isinstance(value, str))
                if not valid:
                    raise ValueError(f"Invalid expected {key}")
        final_domain = turns[-1]["expected"].get("domain")
        if domain is None:
            domain = final_domain
        elif final_domain is not None and final_domain != domain:
            raise ValueError("expected_domain conflicts with final turn")
    else:
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or any(not isinstance(m, str) or not m.strip() for m in messages):
            raise ValueError("Each legacy case needs nonempty messages")
        turns = [{"message": m, "expected": {}} for m in messages]
        turns[-1]["expected"] = {"domain": domain, "is_final": True}
    if not isinstance(domain, str) or not domain:
        raise ValueError("Supply expected_domain or a final expected domain")
    return {"id": case["id"], "expected_domain": domain, "turns": turns}


def evaluate(base_url, api_key, cases, timeout):
    # Validate the whole dataset before sending any requests.
    cases = [normalize_case(case) for case in cases]
    outcomes, latencies = [], []
    for case in cases:
        conversation, version, last, error = None, 0, None, None
        turn_results, stopped = [], None
        premature = False
        for index, turn in enumerate(case["turns"], 1):
            expected = turn["expected"]
            record = {"turn": index, "expected": expected, "actual": None,
                      "passed": False, "mismatches": {}, "error": None}
            turn_results.append(record)
            if stopped:
                record["skipped"] = stopped
                continue
            body = {"request_id": str(uuid4()), "conversation_id": conversation,
                    "expected_version": version, "message": turn["message"]}
            request = urllib.request.Request(
                base_url.rstrip("/") + "/api/v1/classify",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "X-API-Key": api_key},
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    actual = json.load(response)
                required = {"conversation_id", "version", "status", "domain", "is_final", "clarification_count", "reason_code"}
                if not isinstance(actual, dict) or not required <= actual.keys():
                    raise ValueError("Missing response fields")
                if (type(actual["is_final"]) is not bool or type(actual["version"]) is not int
                        or type(actual["clarification_count"]) is not int
                        or actual["status"] not in {"CLASSIFIED", "UNKNOWN", "NEEDS_CLARIFICATION"}
                        or not isinstance(actual["conversation_id"], str)
                        or not isinstance(actual["reason_code"], str)
                        or not (actual["domain"] is None or isinstance(actual["domain"], str))):
                    raise ValueError("Invalid response fields")
                last = actual
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                error = (f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else
                         "INVALID_RESPONSE" if isinstance(exc, ValueError) else "TRANSPORT_ERROR")
                record["error"] = error
                stopped = "TECHNICAL_ERROR"
                continue
            finally:
                latencies.append((time.perf_counter() - started) * 1000)
            record["actual"] = actual
            record["mismatches"] = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected.items() if actual.get(key) != value
            }
            premature = actual["is_final"] and index < len(case["turns"])
            record["premature_completion"] = premature
            record["passed"] = not record["mismatches"] and not premature
            conversation, version = actual["conversation_id"], actual["version"]
            if actual["is_final"]:
                stopped = "PREMATURE_COMPLETION" if premature else "COMPLETED"
        predicted = "ERROR" if error else last["domain"] if last and last["is_final"] else "UNRESOLVED"
        outcomes.append({"id": case["id"], "expected": case["expected_domain"], "predicted": predicted,
                         "clarifications": last["clarification_count"] if last else 0,
                         "status": last["status"] if last else None,
                         "reason_code": last["reason_code"] if last else None, "error": error,
                         "passed": all(t["passed"] for t in turn_results),
                         "premature_completion": premature, "turns": turn_results})
    return summarize(outcomes, latencies)


def summarize(outcomes, latencies):
    labels = sorted({row["expected"] for row in outcomes})
    metrics = {}
    for label in labels:
        tp = sum(row["expected"] == label == row["predicted"] for row in outcomes)
        fp = sum(row["expected"] != label == row["predicted"] for row in outcomes)
        fn = sum(row["expected"] == label != row["predicted"] for row in outcomes)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        metrics[label] = {"precision": precision, "recall": recall,
                          "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0}
    count = len(outcomes)
    classified = [r for r in outcomes if r["predicted"] not in {"UNKNOWN", "ERROR", "UNRESOLVED"}]
    unknown = [r for r in outcomes if r["expected"] == "UNKNOWN"]
    latency = sorted(latencies)
    def percentile(q):
        return latency[max(0, math.ceil(len(latency) * q) - 1)] if latency else None
    checked = [r for r in outcomes if "turns" in r]
    turns = [t for r in checked for t in r["turns"]]
    return {
        "dialogue_pass_rate": sum(r["passed"] for r in checked) / len(checked) if checked else None,
        "turn_pass_rate": sum(t["passed"] for t in turns) / len(turns) if turns else None,
        "planned_turns": len(turns),
        "skipped_turns": sum("skipped" in t for t in turns),
        "premature_completions": sum(r["premature_completion"] for r in checked),
        "cases": count, "per_domain": metrics,
        "macro_f1": sum(v["f1"] for v in metrics.values()) / len(metrics) if metrics else 0,
        "auto_classification_coverage": len(classified) / count if count else 0,
        "auto_classification_accuracy": sum(r["expected"] == r["predicted"] for r in classified) / len(classified) if classified else None,
        "out_of_scope_misroute_rate": sum(r in classified for r in unknown) / len(unknown) if unknown else None,
        "out_of_scope_unresolved_rate": sum(r["predicted"] == "UNRESOLVED" for r in unknown) / len(unknown) if unknown else None,
        "mean_clarifications": sum(r["clarifications"] for r in outcomes) / count if count else 0,
        "clarification_limit_violations": sum(r["clarifications"] > 2 for r in outcomes),
        "technical_errors": sum(r["error"] is not None for r in outcomes),
        "latency_ms": {"p50": percentile(0.5), "p95": percentile(0.95)},
        "confusion": dict(Counter(f'{r["expected"]} -> {r["predicted"]}' for r in outcomes)),
        "results": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    try:
        cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not cases:
            raise ValueError("Dataset must not be empty")
        for case in cases:
            normalize_case(case)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report = evaluate(args.url, os.environ["API_KEY"], cases, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

