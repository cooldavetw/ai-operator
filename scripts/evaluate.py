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


def evaluate(base_url, api_key, cases, timeout):
    outcomes, latencies = [], []
    for case in cases:
        conversation, version, last, error = None, 0, None, None
        for text in case["messages"]:
            body = {"request_id": str(uuid4()), "conversation_id": conversation,
                    "expected_version": version, "message": text}
            request = urllib.request.Request(
                base_url.rstrip("/") + "/api/v1/classify",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "X-API-Key": api_key},
            )
            started = time.perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    last = json.load(response)
            except (urllib.error.URLError, TimeoutError) as exc:
                error = f"HTTP_{exc.code}" if isinstance(exc, urllib.error.HTTPError) else "TRANSPORT_ERROR"
                break
            finally:
                latencies.append((time.perf_counter() - started) * 1000)
            conversation, version = last["conversation_id"], last["version"]
            if last["is_final"]:
                break
        predicted = "ERROR" if error else last["domain"] if last and last["is_final"] else "UNRESOLVED"
        outcomes.append({"id": case["id"], "expected": case["expected_domain"], "predicted": predicted,
                         "clarifications": last["clarification_count"] if last else 0, "error": error})
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
    return {
        "cases": count, "per_domain": metrics,
        "macro_f1": sum(v["f1"] for v in metrics.values()) / len(metrics) if metrics else 0,
        "auto_classification_coverage": len(classified) / count if count else 0,
        "auto_classification_accuracy": sum(r["expected"] == r["predicted"] for r in classified) / len(classified) if classified else None,
        "out_of_scope_misroute_rate": sum(r in classified for r in unknown) / len(unknown) if unknown else None,
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
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases or any(not c.get("messages") or not c.get("expected_domain") or not c.get("id") for c in cases):
        parser.error("Each case needs id, nonempty messages, and expected_domain")
    report = evaluate(args.url, os.environ["API_KEY"], cases, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

