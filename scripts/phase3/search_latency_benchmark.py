#!/usr/bin/env python3
"""
Phase 3: Search latency benchmark for /videos/{id}/search.

Usage:
  .venv/bin/python scripts/phase3/search_latency_benchmark.py --video-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.env import load_env


DEFAULT_QUERIES = [
    "person speaking to camera",
    "close up shot",
    "text on screen",
]


@dataclass(frozen=True)
class BenchmarkSample:
    query_text: str
    run_index: int
    latency_ms: float
    status_code: int
    result_count: int
    error: str | None = None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _post_search(
    *,
    api_url: str,
    video_id: str,
    query_text: str,
    limit: int,
    timeout_s: float,
) -> tuple[int, dict[str, Any], float]:
    endpoint = f"{api_url.rstrip('/')}/videos/{video_id}/search"
    payload = json.dumps({"query_text": query_text, "limit": limit}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
            status_code = response.status
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status_code = exc.code
    except error.URLError as exc:
        raise RuntimeError(f"Failed to connect to API: {exc.reason}") from exc

    latency_ms = (time.perf_counter() - start) * 1000

    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        data = {"raw_body": body}

    return status_code, data, latency_ms


def _load_queries(queries_file: Path | None) -> list[str]:
    if queries_file is None:
        return DEFAULT_QUERIES

    payload = json.loads(queries_file.read_text())
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("queries file must be a JSON array of strings")

    queries = [item.strip() for item in payload if item.strip()]
    if not queries:
        raise ValueError("queries file must contain at least one non-empty query")
    return queries


def run_benchmark(
    *,
    api_url: str,
    video_id: str,
    queries: list[str],
    runs_per_query: int,
    limit: int,
    timeout_s: float,
    sleep_between_s: float,
) -> list[BenchmarkSample]:
    samples: list[BenchmarkSample] = []

    for query_text in queries:
        for run_index in range(runs_per_query):
            status_code, data, latency_ms = _post_search(
                api_url=api_url,
                video_id=video_id,
                query_text=query_text,
                limit=limit,
                timeout_s=timeout_s,
            )

            if status_code == 200:
                result_count = len(data.get("results", []))
                sample = BenchmarkSample(
                    query_text=query_text,
                    run_index=run_index,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    result_count=result_count,
                )
            else:
                detail = data.get("detail") if isinstance(data, dict) else str(data)
                sample = BenchmarkSample(
                    query_text=query_text,
                    run_index=run_index,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    result_count=0,
                    error=str(detail),
                )

            samples.append(sample)
            status_label = "OK" if status_code == 200 else "ERR"
            print(
                f"[{status_label}] query={query_text!r} run={run_index + 1}/{runs_per_query} "
                f"latency={latency_ms:.1f}ms status={status_code} results={sample.result_count}"
            )

            if sleep_between_s > 0:
                time.sleep(sleep_between_s)

    return samples


def print_summary(samples: list[BenchmarkSample]) -> int:
    ok_samples = [sample for sample in samples if sample.status_code == 200]
    hot_samples = [sample for sample in ok_samples if sample.run_index > 0]
    cold_samples = [sample for sample in ok_samples if sample.run_index == 0]

    all_latencies = [sample.latency_ms for sample in ok_samples]
    hot_latencies = [sample.latency_ms for sample in hot_samples]
    cold_latencies = [sample.latency_ms for sample in cold_samples]

    print("\n=== Search Latency Summary ===")
    print(f"Total requests: {len(samples)}")
    print(f"Successful requests: {len(ok_samples)}")

    if all_latencies:
        print(
            "All successful: "
            f"mean={statistics.mean(all_latencies):.1f}ms "
            f"p50={_percentile(all_latencies, 0.50):.1f}ms "
            f"p95={_percentile(all_latencies, 0.95):.1f}ms"
        )

    if cold_latencies:
        print(
            "Cold candidate (first run/query): "
            f"mean={statistics.mean(cold_latencies):.1f}ms "
            f"p50={_percentile(cold_latencies, 0.50):.1f}ms"
        )

    if hot_latencies:
        print(
            "Hot path (run>1/query): "
            f"mean={statistics.mean(hot_latencies):.1f}ms "
            f"p50={_percentile(hot_latencies, 0.50):.1f}ms "
            f"p95={_percentile(hot_latencies, 0.95):.1f}ms"
        )

    failures = [sample for sample in samples if sample.status_code != 200]
    if failures:
        print("\nFailures:")
        for sample in failures:
            print(
                f"- query={sample.query_text!r} run={sample.run_index + 1} "
                f"status={sample.status_code} error={sample.error}"
            )

    return 0 if not failures else 1


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="Benchmark video search latency")
    parser.add_argument("--video-id", required=True, help="Video UUID to benchmark")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Optional JSON file with query array",
    )
    parser.add_argument(
        "--runs-per-query",
        type=int,
        default=3,
        help="How many sequential runs per query (default: 3)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Search limit (default: 5)")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="HTTP timeout per request in seconds (default: 60)",
    )
    parser.add_argument(
        "--sleep-between-s",
        type=float,
        default=0.0,
        help="Sleep between requests in seconds (default: 0)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write raw benchmark samples as JSON",
    )
    args = parser.parse_args()

    if args.runs_per_query <= 0:
        raise ValueError("--runs-per-query must be > 0")

    queries = _load_queries(args.queries_file)
    print(
        "Running benchmark with "
        f"queries={len(queries)} runs_per_query={args.runs_per_query} api={args.api_url}"
    )

    samples = run_benchmark(
        api_url=args.api_url,
        video_id=args.video_id,
        queries=queries,
        runs_per_query=args.runs_per_query,
        limit=args.limit,
        timeout_s=args.timeout_s,
        sleep_between_s=args.sleep_between_s,
    )

    if args.json_output is not None:
        payload = [asdict(sample) for sample in samples]
        args.json_output.write_text(json.dumps(payload, indent=2))
        print(f"Saved raw samples to {args.json_output}")

    return print_summary(samples)


if __name__ == "__main__":
    sys.exit(main())
