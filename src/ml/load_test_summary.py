"""Parse Locust CSV benchmark stats and populate the 'load_test' block in metrics.json.

Usage:
    python -m src.ml.load_test_summary
    python -m src.ml.load_test_summary --csv locust_stats_stats.csv --users 30 --duration 30s
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_locust_csv(csv_path: Path, users: int = 30, duration: str = "30s") -> dict[str, Any]:
    """Parse a Locust _stats.csv file into structured benchmark metrics."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Locust stats CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    endpoints: list[dict[str, Any]] = []
    aggregate_data: dict[str, Any] = {}

    for _, row in df.iterrows():
        raw_name = str(row["Name"]).strip()
        is_agg = raw_name == "Aggregated" or pd.isna(row.get("Type"))

        req_count = int(row["Request Count"])
        fail_count = int(row["Failure Count"])
        fail_pct = round((float(fail_count) / max(req_count, 1)) * 100, 2)
        p50 = int(round(float(row["Median Response Time"])))
        p95 = int(round(float(row["95%"])))
        avg = round(float(row["Average Response Time"]), 1)
        rps = round(float(row["Requests/s"]), 2)

        entry = {
            "name": raw_name,
            "requests": req_count,
            "failures": fail_pct,
            "p50": p50,
            "p95": p95,
            "avg": avg,
            "rps": rps,
        }

        if is_agg:
            approx_rps = int(round(rps))
            entry["offered_load_desc"] = (
                f"Executed against the production Linux Docker container running multi-worker Gunicorn "
                f"with {users} concurrent users for {duration} via headless Locust, representing "
                f"~{approx_rps} req/s offered load ({users} Locust users with think time; "
                f"load generator and server share a GitHub Actions runner):"
            )
            aggregate_data = entry
        else:
            endpoints.append(entry)

    return {
        "users": users,
        "duration": duration,
        "endpoints": endpoints,
        "aggregate": aggregate_data,
    }


def update_metrics_load_test(
    load_test_data: dict[str, Any],
    metrics_path: Path = Path("model_files/metrics.json"),
) -> None:
    """Safely merge the 'load_test' block into metrics.json without clobbering other keys."""
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    metrics["load_test"] = load_test_data

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Successfully updated 'load_test' block in %s", metrics_path)


def main() -> None:
    """CLI entrypoint for parsing locust stats and updating metrics.json."""
    parser = argparse.ArgumentParser(description="Parse Locust CSV load test results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("model_files/locust_benchmark.csv"),
        help="Path to Locust _stats.csv file.",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=30,
        help="Number of concurrent Locust users used in test.",
    )
    parser.add_argument(
        "--duration",
        type=str,
        default="30s",
        help="Duration of load test.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("model_files/metrics.json"),
        help="Path to metrics.json file.",
    )
    args = parser.parse_args()

    # Fallback to local default if specified file does not exist
    target_csv = args.csv
    if not target_csv.exists() and Path("model_files/locust_benchmark.csv").exists():
        target_csv = Path("model_files/locust_benchmark.csv")

    load_test_data = parse_locust_csv(target_csv, users=args.users, duration=args.duration)
    update_metrics_load_test(load_test_data, metrics_path=args.metrics_path)
    logger.info(
        "Load test parsed: %d total requests, %d ms p50, %d ms p95, %.2f req/s",
        load_test_data["aggregate"]["requests"],
        load_test_data["aggregate"]["p50"],
        load_test_data["aggregate"]["p95"],
        load_test_data["aggregate"]["rps"],
    )


if __name__ == "__main__":
    main()
