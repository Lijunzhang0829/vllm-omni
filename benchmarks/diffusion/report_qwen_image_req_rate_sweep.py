#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a P95-vs-request-rate plot and summary tables from sweep metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render plots and tables for Qwen-Image request-rate sweeps.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Sweep output directory created by run_qwen_image_req_rate_sweep.py.",
    )
    return parser.parse_args()


def _rate_sort_key(value: str) -> tuple[int, float]:
    lowered = value.lower()
    if lowered == "inf":
        return (1, float("inf"))
    return (0, float(value))


def _load_rows(metrics_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_dir.glob("rate_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "request_rate": str(data["request_rate"]),
                "benchmark_duration_s": float(data.get("duration", 0.0)),
                "throughput_qps": float(data.get("throughput_qps", 0.0)),
                "latency_mean_s": float(data.get("latency_mean", 0.0)),
                "latency_p95_s": float(data.get("latency_p95", 0.0)),
                "latency_p99_s": float(data.get("latency_p99", 0.0)),
            }
        )
    rows.sort(key=lambda row: _rate_sort_key(str(row["request_rate"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "request_rate",
        "benchmark_duration_s",
        "throughput_qps",
        "latency_mean_s",
        "latency_p95_s",
        "latency_p99_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| req rate | Benchmark duration (s) | Request throughput (req/s) | Latency Mean (s) | Latency P95 (s) | Latency P99 (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {request_rate} | {benchmark_duration_s:.4f} | {throughput_qps:.4f} | "
            "{latency_mean_s:.4f} | {latency_p95_s:.4f} | {latency_p99_s:.4f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["request_rate"]) for row in rows]
    p95 = [float(row["latency_p95_s"]) for row in rows]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, p95, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("--request-rate")
    ax.set_ylabel("E2E P95 latency (s)")
    ax.set_title("Qwen-Image request-rate sweep")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    metrics_dir = args.input_dir / "metrics"
    if not metrics_dir.exists():
        raise SystemExit(f"Metrics directory not found: {metrics_dir}")

    rows = _load_rows(metrics_dir)
    if not rows:
        raise SystemExit(f"No metrics JSON files found under {metrics_dir}")

    csv_path = args.input_dir / "request_rate_summary.csv"
    md_path = args.input_dir / "request_rate_summary.md"
    plot_path = args.input_dir / "request_rate_vs_p95.png"

    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    _write_plot(plot_path, rows)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote Plot: {plot_path}")
    print()
    print(md_path.read_text(encoding='utf-8'))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
