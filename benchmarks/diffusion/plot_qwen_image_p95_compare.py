#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a grouped bar chart comparing baseline vs super_pxx P95."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot grouped P95 bar chart from two request-rate summary CSV files.")
    parser.add_argument("--baseline-csv", type=Path, required=True, help="Path to baseline request_rate_summary.csv")
    parser.add_argument("--super-pxx-csv", type=Path, required=True, help="Path to super_pxx request_rate_summary.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/diffusion/results/qwen_image_p95_compare.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def load_p95_rows(path: Path) -> dict[str, float]:
    rows: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rate = row["request_rate"].strip()
            rows[rate] = float(row["latency_p95_s"])
    return rows


def fmt_value(value: float) -> str:
    return f"{value:.1f}"


def fmt_improvement(baseline: float, candidate: float) -> str:
    if baseline <= 0:
        return "n/a"
    improvement_pct = (baseline - candidate) / baseline * 100.0
    return f"-{improvement_pct:.1f}%"


def main() -> int:
    args = parse_args()

    baseline_rows = load_p95_rows(args.baseline_csv)
    super_rows = load_p95_rows(args.super_pxx_csv)

    ordered_rates = [f"{rate:.1f}" for rate in np.arange(0.3, 1.0, 0.1)]
    missing = [rate for rate in ordered_rates if rate not in baseline_rows or rate not in super_rows]
    if missing:
        raise ValueError(f"Missing request_rate rows in input CSVs: {missing}")

    baseline_p95 = [baseline_rows[rate] for rate in ordered_rates]
    super_p95 = [super_rows[rate] for rate in ordered_rates]

    x = np.arange(len(ordered_rates))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 7))
    baseline_bars = ax.bar(x - width / 2, baseline_p95, width, label="baseline", color="#5B7DB1")
    super_bars = ax.bar(x + width / 2, super_p95, width, label="super_pxx", color="#D97A41")

    ax.set_title("Qwen-Image P95 E2E Latency Comparison (500 Requests)")
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("P95 E2E Latency (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_rates)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.text(
        0.99,
        0.98,
        "Lower is better",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F5F5F5", "edgecolor": "#DDDDDD"},
    )

    y_max = max(max(baseline_p95), max(super_p95))
    value_offset = y_max * 0.015
    improvement_offset = y_max * 0.055

    for bar, value in zip(baseline_bars, baseline_p95):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + value_offset,
            fmt_value(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar, base, value in zip(super_bars, baseline_p95, super_p95):
        center_x = bar.get_x() + bar.get_width() / 2
        ax.text(
            center_x,
            bar.get_height() + value_offset,
            fmt_value(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            center_x,
            bar.get_height() + improvement_offset,
            fmt_improvement(base, value),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#7A2E0E",
            fontweight="bold",
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote Plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
