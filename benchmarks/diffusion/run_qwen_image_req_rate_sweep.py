#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run Qwen-Image request-rate sweeps against diffusion_benchmark_serving.py."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm


def _parse_request_rates(value: str) -> list[str]:
    rates = [item.strip() for item in value.split(",") if item.strip()]
    if not rates:
        raise argparse.ArgumentTypeError("At least one request rate is required.")
    return rates


def _rate_slug(rate: str) -> str:
    return rate.replace(".", "_")


def _load_summary_rows(metrics_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(metrics_dir.glob("rate_*.json")):
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append(
            {
                "request_rate": str(metrics.get("request_rate", path.stem.removeprefix("rate_").replace("_", "."))),
                "benchmark_duration_s": metrics.get("duration", 0),
                "throughput_qps": metrics.get("throughput_qps", 0),
                "latency_mean_s": metrics.get("latency_mean", 0),
                "latency_p95_s": metrics.get("latency_p95", 0),
                "latency_p99_s": metrics.get("latency_p99", 0),
                "metrics_path": str(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep request rates for Qwen-Image diffusion benchmarks.")
    parser.add_argument(
        "--request-rates",
        type=_parse_request_rates,
        required=True,
        help="Comma-separated request rates to test, e.g. 0.05,0.1,0.2,0.5,1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store per-rate logs, metrics, and optional dumped media.",
    )
    parser.add_argument(
        "--benchmark-script",
        type=Path,
        default=Path(__file__).with_name("diffusion_benchmark_serving.py"),
        help="Path to diffusion_benchmark_serving.py.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Server base URL, e.g. http://127.0.0.1:8091")
    parser.add_argument("--host", type=str, default="localhost", help="Server host if --base-url is unset.")
    parser.add_argument("--port", type=int, default=8091, help="Server port if --base-url is unset.")
    parser.add_argument("--model", type=str, default="default", help="Model name sent to the benchmark.")
    parser.add_argument("--backend", type=str, default="vllm-omni", help="Benchmark backend.")
    parser.add_argument("--dataset", type=str, default="random", help="Benchmark dataset.")
    parser.add_argument("--task", type=str, default="t2i", help="Benchmark task.")
    parser.add_argument("--num-prompts", type=int, default=200, help="Number of prompts per run.")
    parser.add_argument("--seed", type=int, default=42, help="Diffusion generation seed passed to every request.")
    parser.add_argument(
        "--random-request-seed",
        type=int,
        default=42,
        help="Seed for sampling request profiles from --random-request-config.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1000, help="Max in-flight requests.")
    parser.add_argument("--warmup-requests", type=int, default=1, help="Warmup request count.")
    parser.add_argument(
        "--warmup-num-inference-steps",
        type=int,
        default=None,
        help="Warmup num_inference_steps override. If unset, benchmark default is used.",
    )
    parser.add_argument("--enable-negative-prompt", action="store_true", help="Pass through to benchmark.")
    parser.add_argument("--prompt", type=str, default=None, help="Fixed prompt for the random dataset.")
    parser.add_argument(
        "--random-request-config",
        type=str,
        required=True,
        help="JSON string passed through to the benchmark.",
    )
    parser.add_argument(
        "--dump-responses",
        action="store_true",
        help="If set, dump generated images for each rate under output_dir/dumps/<rate>/.",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Extra raw arguments appended to each benchmark command.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip request rates whose metrics JSON already exists and looks complete.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun all request rates even if metrics JSON already exists.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, rate: str, metrics_path: Path, log_path: Path) -> list[str]:
    command: list[str] = [
        sys.executable,
        str(args.benchmark_script),
        "--backend",
        args.backend,
        "--dataset",
        args.dataset,
        "--task",
        args.task,
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
        "--random-request-seed",
        str(args.random_request_seed),
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        rate,
        "--random-request-config",
        args.random_request_config,
        "--output-file",
        str(metrics_path),
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    else:
        command.extend(["--host", args.host, "--port", str(args.port)])
    if args.model:
        command.extend(["--model", args.model])
    if args.warmup_requests is not None:
        command.extend(["--warmup-requests", str(args.warmup_requests)])
    if args.warmup_num_inference_steps is not None:
        command.extend(["--warmup-num-inference-steps", str(args.warmup_num_inference_steps)])
    if args.enable_negative_prompt:
        command.append("--enable-negative-prompt")
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.dump_responses:
        dump_dir = log_path.parent.parent / "dumps" / f"rate_{_rate_slug(rate)}"
        command.extend(["--dump-responses-dir", str(dump_dir)])
    if args.extra_args:
        command.extend(shlex.split(args.extra_args))
    return command


def main() -> int:
    args = parse_args()
    if args.resume and args.force_rerun:
        raise SystemExit("--resume and --force-rerun cannot be used together.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    metrics_dir = args.output_dir / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=len(args.request_rates), desc="Request-rate sweep", unit="run")

    for index, rate in enumerate(args.request_rates, start=1):
        slug = _rate_slug(rate)
        metrics_path = metrics_dir / f"rate_{slug}.json"
        log_path = logs_dir / f"rate_{slug}.log"
        if args.resume and metrics_path.exists():
            try:
                existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                existing_metrics = None
            if isinstance(existing_metrics, dict) and "throughput_qps" in existing_metrics and "latency_p95" in existing_metrics:
                print()
                print(f"[{index}/{len(args.request_rates)}] request_rate={rate} already completed, skipping due to --resume")
                print(f"Metrics: {metrics_path}")
                progress.update(1)
                progress.set_postfix_str(f"skipped_rate={rate}")
                continue

        command = build_command(args, rate, metrics_path, log_path)
        joined = shlex.join(command)

        print()
        print(f"[{index}/{len(args.request_rates)}] request_rate={rate}")
        print(f"Command: {joined}")
        print(f"Log: {log_path}")
        print(f"Metrics: {metrics_path}")

        start_time = time.time()
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {joined}\n\n")
            log_file.flush()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(f"[rate={rate}] {line}")
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()

        duration_s = time.time() - start_time
        if return_code != 0:
            print(f"Run failed for request_rate={rate} with exit code {return_code}")
            return return_code

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["request_rate"] = rate
        metrics["sweep_command"] = command
        metrics["sweep_wall_time_s"] = duration_s
        metrics["log_path"] = str(log_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        progress.update(1)
        progress.set_postfix_str(f"last_rate={rate}")

    progress.close()
    summary_path = args.output_dir / "sweep_summary.json"
    summary = _load_summary_rows(metrics_dir)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print(f"Sweep complete. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
