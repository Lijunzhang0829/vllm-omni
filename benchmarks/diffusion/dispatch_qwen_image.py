#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dispatcher for multi-server Qwen-Image deployment."""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_REQUEST_TIMEOUT_S = 60 * 60
DELAY_X_DISPATCHER_SACRIFICIAL_HEADER = "X-DelayX-Sacrificial"
QWEN_IMAGE_PROFILE_LATENCY_S: dict[tuple[int, int, int, int], float] = {
    (512, 512, 20, 1): 22.35,
    (768, 768, 20, 1): 20.62,
    (1024, 1024, 25, 1): 33.90,
    (1536, 1536, 35, 1): 102.66,
}
QWEN_IMAGE_FALLBACK_REFERENCE_KEY = (1024, 1024, 25, 1)
QWEN_IMAGE_FALLBACK_REFERENCE_LATENCY_S = QWEN_IMAGE_PROFILE_LATENCY_S[QWEN_IMAGE_FALLBACK_REFERENCE_KEY]


def _parse_size(size: str | None) -> tuple[int | None, int | None]:
    if not size or "x" not in size:
        return None, None
    try:
        width_str, height_str = size.lower().split("x", 1)
        return int(width_str), int(height_str)
    except Exception:
        return None, None


def estimate_request_shape(payload: dict[str, Any]) -> tuple[int, int, int, int]:
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    width = extra_body.get("width")
    height = extra_body.get("height")
    steps = extra_body.get("num_inference_steps")
    num_frames = extra_body.get("num_frames", payload.get("num_frames", 1))

    if width is None or height is None:
        width, height = _parse_size(extra_body.get("size"))
    if width is None or height is None:
        width, height = _parse_size(payload.get("size"))

    if width is None:
        width = payload.get("width", 1024)
    if height is None:
        height = payload.get("height", 1024)
    if steps is None:
        steps = payload.get("num_inference_steps", extra_body.get("steps", 50))

    try:
        width_i = max(1, int(width))
        height_i = max(1, int(height))
        steps_i = max(1, int(steps))
        num_frames_i = max(1, int(num_frames))
    except Exception:
        width_i, height_i, steps_i, num_frames_i = 1024, 1024, 50, 1

    return width_i, height_i, steps_i, num_frames_i


def estimate_request_cost(payload: dict[str, Any]) -> int:
    width_i, height_i, steps_i, num_frames_i = estimate_request_shape(payload)
    return width_i * height_i * steps_i * num_frames_i


def estimate_request_service_s(payload: dict[str, Any]) -> float:
    width_i, height_i, steps_i, num_frames_i = estimate_request_shape(payload)
    profile_key = (width_i, height_i, steps_i, num_frames_i)
    if profile_key in QWEN_IMAGE_PROFILE_LATENCY_S:
        return QWEN_IMAGE_PROFILE_LATENCY_S[profile_key]

    total_cost = width_i * height_i * steps_i * num_frames_i
    reference_cost = (
        QWEN_IMAGE_FALLBACK_REFERENCE_KEY[0]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[1]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[2]
        * QWEN_IMAGE_FALLBACK_REFERENCE_KEY[3]
    )
    if total_cost <= 0 or reference_cost <= 0:
        return 0.0
    return QWEN_IMAGE_FALLBACK_REFERENCE_LATENCY_S * (float(total_cost) / float(reference_cost))


@dataclass
class BackendState:
    name: str
    base_url: str
    inflight_normal_requests: int = 0
    inflight_normal_cost: int = 0
    inflight_sacrificial_requests: int = 0
    inflight_sacrificial_cost: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    latency_ema_s: float = 0.0
    last_error: str = ""

    def normal_score(self) -> tuple[int, int, float, str]:
        return (self.inflight_normal_cost, self.inflight_normal_requests, self.latency_ema_s, self.name)

    def effective_total_score(self, sacrificial_load_factor: float) -> tuple[float, float, float, str]:
        factor = max(0.0, float(sacrificial_load_factor))
        return (
            self.inflight_normal_cost + factor * self.inflight_sacrificial_cost,
            self.inflight_normal_requests + factor * self.inflight_sacrificial_requests,
            self.latency_ema_s,
            self.name,
        )


def proxy_request(
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_s: int,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url=url, data=body, method=method)
    for key, value in headers.items():
        if key.lower() in {"host", "content-length", "transfer-encoding", "connection"}:
            continue
        request.add_header(key, value)

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_body = response.read()
        response_headers = {key: value for key, value in response.headers.items()}
        return response.status, response_body, response_headers


class BackendPool:
    def __init__(
        self,
        backend_urls: list[str],
        delay_x_quota_every: int = 0,
        delay_x_quota_amount: int = 0,
        delay_x_threshold_ratio: float = 0.8,
        delay_x_sacrificial_load_factor: float = 0.1,
    ):
        self.backends = [
            BackendState(name=f"server-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(backend_urls)
        ]
        self._lock = threading.Lock()
        self._delay_x_quota_every = max(0, int(delay_x_quota_every))
        self._delay_x_quota_amount = max(0, int(delay_x_quota_amount))
        self._delay_x_threshold_ratio = max(0.0, float(delay_x_threshold_ratio))
        self._delay_x_sacrificial_load_factor = max(0.0, float(delay_x_sacrificial_load_factor))
        self._arrival_counter = 0
        self._delay_x_credits = 0
        self._delay_x_global_max_service_s = 0.0

    def choose_backend(self, estimated_cost: int, estimated_service_s: float) -> tuple[BackendState, bool]:
        with self._lock:
            self._arrival_counter += 1
            if (
                self._delay_x_quota_every > 0
                and self._delay_x_quota_amount > 0
                and self._arrival_counter % self._delay_x_quota_every == 0
            ):
                self._delay_x_credits += self._delay_x_quota_amount

            self._delay_x_global_max_service_s = max(self._delay_x_global_max_service_s, max(0.0, estimated_service_s))
            mark_sacrificial = (
                self._delay_x_credits > 0
                and self._delay_x_global_max_service_s > 0
                and estimated_service_s >= self._delay_x_threshold_ratio * self._delay_x_global_max_service_s
            )
            if mark_sacrificial:
                self._delay_x_credits = max(0, self._delay_x_credits - 1)

            if mark_sacrificial:
                selected = min(
                    self.backends,
                    key=lambda backend: backend.effective_total_score(self._delay_x_sacrificial_load_factor),
                )
                selected.inflight_sacrificial_requests += 1
                selected.inflight_sacrificial_cost += estimated_cost
            else:
                selected = min(self.backends, key=lambda backend: backend.normal_score())
                selected.inflight_normal_requests += 1
                selected.inflight_normal_cost += estimated_cost
            return selected, mark_sacrificial

    def complete_backend(
        self,
        backend: BackendState,
        estimated_cost: int,
        sacrificial: bool,
        latency_s: float,
        success: bool,
        error: str,
    ) -> None:
        with self._lock:
            if sacrificial:
                backend.inflight_sacrificial_requests = max(0, backend.inflight_sacrificial_requests - 1)
                backend.inflight_sacrificial_cost = max(0, backend.inflight_sacrificial_cost - estimated_cost)
            else:
                backend.inflight_normal_requests = max(0, backend.inflight_normal_requests - 1)
                backend.inflight_normal_cost = max(0, backend.inflight_normal_cost - estimated_cost)
            backend.last_error = error
            if success:
                backend.completed_requests += 1
            else:
                backend.failed_requests += 1
            if latency_s > 0:
                if backend.latency_ema_s <= 0:
                    backend.latency_ema_s = latency_s
                else:
                    backend.latency_ema_s = 0.8 * backend.latency_ema_s + 0.2 * latency_s

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": backend.name,
                    "base_url": backend.base_url,
                    "inflight_normal_requests": backend.inflight_normal_requests,
                    "inflight_normal_cost": backend.inflight_normal_cost,
                    "inflight_sacrificial_requests": backend.inflight_sacrificial_requests,
                    "inflight_sacrificial_cost": backend.inflight_sacrificial_cost,
                    "completed_requests": backend.completed_requests,
                    "failed_requests": backend.failed_requests,
                    "latency_ema_s": round(backend.latency_ema_s, 4),
                    "delay_x_credits": self._delay_x_credits,
                    "delay_x_global_max_service_s": round(self._delay_x_global_max_service_s, 4),
                    "delay_x_sacrificial_load_factor": self._delay_x_sacrificial_load_factor,
                    "last_error": backend.last_error,
                }
                for backend in self.backends
            ]


def inject_delay_x_sacrificial_header(headers: dict[str, str], mark_sacrificial: bool) -> dict[str, str]:
    if not mark_sacrificial:
        return headers
    updated_headers = dict(headers)
    updated_headers[DELAY_X_DISPATCHER_SACRIFICIAL_HEADER] = "1"
    return updated_headers


class DispatcherHandler(BaseHTTPRequestHandler):
    pool: BackendPool
    request_timeout_s: int

    def _read_body(self) -> bytes:
        content_length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(content_length) if content_length > 0 else b""

    def _write_response(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            if key.lower() in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str) -> None:
        print(f"[dispatcher] {status} {message}", flush=True)
        body = json.dumps({"error": message}).encode("utf-8")
        self._write_response(status, body, {"Content-Type": "application/json"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_response(HTTPStatus.OK, b"ok", {"Content-Type": "text/plain"})
            return

        if self.path == "/backends":
            body = json.dumps({"backends": self.pool.snapshot()}, indent=2).encode("utf-8")
            self._write_response(HTTPStatus.OK, body, {"Content-Type": "application/json"})
            return

        if self.path == "/v1/models":
            try:
                backend = min(self.pool.backends, key=lambda item: item.normal_score())
                status, body, headers = proxy_request(
                    "GET",
                    f"{backend.base_url}{self.path}",
                    None,
                    dict(self.headers.items()),
                    self.request_timeout_s,
                )
                self._write_response(status, body, headers)
            except urllib.error.HTTPError as exc:
                self._write_response(exc.code, exc.read(), {"Content-Type": exc.headers.get_content_type()})
            except Exception as exc:  # noqa: BLE001
                self._json_error(HTTPStatus.BAD_GATEWAY, str(exc))
            return

        self._json_error(HTTPStatus.NOT_FOUND, f"Unsupported path: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/v1/chat/completions", "/v1/images/generations"}:
            self._json_error(HTTPStatus.NOT_FOUND, f"Unsupported path: {self.path}")
            return

        body = self._read_body()
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as exc:  # noqa: BLE001
            self._json_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
            return

        estimated_cost = estimate_request_cost(payload)
        estimated_service_s = estimate_request_service_s(payload)
        started_at = time.perf_counter()
        try:
            backend, mark_sacrificial = self.pool.choose_backend(estimated_cost, estimated_service_s)
        except Exception as exc:  # noqa: BLE001
            self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        forward_headers = inject_delay_x_sacrificial_header(dict(self.headers.items()), mark_sacrificial)
        try:
            status, response_body, response_headers = proxy_request(
                "POST",
                f"{backend.base_url}{self.path}",
                body,
                forward_headers,
                self.request_timeout_s,
            )
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(
                backend,
                estimated_cost,
                mark_sacrificial,
                latency_s,
                True,
                "",
            )
            self._write_response(status, response_body, response_headers)
        except urllib.error.HTTPError as exc:
            latency_s = time.perf_counter() - started_at
            error_body = exc.read()
            response_headers = {key: value for key, value in exc.headers.items()}
            print(f"[dispatcher] backend={backend.name} HTTP {exc.code} path={self.path}", flush=True)
            self.pool.complete_backend(
                backend,
                estimated_cost,
                mark_sacrificial,
                latency_s,
                False,
                f"HTTP {exc.code}",
            )
            self._write_response(exc.code, error_body, {"Content-Type": exc.headers.get_content_type()})
        except Exception as exc:  # noqa: BLE001
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(backend, estimated_cost, mark_sacrificial, latency_s, False, str(exc))
            print(f"[dispatcher] backend={backend.name} exception path={self.path}: {exc}", flush=True)
            self._json_error(HTTPStatus.BAD_GATEWAY, f"{backend.name} failed: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        return
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatcher for multi-server Qwen-Image deployment.")
    parser.add_argument("--host", default="0.0.0.0", help="Dispatcher listen host.")
    parser.add_argument("--port", type=int, default=8090, help="Dispatcher listen port.")
    parser.add_argument(
        "--backend-urls",
        nargs="+",
        required=True,
        help="Backend base URLs, e.g. http://127.0.0.1:8091 http://127.0.0.1:8092",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Proxy request timeout in seconds.",
    )
    parser.add_argument(
        "--delay-x-quota-every",
        type=int,
        default=0,
        help="In lightweight dispatcher mode, add delay_x sacrificial credits every N arrivals.",
    )
    parser.add_argument(
        "--delay-x-quota-amount",
        type=int,
        default=0,
        help="In lightweight dispatcher mode, number of delay_x sacrificial credits generated per quota event.",
    )
    parser.add_argument(
        "--delay-x-threshold-ratio",
        type=float,
        default=0.8,
        help="Mark a newly arrived request as sacrificial when estimated service >= ratio * global max service.",
    )
    parser.add_argument(
        "--delay-x-sacrificial-load-factor",
        type=float,
        default=0.1,
        help="When dispatching sacrificial requests, count their load at this fraction of normal load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = BackendPool(
        args.backend_urls,
        delay_x_quota_every=args.delay_x_quota_every,
        delay_x_quota_amount=args.delay_x_quota_amount,
        delay_x_threshold_ratio=args.delay_x_threshold_ratio,
        delay_x_sacrificial_load_factor=args.delay_x_sacrificial_load_factor,
    )

    DispatcherHandler.pool = pool
    DispatcherHandler.request_timeout_s = args.request_timeout

    server = ThreadingHTTPServer((args.host, args.port), DispatcherHandler)
    print(f"Dispatcher listening on http://{args.host}:{args.port}")
    print("Scheduling policy: least-load + optional arrival-time delay_x sacrificial tagging")
    for backend in pool.backends:
        print(f"  - {backend.name}: {backend.base_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
