#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Least-load dispatcher for multiple Qwen-Image backends."""

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


def _parse_size(size: str | None) -> tuple[int | None, int | None]:
    if not size or "x" not in size:
        return None, None
    try:
        width_str, height_str = size.lower().split("x", 1)
        return int(width_str), int(height_str)
    except Exception:
        return None, None


def estimate_request_cost(payload: dict[str, Any]) -> int:
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}

    width = extra_body.get("width")
    height = extra_body.get("height")
    steps = extra_body.get("num_inference_steps")

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
    except Exception:
        width_i, height_i, steps_i = 1024, 1024, 50

    return width_i * height_i * steps_i


@dataclass
class BackendState:
    name: str
    base_url: str
    healthy: bool = True
    inflight_requests: int = 0
    inflight_cost: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    latency_ema_s: float = 0.0
    last_error: str = ""
    last_healthcheck_ts: float = 0.0

    def score(self) -> tuple[int, int, float, str]:
        return (self.inflight_cost, self.inflight_requests, self.latency_ema_s, self.name)


class BackendPool:
    def __init__(self, backend_urls: list[str], max_inflight_per_backend: int):
        self.backends = [
            BackendState(name=f"server-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(backend_urls)
        ]
        self.max_inflight_per_backend = max_inflight_per_backend
        self._lock = threading.Lock()

    def choose_backend(self, estimated_cost: int) -> BackendState:
        with self._lock:
            candidates = [
                backend
                for backend in self.backends
                if backend.healthy
                and (self.max_inflight_per_backend <= 0 or backend.inflight_requests < self.max_inflight_per_backend)
            ]
            if not candidates:
                candidates = [backend for backend in self.backends if backend.healthy]
            if not candidates:
                raise RuntimeError("No healthy backends available")

            selected = min(candidates, key=lambda backend: backend.score())
            selected.inflight_requests += 1
            selected.inflight_cost += estimated_cost
            return selected

    def complete_backend(
        self,
        backend: BackendState,
        estimated_cost: int,
        latency_s: float,
        success: bool,
        error: str,
    ) -> None:
        with self._lock:
            backend.inflight_requests = max(0, backend.inflight_requests - 1)
            backend.inflight_cost = max(0, backend.inflight_cost - estimated_cost)
            backend.last_error = error
            if success:
                backend.completed_requests += 1
                backend.healthy = True
            else:
                backend.failed_requests += 1
            if latency_s > 0:
                if backend.latency_ema_s <= 0:
                    backend.latency_ema_s = latency_s
                else:
                    backend.latency_ema_s = 0.8 * backend.latency_ema_s + 0.2 * latency_s

    def set_health(self, backend: BackendState, healthy: bool, error: str = "") -> None:
        with self._lock:
            backend.healthy = healthy
            backend.last_error = error
            backend.last_healthcheck_ts = time.time()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": backend.name,
                    "base_url": backend.base_url,
                    "healthy": backend.healthy,
                    "inflight_requests": backend.inflight_requests,
                    "inflight_cost": backend.inflight_cost,
                    "completed_requests": backend.completed_requests,
                    "failed_requests": backend.failed_requests,
                    "latency_ema_s": round(backend.latency_ema_s, 4),
                    "last_error": backend.last_error,
                    "last_healthcheck_ts": backend.last_healthcheck_ts,
                }
                for backend in self.backends
            ]


def proxy_request(
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_s: int,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url=url, data=body, method=method)
    for key, value in headers.items():
        if key.lower() == "host":
            continue
        request.add_header(key, value)

    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response_body = response.read()
        response_headers = {key: value for key, value in response.headers.items()}
        return response.status, response_body, response_headers


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
        body = json.dumps({"error": message}).encode("utf-8")
        self._write_response(status, body, {"Content-Type": "application/json"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            healthy = any(item["healthy"] for item in self.pool.snapshot())
            status = HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE
            self._write_response(status, b"ok" if healthy else b"unhealthy", {"Content-Type": "text/plain"})
            return

        if self.path == "/backends":
            body = json.dumps({"backends": self.pool.snapshot()}, indent=2).encode("utf-8")
            self._write_response(HTTPStatus.OK, body, {"Content-Type": "application/json"})
            return

        if self.path == "/v1/models":
            backends = [backend for backend in self.pool.backends if backend.healthy]
            if not backends:
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, "No healthy backends available")
                return
            backend = min(backends, key=lambda item: item.score())
            try:
                status, body, headers = proxy_request(
                    "GET",
                    f"{backend.base_url}{self.path}",
                    None,
                    dict(self.headers.items()),
                    self.request_timeout_s,
                )
                self.pool.set_health(backend, True)
                self._write_response(status, body, headers)
            except urllib.error.HTTPError as exc:
                self.pool.set_health(backend, False, f"HTTP {exc.code}")
                self._write_response(exc.code, exc.read(), {"Content-Type": exc.headers.get_content_type()})
            except Exception as exc:
                self.pool.set_health(backend, False, str(exc))
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
        except Exception as exc:
            self._json_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
            return

        estimated_cost = estimate_request_cost(payload)
        started_at = time.perf_counter()
        try:
            backend = self.pool.choose_backend(estimated_cost)
        except Exception as exc:
            self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        try:
            status, response_body, response_headers = proxy_request(
                "POST",
                f"{backend.base_url}{self.path}",
                body,
                dict(self.headers.items()),
                self.request_timeout_s,
            )
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(backend, estimated_cost, latency_s, True, "")
            self._write_response(status, response_body, response_headers)
        except urllib.error.HTTPError as exc:
            latency_s = time.perf_counter() - started_at
            error_body = exc.read()
            self.pool.complete_backend(backend, estimated_cost, latency_s, False, f"HTTP {exc.code}")
            self._write_response(exc.code, error_body, {"Content-Type": exc.headers.get_content_type()})
        except Exception as exc:
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(backend, estimated_cost, latency_s, False, str(exc))
            self._json_error(HTTPStatus.BAD_GATEWAY, f"{backend.name} failed: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        return


def healthcheck_loop(pool: BackendPool, interval_s: int, timeout_s: int) -> None:
    while True:
        for backend in pool.backends:
            try:
                status, _, _ = proxy_request("GET", f"{backend.base_url}/health", None, {}, timeout_s)
                pool.set_health(backend, status == HTTPStatus.OK, "" if status == HTTPStatus.OK else f"HTTP {status}")
            except Exception as exc:
                pool.set_health(backend, False, str(exc))
        time.sleep(interval_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Least-load dispatcher for multi-server Qwen-Image deployment.")
    parser.add_argument("--host", default="0.0.0.0", help="Dispatcher listen host.")
    parser.add_argument("--port", type=int, default=8090, help="Dispatcher listen port.")
    parser.add_argument(
        "--backend-urls",
        nargs="+",
        required=True,
        help="Backend base URLs, e.g. http://127.0.0.1:8091 http://127.0.0.1:8092",
    )
    parser.add_argument(
        "--max-inflight-per-backend",
        type=int,
        default=2,
        help="Inflight request cap per backend. Set <=0 to disable.",
    )
    parser.add_argument("--healthcheck-interval", type=int, default=5, help="Healthcheck interval in seconds.")
    parser.add_argument("--healthcheck-timeout", type=int, default=5, help="Healthcheck timeout in seconds.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Proxy request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = BackendPool(args.backend_urls, max_inflight_per_backend=args.max_inflight_per_backend)
    DispatcherHandler.pool = pool
    DispatcherHandler.request_timeout_s = args.request_timeout

    health_thread = threading.Thread(
        target=healthcheck_loop,
        args=(pool, args.healthcheck_interval, args.healthcheck_timeout),
        daemon=True,
    )
    health_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), DispatcherHandler)
    print(f"Dispatcher listening on http://{args.host}:{args.port}")
    for backend in pool.backends:
        print(f"  - {backend.name}: {backend.base_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
