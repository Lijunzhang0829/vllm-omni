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
DELAY_X_DISPATCHER_QUOTA_EXTRA_BODY_KEY = "_delay_x_dispatcher_quota_amount"
DELAY_X_MAX_ACTIVE_LATENCY_HEADER = "X-DelayX-Max-Active-Latency"


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
    inflight_requests: int = 0
    inflight_cost: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    latency_ema_s: float = 0.0
    last_error: str = ""

    def score(self) -> tuple[int, int, float, str]:
        return (self.inflight_cost, self.inflight_requests, self.latency_ema_s, self.name)


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


class BackendPool:
    def __init__(
        self,
        backend_urls: list[str],
        delay_x_quota_every: int = 0,
        delay_x_quota_amount: int = 0,
    ):
        self.backends = [
            BackendState(name=f"server-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(backend_urls)
        ]
        self._lock = threading.Lock()
        self._delay_x_quota_every = max(0, int(delay_x_quota_every))
        self._delay_x_quota_amount = max(0, int(delay_x_quota_amount))
        self._arrival_counter = 0
        self._next_delay_x_backend_index = 0
        self._pending_delay_x_quotas: dict[str, int] = {backend.name: 0 for backend in self.backends}
        self._max_active_latency_s: dict[str, float] = {backend.name: 0.0 for backend in self.backends}

    def choose_backend(self, estimated_cost: int) -> tuple[BackendState, int]:
        with self._lock:
            self._arrival_counter += 1
            if (
                self._delay_x_quota_every > 0
                and self._delay_x_quota_amount > 0
                and self._arrival_counter % self._delay_x_quota_every == 0
            ):
                quota_backend = self._select_quota_backend_locked()
                self._pending_delay_x_quotas[quota_backend.name] += self._delay_x_quota_amount
                self._max_active_latency_s[quota_backend.name] = 0.0

            selected = min(self.backends, key=lambda backend: backend.score())
            selected.inflight_requests += 1
            selected.inflight_cost += estimated_cost
            quota_amount = self._pending_delay_x_quotas.get(selected.name, 0)
            if quota_amount > 0:
                self._pending_delay_x_quotas[selected.name] = 0
            return selected, quota_amount

    def complete_backend(
        self,
        backend: BackendState,
        estimated_cost: int,
        latency_s: float,
        success: bool,
        error: str,
        reported_max_active_latency_s: float | None = None,
    ) -> None:
        with self._lock:
            backend.inflight_requests = max(0, backend.inflight_requests - 1)
            backend.inflight_cost = max(0, backend.inflight_cost - estimated_cost)
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
            if reported_max_active_latency_s is not None:
                backend_name = backend.name
                self._max_active_latency_s[backend_name] = max(0.0, float(reported_max_active_latency_s))

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": backend.name,
                    "base_url": backend.base_url,
                    "inflight_requests": backend.inflight_requests,
                    "inflight_cost": backend.inflight_cost,
                    "completed_requests": backend.completed_requests,
                    "failed_requests": backend.failed_requests,
                    "latency_ema_s": round(backend.latency_ema_s, 4),
                    "pending_delay_x_quota": self._pending_delay_x_quotas.get(backend.name, 0),
                    "max_active_latency_s": round(self._max_active_latency_s.get(backend.name, 0.0), 4),
                    "last_error": backend.last_error,
                }
                for backend in self.backends
            ]

    def _select_quota_backend_locked(self) -> BackendState:
        selected = max(
            self.backends,
            key=lambda backend: (
                self._max_active_latency_s.get(backend.name, 0.0),
                -backend.inflight_cost,
                -backend.inflight_requests,
                -self.backends.index(backend),
            ),
        )
        if self._max_active_latency_s.get(selected.name, 0.0) <= 0:
            selected = self.backends[self._next_delay_x_backend_index % len(self.backends)]
        self._next_delay_x_backend_index = (self._next_delay_x_backend_index + 1) % len(self.backends)
        return selected


def inject_delay_x_quota(path: str, payload: dict[str, Any], quota_amount: int) -> bytes:
    if quota_amount <= 0:
        return json.dumps(payload).encode("utf-8")

    payload_copy = dict(payload)
    if path == "/v1/chat/completions":
        extra_body = payload_copy.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        else:
            extra_body = dict(extra_body)
        extra_body[DELAY_X_DISPATCHER_QUOTA_EXTRA_BODY_KEY] = quota_amount
        payload_copy["extra_body"] = extra_body
    elif path == "/v1/images/generations":
        payload_copy[DELAY_X_DISPATCHER_QUOTA_EXTRA_BODY_KEY] = quota_amount

    return json.dumps(payload_copy).encode("utf-8")


def parse_delay_x_max_active_latency_s(headers: dict[str, str]) -> float | None:
    value = headers.get(DELAY_X_MAX_ACTIVE_LATENCY_HEADER)
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except Exception:
        return None


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
                backend = min(self.pool.backends, key=lambda item: item.score())
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
        started_at = time.perf_counter()
        try:
            backend, quota_amount = self.pool.choose_backend(estimated_cost)
        except Exception as exc:  # noqa: BLE001
            self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return

        forward_body = inject_delay_x_quota(self.path, payload, quota_amount)
        try:
            status, response_body, response_headers = proxy_request(
                "POST",
                f"{backend.base_url}{self.path}",
                forward_body,
                dict(self.headers.items()),
                self.request_timeout_s,
            )
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(
                backend,
                estimated_cost,
                latency_s,
                True,
                "",
                parse_delay_x_max_active_latency_s(response_headers),
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
                latency_s,
                False,
                f"HTTP {exc.code}",
                parse_delay_x_max_active_latency_s(response_headers),
            )
            self._write_response(exc.code, error_body, {"Content-Type": exc.headers.get_content_type()})
        except Exception as exc:  # noqa: BLE001
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(backend, estimated_cost, latency_s, False, str(exc))
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
        help="In lightweight dispatcher mode, generate one RR delay_x quota every N arrivals.",
    )
    parser.add_argument(
        "--delay-x-quota-amount",
        type=int,
        default=0,
        help="In lightweight dispatcher mode, number of RR delay_x quotas generated per quota event.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = BackendPool(
        args.backend_urls,
        delay_x_quota_every=args.delay_x_quota_every,
        delay_x_quota_amount=args.delay_x_quota_amount,
    )

    DispatcherHandler.pool = pool
    DispatcherHandler.request_timeout_s = args.request_timeout

    server = ThreadingHTTPServer((args.host, args.port), DispatcherHandler)
    print(f"Dispatcher listening on http://{args.host}:{args.port}")
    print("Scheduling policy: least-load + optional RR delay_x quota")
    for backend in pool.backends:
        print(f"  - {backend.name}: {backend.base_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
