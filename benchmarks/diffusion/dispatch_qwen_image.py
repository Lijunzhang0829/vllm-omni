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
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_REQUEST_TIMEOUT_S = 60 * 60
DEFAULT_SECONDS_PER_COST_S = 33.9 / float(1024 * 1024 * 25)


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
class PendingRequest:
    request_id: int
    path: str
    body: bytes
    headers: dict[str, str]
    payload: dict[str, Any]
    estimated_cost: int
    estimated_service_s: float
    arrival_time_s: float
    sacrificial: bool = False
    started_at_s: float | None = None
    completed_at_s: float | None = None
    backend_name: str = ""
    response_status: int = 0
    response_body: bytes = b""
    response_headers: dict[str, str] = field(default_factory=dict)
    error_message: str = ""
    done_event: threading.Event = field(default_factory=threading.Event)


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
    active_request: PendingRequest | None = None
    queued_requests: list[PendingRequest] = field(default_factory=list)

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
    def __init__(self, backend_urls: list[str]):
        self.backends = [
            BackendState(name=f"server-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(backend_urls)
        ]
        self._lock = threading.Lock()

    def choose_backend(self, estimated_cost: int) -> BackendState:
        with self._lock:
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


class QueuedBackendPool:
    def __init__(
        self,
        backend_urls: list[str],
        scheduling_policy: str,
        request_timeout_s: int,
        delay_x_quota_every: int,
        delay_x_quota_amount: int,
        delay_x_tail_penalty: float,
        delay_x_split_dispatch_loads: bool,
        initial_seconds_per_cost_s: float = DEFAULT_SECONDS_PER_COST_S,
    ) -> None:
        self.backends = [
            BackendState(name=f"server-{idx}", base_url=url.rstrip("/")) for idx, url in enumerate(backend_urls)
        ]
        self.scheduling_policy = scheduling_policy
        self.request_timeout_s = request_timeout_s
        self.delay_x_quota_every = delay_x_quota_every
        self.delay_x_quota_amount = delay_x_quota_amount
        self.delay_x_tail_penalty = delay_x_tail_penalty
        self.delay_x_split_dispatch_loads = delay_x_split_dispatch_loads

        self._condition = threading.Condition()
        self._shutdown = False
        self._request_counter = 0
        self._arrival_counter = 0
        self._next_delay_x_backend_index = 0
        self._seconds_per_cost_ema = initial_seconds_per_cost_s
        self._workers = [
            threading.Thread(target=self._backend_loop, args=(backend,), daemon=True) for backend in self.backends
        ]
        for worker in self._workers:
            worker.start()

    def _estimate_service_s_locked(self, estimated_cost: int) -> float:
        return max(0.001, estimated_cost * self._seconds_per_cost_ema)

    def _remaining_service_s_locked(self, request: PendingRequest, now_s: float) -> float:
        if request.started_at_s is None:
            return request.estimated_service_s
        return max(0.0, request.estimated_service_s - (now_s - request.started_at_s))

    def _predicted_latency_s_locked(self, request: PendingRequest, now_s: float) -> float:
        return max(0.0, now_s - request.arrival_time_s) + self._remaining_service_s_locked(request, now_s)

    def _normal_load_s_locked(self, backend: BackendState, now_s: float) -> float:
        total = 0.0
        if backend.active_request is not None and not backend.active_request.sacrificial:
            total += self._remaining_service_s_locked(backend.active_request, now_s)
        for request in backend.queued_requests:
            if not request.sacrificial:
                total += request.estimated_service_s
        return total

    def _total_load_s_locked(self, backend: BackendState, now_s: float) -> float:
        total = 0.0
        if backend.active_request is not None:
            total += self._remaining_service_s_locked(backend.active_request, now_s)
        for request in backend.queued_requests:
            total += request.estimated_service_s
        return total

    def _request_priority_locked(self, request: PendingRequest, now_s: float) -> tuple[float, float, int]:
        if self.scheduling_policy == "delay-x":
            predicted_latency_s = self._predicted_latency_s_locked(request, now_s)
            if request.sacrificial:
                predicted_latency_s /= self.delay_x_tail_penalty
            return (-predicted_latency_s, request.arrival_time_s, request.request_id)

        remaining_service_s = self._remaining_service_s_locked(request, now_s)
        return (remaining_service_s, request.arrival_time_s, request.request_id)

    def _select_next_request_locked(self, backend: BackendState, now_s: float) -> PendingRequest | None:
        if not backend.queued_requests:
            return None
        next_request = min(
            backend.queued_requests,
            key=lambda request: self._request_priority_locked(request, now_s),
        )
        backend.queued_requests.remove(next_request)
        return next_request

    def _delay_x_base_priority_locked(self, request: PendingRequest, now_s: float) -> tuple[float, float, int]:
        predicted_latency_s = self._predicted_latency_s_locked(request, now_s)
        return (-predicted_latency_s, request.arrival_time_s, request.request_id)

    def _select_delay_x_sacrificial_candidate_locked(
        self,
        backend: BackendState,
        now_s: float,
    ) -> PendingRequest | None:
        candidates = [request for request in backend.queued_requests if not request.sacrificial]
        if not candidates:
            return None

        ordered = sorted(candidates, key=lambda request: self._delay_x_base_priority_locked(request, now_s))
        active_remaining_s = 0.0
        if backend.active_request is not None:
            active_remaining_s = self._remaining_service_s_locked(backend.active_request, now_s)

        cumulative_wait_s = active_remaining_s
        selected: PendingRequest | None = None
        selected_latency_s = -1.0
        for request in ordered:
            predicted_completion_latency_s = (
                max(0.0, now_s - request.arrival_time_s) + cumulative_wait_s + request.estimated_service_s
            )
            if predicted_completion_latency_s > selected_latency_s:
                selected = request
                selected_latency_s = predicted_completion_latency_s
            cumulative_wait_s += request.estimated_service_s
        return selected

    def _issue_delay_x_quota_locked(self) -> None:
        if not self.backends:
            return
        backend = self.backends[self._next_delay_x_backend_index % len(self.backends)]
        self._next_delay_x_backend_index = (self._next_delay_x_backend_index + 1) % len(self.backends)
        candidate = self._select_delay_x_sacrificial_candidate_locked(backend, time.perf_counter())
        if candidate is not None:
            candidate.sacrificial = True

    def _select_dispatch_backend_locked(self, sacrificial: bool) -> BackendState:
        healthy_backends = [backend for backend in self.backends if backend.healthy]
        if not healthy_backends:
            raise RuntimeError("No healthy backends available")

        now_s = time.perf_counter()
        if self.scheduling_policy == "delay-x" and self.delay_x_split_dispatch_loads:
            return min(
                healthy_backends,
                key=lambda backend: (
                    self._total_load_s_locked(backend, now_s)
                    if sacrificial
                    else self._normal_load_s_locked(backend, now_s),
                    len(backend.queued_requests) + (1 if backend.active_request is not None else 0),
                    backend.name,
                ),
            )

        return min(
            healthy_backends,
            key=lambda backend: (
                self._total_load_s_locked(backend, now_s),
                len(backend.queued_requests) + (1 if backend.active_request is not None else 0),
                backend.name,
            ),
        )

    def submit_request(
        self,
        path: str,
        body: bytes,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> PendingRequest:
        estimated_cost = estimate_request_cost(payload)
        with self._condition:
            if self.scheduling_policy == "delay-x":
                self._arrival_counter += 1
                if self.delay_x_quota_every > 0 and self._arrival_counter % self.delay_x_quota_every == 0:
                    for _ in range(self.delay_x_quota_amount):
                        self._issue_delay_x_quota_locked()
            sacrificial = False
            backend = self._select_dispatch_backend_locked(sacrificial)
            self._request_counter += 1
            request = PendingRequest(
                request_id=self._request_counter,
                path=path,
                body=body,
                headers=headers,
                payload=payload,
                estimated_cost=estimated_cost,
                estimated_service_s=self._estimate_service_s_locked(estimated_cost),
                arrival_time_s=time.perf_counter(),
                sacrificial=sacrificial,
                backend_name=backend.name,
            )
            backend.queued_requests.append(request)
            self._condition.notify_all()
            return request

    def _backend_loop(self, backend: BackendState) -> None:
        while True:
            request: PendingRequest | None = None
            with self._condition:
                while True:
                    if self._shutdown:
                        return
                    if backend.healthy and backend.active_request is None:
                        request = self._select_next_request_locked(backend, time.perf_counter())
                        if request is not None:
                            backend.active_request = request
                            backend.inflight_requests = 1
                            backend.inflight_cost = request.estimated_cost
                            request.started_at_s = time.perf_counter()
                            break
                    self._condition.wait()

            assert request is not None
            latency_s = 0.0
            success = False
            error = ""
            try:
                status, response_body, response_headers = proxy_request(
                    "POST",
                    f"{backend.base_url}{request.path}",
                    request.body,
                    request.headers,
                    self.request_timeout_s,
                )
                latency_s = time.perf_counter() - request.started_at_s
                success = True
                request.response_status = status
                request.response_body = response_body
                request.response_headers = response_headers
            except urllib.error.HTTPError as exc:
                latency_s = time.perf_counter() - request.started_at_s
                error = f"HTTP {exc.code}"
                request.response_status = exc.code
                request.response_body = exc.read()
                request.response_headers = {"Content-Type": exc.headers.get_content_type()}
            except Exception as exc:  # noqa: BLE001
                latency_s = time.perf_counter() - request.started_at_s
                error = str(exc)
                request.error_message = error
                request.response_status = HTTPStatus.BAD_GATEWAY
                request.response_body = json.dumps({"error": f"{backend.name} failed: {error}"}).encode("utf-8")
                request.response_headers = {"Content-Type": "application/json"}

            with self._condition:
                request.completed_at_s = time.perf_counter()
                request.done_event.set()
                backend.active_request = None
                backend.inflight_requests = 0
                backend.inflight_cost = 0
                backend.last_error = error
                if success:
                    backend.completed_requests += 1
                    backend.healthy = True
                else:
                    backend.failed_requests += 1
                if latency_s > 0 and request.estimated_cost > 0:
                    backend.latency_ema_s = latency_s if backend.latency_ema_s <= 0 else 0.8 * backend.latency_ema_s + 0.2 * latency_s
                    observed_seconds_per_cost = latency_s / float(request.estimated_cost)
                    self._seconds_per_cost_ema = 0.9 * self._seconds_per_cost_ema + 0.1 * observed_seconds_per_cost
                self._condition.notify_all()

    def set_health(self, backend: BackendState, healthy: bool, error: str = "") -> None:
        with self._condition:
            backend.healthy = healthy
            backend.last_error = error
            backend.last_healthcheck_ts = time.time()
            self._condition.notify_all()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            now_s = time.perf_counter()
            return [
                {
                    "name": backend.name,
                    "base_url": backend.base_url,
                    "healthy": backend.healthy,
                    "inflight_requests": backend.inflight_requests,
                    "inflight_cost": backend.inflight_cost,
                    "queued_requests": len(backend.queued_requests),
                    "normal_load_s": round(self._normal_load_s_locked(backend, now_s), 4),
                    "total_load_s": round(self._total_load_s_locked(backend, now_s), 4),
                    "active_request_id": None if backend.active_request is None else backend.active_request.request_id,
                    "completed_requests": backend.completed_requests,
                    "failed_requests": backend.failed_requests,
                    "latency_ema_s": round(backend.latency_ema_s, 4),
                    "last_error": backend.last_error,
                    "last_healthcheck_ts": backend.last_healthcheck_ts,
                }
                for backend in self.backends
            ]

    def pick_backend_for_get(self) -> BackendState:
        with self._condition:
            healthy_backends = [backend for backend in self.backends if backend.healthy]
            if not healthy_backends:
                raise RuntimeError("No healthy backends available")
            now_s = time.perf_counter()
            return min(
                healthy_backends,
                key=lambda backend: (
                    self._total_load_s_locked(backend, now_s),
                    backend.name,
                ),
            )


class DispatcherHandler(BaseHTTPRequestHandler):
    pool: BackendPool | QueuedBackendPool
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
            try:
                backend = self.pool.pick_backend_for_get() if isinstance(self.pool, QueuedBackendPool) else min(
                    [item for item in self.pool.backends if item.healthy],
                    key=lambda item: item.score(),
                )
            except Exception as exc:  # noqa: BLE001
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return
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
            except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            self._json_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc}")
            return

        if isinstance(self.pool, QueuedBackendPool):
            try:
                request = self.pool.submit_request(
                    path=self.path,
                    body=body,
                    headers=dict(self.headers.items()),
                    payload=payload,
                )
            except Exception as exc:  # noqa: BLE001
                self._json_error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
                return

            request.done_event.wait()
            self._write_response(request.response_status, request.response_body, request.response_headers)
            return

        estimated_cost = estimate_request_cost(payload)
        started_at = time.perf_counter()
        try:
            backend = self.pool.choose_backend(estimated_cost)
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            latency_s = time.perf_counter() - started_at
            self.pool.complete_backend(backend, estimated_cost, latency_s, False, str(exc))
            self._json_error(HTTPStatus.BAD_GATEWAY, f"{backend.name} failed: {exc}")

    def log_message(self, format: str, *args: Any) -> None:
        return


def healthcheck_loop(pool: BackendPool | QueuedBackendPool, interval_s: int, timeout_s: int) -> None:
    while True:
        for backend in pool.backends:
            try:
                status, _, _ = proxy_request("GET", f"{backend.base_url}/health", None, {}, timeout_s)
                pool.set_health(backend, status == HTTPStatus.OK, "" if status == HTTPStatus.OK else f"HTTP {status}")
            except Exception as exc:  # noqa: BLE001
                pool.set_health(backend, False, str(exc))
        time.sleep(interval_s)


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
    parser.add_argument("--healthcheck-interval", type=int, default=5, help="Healthcheck interval in seconds.")
    parser.add_argument("--healthcheck-timeout", type=int, default=5, help="Healthcheck timeout in seconds.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_S,
        help="Proxy request timeout in seconds.",
    )
    parser.add_argument(
        "--scheduling-policy",
        choices=["least-load", "shortest-remaining", "delay-x", "delay_x"],
        default="least-load",
        help="Dispatcher scheduling policy. `least-load` preserves the original behavior. "
        "`shortest-remaining` and `delay-x` use dispatcher-managed local queues.",
    )
    parser.add_argument(
        "--delay-x-quota-every",
        type=int,
        default=20,
        help="Generate delay-x sacrificial quota every N arrivals.",
    )
    parser.add_argument(
        "--delay-x-quota-amount",
        type=int,
        default=1,
        help="Delay-x sacrificial quota generated per period.",
    )
    parser.add_argument(
        "--delay-x-tail-penalty",
        type=float,
        default=100.0,
        help="Divide sacrificial predicted-delay score by this factor so it runs later but still preserves order.",
    )
    parser.add_argument(
        "--delay-x-split-dispatch-loads",
        action="store_true",
        help="Dispatch normal requests by normal-load and sacrificial requests by total-load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized_policy = "delay-x" if args.scheduling_policy == "delay_x" else args.scheduling_policy
    if args.scheduling_policy == "least-load":
        pool: BackendPool | QueuedBackendPool = BackendPool(args.backend_urls)
    else:
        pool = QueuedBackendPool(
            backend_urls=args.backend_urls,
            scheduling_policy=normalized_policy,
            request_timeout_s=args.request_timeout,
            delay_x_quota_every=args.delay_x_quota_every,
            delay_x_quota_amount=args.delay_x_quota_amount,
            delay_x_tail_penalty=args.delay_x_tail_penalty,
            delay_x_split_dispatch_loads=args.delay_x_split_dispatch_loads,
        )

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
    print(f"Scheduling policy: {normalized_policy}")
    for backend in pool.backends:
        print(f"  - {backend.name}: {backend.base_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
