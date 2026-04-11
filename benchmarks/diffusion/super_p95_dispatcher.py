# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import FormData, UploadFile
from vllm.logger import init_logger

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import (
    HEADER_SUPER_P95_ESTIMATED_SERVICE_S,
    HEADER_SUPER_P95_SACRIFICIAL,
    estimate_service_time_s,
    normalize_super_p95_hardware_profile,
    parse_super_p95_load_headers,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


@dataclass
class BackendState:
    name: str
    base_url: str
    hardware_profile: str = "910B2"
    normal_load_s: float = 0.0
    sacrificial_load_s: float = 0.0
    inflight_normal_requests: int = 0
    inflight_sacrificial_requests: int = 0
    latency_ema_s: float = 0.0

    def weighted_total_load_s(self, alpha: float) -> float:
        return self.normal_load_s + alpha * self.sacrificial_load_s

    def score_tuple(self, alpha: float) -> tuple[float, float, float, float, str]:
        inflight_score = self.inflight_normal_requests + alpha * self.inflight_sacrificial_requests
        return (
            self.weighted_total_load_s(alpha),
            inflight_score,
            self.latency_ema_s,
            self.normal_load_s,
            self.name,
        )


@dataclass(frozen=True)
class DispatchDecision:
    backend_index: int
    estimated_service_s: float
    is_sacrificial: bool


@dataclass(frozen=True)
class ManagedBackendSpec:
    device_id: str
    port: int
    base_url: str
    hardware_profile: str


@dataclass
class ManagedBackendProcess:
    spec: ManagedBackendSpec
    process: subprocess.Popen[str]
    log_file: Any


class ManagedBackendLauncher:
    def __init__(
        self,
        *,
        specs: list[ManagedBackendSpec],
        model: str,
        backend_args: list[str],
        backend_env: dict[str, str],
        device_env_var: str,
        health_timeout_s: float,
        health_poll_interval_s: float,
        log_dir: str,
        backend_command: str = "vllm-omni",
    ) -> None:
        self.specs = specs
        self.model = model
        self.backend_args = backend_args
        self.backend_env = backend_env
        self.device_env_var = device_env_var
        self.health_timeout_s = health_timeout_s
        self.health_poll_interval_s = health_poll_interval_s
        self.log_dir = Path(log_dir)
        self.backend_command = backend_command
        self._processes: list[ManagedBackendProcess] = []

    def start_all(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting %d managed super_p95 backends in parallel. Logs: %s",
            len(self.specs),
            self.log_dir,
        )
        try:
            for spec in self.specs:
                self._processes.append(self._start_one(spec))
            self._wait_until_healthy()
            logger.info("All managed super_p95 backends are healthy.")
        except Exception:
            self.stop_all()
            raise

    def stop_all(self) -> None:
        for managed in reversed(self._processes):
            if managed.process.poll() is None:
                managed.process.terminate()
        deadline = time.time() + 20.0
        for managed in reversed(self._processes):
            if managed.process.poll() is None:
                timeout_s = max(deadline - time.time(), 0.0)
                with suppress(subprocess.TimeoutExpired):
                    managed.process.wait(timeout=timeout_s)
            if managed.process.poll() is None:
                managed.process.kill()
                with suppress(subprocess.TimeoutExpired):
                    managed.process.wait(timeout=5.0)
            with suppress(Exception):
                managed.log_file.close()
        self._processes.clear()

    def _start_one(self, spec: ManagedBackendSpec) -> ManagedBackendProcess:
        env = os.environ.copy()
        env[self.device_env_var] = spec.device_id
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.update(self.backend_env)
        # v0.18 backend scheduler consumes the hardware profile from env.
        # Keep managed launch aligned with v0.16 semantics without depending on
        # a serve CLI flag that does not exist in this branch.
        env["VLLM_OMNI_SUPER_P95_HARDWARE_PROFILE"] = spec.hardware_profile
        log_path = self.log_dir / f"backend_{spec.port}.log"
        log_file = open(log_path, "a", encoding="utf-8")
        cmd = [
            self.backend_command,
            "serve",
            self.model,
            "--port",
            str(spec.port),
            *self.backend_args,
        ]
        logger.info(
            "Launching backend port=%s device_id=%s base_url=%s hardware_profile=%s log=%s cmd=%s",
            spec.port,
            spec.device_id,
            spec.base_url,
            spec.hardware_profile,
            log_path,
            shlex.join(cmd),
        )
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return ManagedBackendProcess(spec=spec, process=process, log_file=log_file)

    def _wait_until_healthy(self) -> None:
        deadline = time.time() + self.health_timeout_s
        pending = {managed.spec.port: managed for managed in self._processes}
        while pending:
            failed = [managed for managed in pending.values() if managed.process.poll() is not None]
            if failed:
                details = ", ".join(
                    f"{managed.spec.port}(device={managed.spec.device_id}, log={managed.log_file.name})"
                    for managed in failed
                )
                raise RuntimeError(f"super_p95 backend exited before becoming healthy: {details}")

            ready_ports = [port for port, managed in pending.items() if self._is_healthy(managed.spec.base_url)]
            for port in ready_ports:
                managed = pending.pop(port, None)
                if managed is not None:
                    logger.info(
                        "Managed backend healthy: port=%s device=%s url=%s log=%s",
                        managed.spec.port,
                        managed.spec.device_id,
                        managed.spec.base_url,
                        managed.log_file.name,
                    )

            if not pending:
                return
            if time.time() >= deadline:
                details = ", ".join(
                    f"{managed.spec.port}(device={managed.spec.device_id}, log={managed.log_file.name})"
                    for managed in pending.values()
                )
                raise TimeoutError(f"Timed out waiting for super_p95 backends to become healthy: {details}")
            time.sleep(self.health_poll_interval_s)

    @staticmethod
    def _is_healthy(base_url: str) -> bool:
        try:
            opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
            with opener.open(f"{base_url}/health", timeout=2.0) as response:
                return response.status == 200
        except (urllib_error.URLError, TimeoutError, ValueError):
            return False


class SuperP95Dispatcher:
    def __init__(
        self,
        backend_urls: list[str],
        *,
        backend_hardware_profiles: list[str] | None,
        quota_every: int,
        quota_amount: int,
        threshold_ratio: float,
        sacrificial_load_factor: float,
        request_timeout_s: float,
        backend_launcher: ManagedBackendLauncher | None = None,
    ) -> None:
        if not 1 <= len(backend_urls) <= 8:
            raise ValueError("super_p95 dispatcher supports 1 to 8 backends")
        hardware_profiles = _parse_backend_hardware_profiles(backend_hardware_profiles, len(backend_urls))
        self.backends = [
            BackendState(
                name=f"backend-{idx}",
                base_url=url.rstrip("/"),
                hardware_profile=hardware_profiles[idx],
            )
            for idx, url in enumerate(backend_urls)
        ]
        self.quota_every = max(quota_every, 1)
        self.quota_amount = max(quota_amount, 0)
        self.threshold_ratio = threshold_ratio
        self.sacrificial_load_factor = sacrificial_load_factor
        self.request_timeout_s = request_timeout_s

        self._lock = asyncio.Lock()
        self.arrival_counter = 0
        self.credits = 0
        self.global_max_service_s = 0.0
        self._client: httpx.AsyncClient | None = None
        self._backend_launcher = backend_launcher

    async def startup(self) -> None:
        if self._backend_launcher is not None:
            await asyncio.to_thread(self._backend_launcher.start_all)
        timeout = self.request_timeout_s if self.request_timeout_s > 0 else None
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._backend_launcher is not None:
            await asyncio.to_thread(self._backend_launcher.stop_all)

    async def dispatch_json(self, path: str, body: dict[str, Any], incoming_headers: dict[str, str]) -> Response:
        decision = await self._choose_backend(path, body)
        backend = self.backends[decision.backend_index]
        headers = self._build_forward_headers(incoming_headers, decision)
        assert self._client is not None

        start_time = time.perf_counter()
        try:
            response = await self._client.post(f"{backend.base_url}{path}", json=body, headers=headers)
        except Exception:
            elapsed_s = time.perf_counter() - start_time
            await self._mark_failed_response(decision, elapsed_s)
            raise

        elapsed_s = time.perf_counter() - start_time
        await self._apply_response_feedback(decision, response.headers, elapsed_s)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def dispatch_form(self, path: str, form: FormData, incoming_headers: dict[str, str]) -> Response:
        body = _form_to_estimation_dict(form)
        decision = await self._choose_backend(path, body)
        backend = self.backends[decision.backend_index]
        headers = self._build_forward_headers(incoming_headers, decision)
        headers.pop("content-type", None)

        data: dict[str, str] = {}
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                payload = await value.read()
                files.append(
                    (
                        key,
                        (
                            value.filename or "upload.bin",
                            payload,
                            value.content_type or "application/octet-stream",
                        ),
                    )
                )
            else:
                data[key] = str(value)

        assert self._client is not None
        start_time = time.perf_counter()
        try:
            response = await self._client.post(f"{backend.base_url}{path}", data=data, files=files or None, headers=headers)
        except Exception:
            elapsed_s = time.perf_counter() - start_time
            await self._mark_failed_response(decision, elapsed_s)
            raise

        elapsed_s = time.perf_counter() - start_time
        await self._apply_response_feedback(decision, response.headers, elapsed_s)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def proxy_get(self, path: str) -> Response:
        backend = self.backends[0]
        assert self._client is not None
        response = await self._client.get(f"{backend.base_url}{path}")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def proxy_delete(self, path: str) -> Response:
        backend = self.backends[0]
        assert self._client is not None
        response = await self._client.delete(f"{backend.base_url}{path}")
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=self._filter_response_headers(response.headers),
            media_type=response.headers.get("content-type"),
        )

    async def health(self) -> JSONResponse:
        assert self._client is not None
        statuses: list[dict[str, Any]] = []
        overall_healthy = True
        for backend in self.backends:
            try:
                response = await self._client.get(f"{backend.base_url}/health")
                healthy = response.status_code == 200
                detail = response.text
            except Exception as exc:
                healthy = False
                detail = str(exc)
            overall_healthy = overall_healthy and healthy
            statuses.append({"backend": backend.name, "url": backend.base_url, "healthy": healthy, "detail": detail})
        return JSONResponse(
            status_code=200 if overall_healthy else 503,
            content={"status": "healthy" if overall_healthy else "degraded", "backends": statuses},
        )

    async def _choose_backend(self, path: str, body: dict[str, Any]) -> DispatchDecision:
        estimated_service_s_by_backend = [
            self._estimate_service_s(path, body, backend.hardware_profile) for backend in self.backends
        ]
        estimated_service_s = min(estimated_service_s_by_backend)
        async with self._lock:
            self.arrival_counter += 1
            if self.arrival_counter % self.quota_every == 0:
                self.credits += self.quota_amount

            self.global_max_service_s = max(self.global_max_service_s, estimated_service_s)
            is_sacrificial = (
                self.credits > 0
                and self.global_max_service_s > 0.0
                and estimated_service_s >= self.threshold_ratio * self.global_max_service_s
            )
            if is_sacrificial:
                self.credits -= 1

            backend_index = min(
                range(len(self.backends)),
                key=lambda idx: self.backends[idx].score_tuple(self.sacrificial_load_factor),
            )
            backend = self.backends[backend_index]
            selected_estimated_service_s = estimated_service_s_by_backend[backend_index]
            if is_sacrificial:
                backend.sacrificial_load_s += selected_estimated_service_s
                backend.inflight_sacrificial_requests += 1
            else:
                backend.normal_load_s += selected_estimated_service_s
                backend.inflight_normal_requests += 1
            return DispatchDecision(
                backend_index=backend_index,
                estimated_service_s=selected_estimated_service_s,
                is_sacrificial=is_sacrificial,
            )

    async def _apply_response_feedback(
        self,
        decision: DispatchDecision,
        headers: httpx.Headers,
        elapsed_s: float,
    ) -> None:
        async with self._lock:
            backend = self.backends[decision.backend_index]
            self._update_latency_ema(backend, elapsed_s)
            self._dec_inflight(backend, decision.is_sacrificial)
            authoritative = parse_super_p95_load_headers(headers)
            if authoritative is not None:
                backend.normal_load_s = authoritative.normal_load_s
                backend.sacrificial_load_s = authoritative.sacrificial_load_s
                return
            self._fallback_remove_estimated_load(backend, decision)

    async def _mark_failed_response(self, decision: DispatchDecision, elapsed_s: float) -> None:
        async with self._lock:
            backend = self.backends[decision.backend_index]
            self._update_latency_ema(backend, elapsed_s)
            self._dec_inflight(backend, decision.is_sacrificial)
            self._fallback_remove_estimated_load(backend, decision)

    @staticmethod
    def _update_latency_ema(backend: BackendState, elapsed_s: float) -> None:
        if elapsed_s <= 0.0:
            return
        if backend.latency_ema_s <= 0.0:
            backend.latency_ema_s = elapsed_s
        else:
            backend.latency_ema_s = 0.9 * backend.latency_ema_s + 0.1 * elapsed_s

    @staticmethod
    def _dec_inflight(backend: BackendState, is_sacrificial: bool) -> None:
        if is_sacrificial:
            backend.inflight_sacrificial_requests = max(backend.inflight_sacrificial_requests - 1, 0)
        else:
            backend.inflight_normal_requests = max(backend.inflight_normal_requests - 1, 0)

    @staticmethod
    def _fallback_remove_estimated_load(backend: BackendState, decision: DispatchDecision) -> None:
        if decision.is_sacrificial:
            backend.sacrificial_load_s = max(backend.sacrificial_load_s - decision.estimated_service_s, 0.0)
        else:
            backend.normal_load_s = max(backend.normal_load_s - decision.estimated_service_s, 0.0)

    @staticmethod
    def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
        blocked = {"content-length", "transfer-encoding", "connection", "content-encoding"}
        return {key: value for key, value in headers.items() if key.lower() not in blocked}

    @staticmethod
    def _build_forward_headers(incoming_headers: dict[str, str], decision: DispatchDecision) -> dict[str, str]:
        blocked = {"host", "content-length"}
        headers = {key: value for key, value in incoming_headers.items() if key.lower() not in blocked}
        headers[HEADER_SUPER_P95_ESTIMATED_SERVICE_S] = f"{decision.estimated_service_s:.6f}"
        if decision.is_sacrificial:
            headers[HEADER_SUPER_P95_SACRIFICIAL] = "1"
        else:
            headers.pop(HEADER_SUPER_P95_SACRIFICIAL, None)
        return headers

    @staticmethod
    def _estimate_service_s(path: str, body: dict[str, Any], hardware_profile: str) -> float:
        if path == "/v1/chat/completions":
            extra_body = body.get("extra_body") or {}
            return _estimate_service_s_from_values(
                width=extra_body.get("width"),
                height=extra_body.get("height"),
                num_inference_steps=extra_body.get("num_inference_steps"),
                num_frames=extra_body.get("num_frames"),
                hardware_profile=hardware_profile,
            )
        if path == "/v1/images/generations":
            width, height = _parse_size(body.get("size"))
            return _estimate_service_s_from_values(
                width=width,
                height=height,
                num_inference_steps=body.get("num_inference_steps"),
                num_frames=body.get("num_frames"),
                hardware_profile=hardware_profile,
            )
        if path == "/v1/videos":
            width, height = _parse_size(body.get("size"))
            return _estimate_service_s_from_values(
                width=body.get("width", width),
                height=body.get("height", height),
                num_inference_steps=body.get("num_inference_steps"),
                num_frames=body.get("num_frames"),
                hardware_profile=hardware_profile,
            )
        raise HTTPException(status_code=400, detail=f"Unsupported super_p95 path: {path}")


def _parse_size(size: Any) -> tuple[int | None, int | None]:
    if not isinstance(size, str) or "x" not in size.lower():
        return None, None
    width_str, height_str = size.lower().split("x", 1)
    try:
        return int(width_str), int(height_str)
    except ValueError:
        return None, None


def _estimate_service_s_from_values(
    *,
    width: Any,
    height: Any,
    num_inference_steps: Any,
    num_frames: Any,
    hardware_profile: str,
) -> float:
    sampling_params = OmniDiffusionSamplingParams(
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        num_inference_steps=int(num_inference_steps) if num_inference_steps is not None else 25,
        num_frames=int(num_frames) if num_frames is not None else 1,
    )
    request = OmniDiffusionRequest(prompts=["super_p95"], sampling_params=sampling_params, request_ids=["super_p95"])
    return estimate_service_time_s(request, hardware_profile=hardware_profile)


def _form_to_estimation_dict(form: FormData) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            continue
        body[key] = value
    return body


def _parse_backend_hardware_profiles(raw: list[str] | None | str, num_backends: int) -> list[str]:
    if raw is None:
        return [normalize_super_p95_hardware_profile(None)] * num_backends
    if isinstance(raw, str):
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        parsed = [item.strip() for item in raw if item.strip()]
    if len(parsed) == 1 and num_backends > 1:
        parsed = parsed * num_backends
    if len(parsed) != num_backends:
        raise ValueError(f"--backend-hardware-profiles must provide exactly {num_backends} entries")
    return [normalize_super_p95_hardware_profile(item) for item in parsed]


def build_app(dispatcher: SuperP95Dispatcher) -> FastAPI:
    app = FastAPI(title="super_p95 dispatcher")

    @app.on_event("startup")
    async def _startup() -> None:
        await dispatcher.startup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await dispatcher.shutdown()

    @app.get("/health")
    async def health() -> JSONResponse:
        return await dispatcher.health()

    @app.get("/v1/models")
    async def models() -> Response:
        return await dispatcher.proxy_get("/v1/models")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        return await dispatcher.dispatch_json("/v1/chat/completions", body, dict(request.headers))

    @app.post("/v1/images/generations")
    async def image_generations(request: Request) -> Response:
        body = await request.json()
        return await dispatcher.dispatch_json("/v1/images/generations", body, dict(request.headers))

    @app.post("/v1/videos")
    async def videos(request: Request) -> Response:
        form = await request.form()
        return await dispatcher.dispatch_form("/v1/videos", form, dict(request.headers))

    @app.get("/v1/videos/{video_id}")
    async def retrieve_video(video_id: str) -> Response:
        return await dispatcher.proxy_get(f"/v1/videos/{video_id}")

    @app.delete("/v1/videos/{video_id}")
    async def delete_video(video_id: str) -> Response:
        return await dispatcher.proxy_delete(f"/v1/videos/{video_id}")

    @app.get("/v1/videos/{video_id}/content")
    async def retrieve_video_content(video_id: str) -> Response:
        return await dispatcher.proxy_get(f"/v1/videos/{video_id}/content")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="super_p95 dispatcher for diffusion servers")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--backend-url",
        dest="backend_urls",
        action="append",
        help="Backend base URL. Repeat 1 to 8 times, e.g. http://127.0.0.1:8091",
    )
    parser.add_argument("--num-servers", type=int, help="Number of managed backend servers to launch (1-8).")
    parser.add_argument(
        "--device-ids",
        help="Comma-separated device ids for managed launch, e.g. 0,1,2,3. Defaults to 0..num_servers-1.",
    )
    parser.add_argument("--model", help="Model name for managed backend launch, e.g. Qwen/Qwen-Image.")
    parser.add_argument("--backend-host", default="127.0.0.1", help="Host used for managed backend URLs.")
    parser.add_argument("--backend-start-port", type=int, default=8091, help="Starting port for managed backends.")
    parser.add_argument(
        "--backend-hardware-profiles",
        help="Comma-separated hardware profiles for backends, e.g. 910B2,910B3.",
    )
    parser.add_argument(
        "--backend-args",
        action="append",
        default=["--omni"],
        help="Extra CLI args appended to each managed backend command.",
    )
    parser.add_argument(
        "--backend-env",
        action="append",
        default=[],
        help="Extra environment for managed backends in KEY=VALUE form. Repeat as needed.",
    )
    parser.add_argument(
        "--device-env-var",
        default="ASCEND_RT_VISIBLE_DEVICES",
        help="Environment variable used to pin a managed backend to a single device.",
    )
    parser.add_argument(
        "--backend-health-timeout-s",
        type=float,
        default=900.0,
        help="How long dispatcher startup waits for managed backends to pass /health.",
    )
    parser.add_argument(
        "--backend-health-poll-interval-s",
        type=float,
        default=5.0,
        help="Polling interval while waiting for managed backends to pass /health.",
    )
    parser.add_argument(
        "--backend-log-dir",
        default="/tmp/super_p95_backends",
        help="Directory for managed backend stdout/stderr logs.",
    )
    parser.add_argument("--quota-every", type=int, default=20)
    parser.add_argument("--quota-amount", type=int, default=1)
    parser.add_argument("--threshold-ratio", type=float, default=0.8)
    parser.add_argument("--sacrificial-load-factor", type=float, default=0.1)
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=0.0,
        help="Dispatcher-to-backend request timeout in seconds. Set to 0 to disable timeout entirely.",
    )
    return parser.parse_args()


def _parse_backend_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --backend-env value {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --backend-env value {item!r}; key cannot be empty")
        env[key] = value
    return env


def _parse_device_ids(device_ids: str | None, num_servers: int) -> list[str]:
    if device_ids is None:
        return [str(idx) for idx in range(num_servers)]
    parsed = [item.strip() for item in device_ids.split(",") if item.strip()]
    if len(parsed) != num_servers:
        raise ValueError(f"--device-ids must provide exactly {num_servers} entries")
    return parsed


def _build_managed_specs(
    host: str,
    start_port: int,
    device_ids: list[str],
    hardware_profiles: list[str],
) -> list[ManagedBackendSpec]:
    return [
        ManagedBackendSpec(
            device_id=device_id,
            port=start_port + idx,
            base_url=f"http://{host}:{start_port + idx}",
            hardware_profile=hardware_profiles[idx],
        )
        for idx, device_id in enumerate(device_ids)
    ]


def build_dispatcher_from_args(args: argparse.Namespace) -> SuperP95Dispatcher:
    manual_urls = [url.rstrip("/") for url in (args.backend_urls or [])]
    use_managed = args.num_servers is not None or args.model is not None or args.device_ids is not None
    backend_hardware_profiles_arg = getattr(args, "backend_hardware_profiles", None)

    if manual_urls and use_managed:
        raise ValueError("Choose either --backend-url or managed launch args, not both")

    backend_launcher = None
    if manual_urls:
        backend_urls = manual_urls
    else:
        if args.num_servers is None or args.model is None:
            raise ValueError("Managed launch requires both --num-servers and --model")
        if not 1 <= args.num_servers <= 8:
            raise ValueError("--num-servers must be between 1 and 8")
        device_ids = _parse_device_ids(args.device_ids, args.num_servers)
        hardware_profiles = _parse_backend_hardware_profiles(backend_hardware_profiles_arg, args.num_servers)
        specs = _build_managed_specs(args.backend_host, args.backend_start_port, device_ids, hardware_profiles)
        backend_urls = [spec.base_url for spec in specs]
        backend_args: list[str] = []
        for raw in args.backend_args:
            backend_args.extend(shlex.split(raw))
        backend_launcher = ManagedBackendLauncher(
            specs=specs,
            model=args.model,
            backend_args=backend_args,
            backend_env=_parse_backend_env(args.backend_env),
            device_env_var=args.device_env_var,
            health_timeout_s=args.backend_health_timeout_s,
            health_poll_interval_s=args.backend_health_poll_interval_s,
            log_dir=args.backend_log_dir,
        )

    return SuperP95Dispatcher(
        backend_urls=backend_urls,
        backend_hardware_profiles=_parse_backend_hardware_profiles(backend_hardware_profiles_arg, len(backend_urls)),
        quota_every=args.quota_every,
        quota_amount=args.quota_amount,
        threshold_ratio=args.threshold_ratio,
        sacrificial_load_factor=args.sacrificial_load_factor,
        request_timeout_s=args.request_timeout_s,
        backend_launcher=backend_launcher,
    )


def main() -> None:
    args = parse_args()
    dispatcher = build_dispatcher_from_args(args)
    uvicorn.run(build_app(dispatcher), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
