import asyncio
from argparse import Namespace

import pytest

from benchmarks.diffusion.super_p95_dispatcher import (
    ManagedBackendLauncher,
    ManagedBackendSpec,
    SuperP95Dispatcher,
    build_dispatcher_from_args,
)
from vllm_omni.diffusion.super_p95 import (
    HEADER_SUPER_P95_ESTIMATED_SERVICE_S,
    HEADER_SUPER_P95_SACRIFICIAL,
)


def test_dispatcher_marks_large_request_sacrificial_after_credit_replenish():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=2,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )

    async def _run():
        first = await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 512, "height": 512, "num_inference_steps": 20}},
        )
        second = await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 1536, "height": 1536, "num_inference_steps": 35}},
        )
        return first, second

    first, second = asyncio.run(_run())

    assert first.is_sacrificial is False
    assert second.is_sacrificial is True
    assert dispatcher.credits == 0


def test_dispatcher_routes_to_lower_weighted_load_backend():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )
    dispatcher.backends[0].normal_load_s = 40.0
    dispatcher.backends[1].normal_load_s = 5.0

    async def _run():
        return await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 768, "height": 768, "num_inference_steps": 20}},
        )

    decision = asyncio.run(_run())

    assert decision.backend_index == 1


def test_build_forward_headers_includes_super_p95_metadata():
    headers = SuperP95Dispatcher._build_forward_headers(
        {"Authorization": "Bearer test", "Host": "ignored"},
        decision=type("Decision", (), {"estimated_service_s": 12.5, "is_sacrificial": True})(),
    )

    assert headers[HEADER_SUPER_P95_ESTIMATED_SERVICE_S] == "12.500000"
    assert headers[HEADER_SUPER_P95_SACRIFICIAL] == "1"
    assert "Host" not in headers


def test_build_dispatcher_from_args_managed_launch_defaults():
    args = Namespace(
        backend_urls=None,
        num_servers=2,
        device_ids=None,
        model="Qwen/Qwen-Image",
        backend_host="127.0.0.1",
        backend_start_port=8091,
        backend_hardware_profiles=None,
        backend_args="--omni --vae-use-slicing --vae-use-tiling",
        backend_env=["VLLM_PLUGINS=ascend"],
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        backend_health_timeout_s=900.0,
        backend_health_poll_interval_s=5.0,
        backend_log_dir="/tmp/super_p95_test_logs",
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=600.0,
    )

    dispatcher = build_dispatcher_from_args(args)

    assert [backend.base_url for backend in dispatcher.backends] == [
        "http://127.0.0.1:8091",
        "http://127.0.0.1:8092",
    ]
    assert dispatcher._backend_launcher is not None
    assert [spec.device_id for spec in dispatcher._backend_launcher.specs] == ["0", "1"]
    assert [spec.hardware_profile for spec in dispatcher._backend_launcher.specs] == ["910B2", "910B2"]
    assert dispatcher._backend_launcher.backend_args == ["--omni", "--vae-use-slicing", "--vae-use-tiling"]
    assert dispatcher._backend_launcher.backend_env["VLLM_PLUGINS"] == "ascend"


def test_build_dispatcher_from_args_rejects_manual_and_managed_mix():
    args = Namespace(
        backend_urls=["http://127.0.0.1:8091"],
        num_servers=2,
        device_ids="0,1",
        model="Qwen/Qwen-Image",
        backend_host="127.0.0.1",
        backend_start_port=8091,
        backend_hardware_profiles=None,
        backend_args="--omni",
        backend_env=[],
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        backend_health_timeout_s=900.0,
        backend_health_poll_interval_s=5.0,
        backend_log_dir="/tmp/super_p95_test_logs",
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=600.0,
    )

    with pytest.raises(ValueError, match="Choose either --backend-url or managed launch args"):
        build_dispatcher_from_args(args)


def test_managed_backend_launcher_builds_expected_command_and_env(tmp_path, monkeypatch):
    popen_calls = []

    class FakeProcess:
        def __init__(self):
            self._returncode = None

        def poll(self):
            return self._returncode

        def terminate(self):
            self._returncode = 0

        def wait(self, timeout=None):
            self._returncode = 0
            return 0

        def kill(self):
            self._returncode = -9

    def fake_popen(cmd, env, stdout, stderr, text):
        popen_calls.append(
            {
                "cmd": cmd,
                "env_device": env["ASCEND_RT_VISIBLE_DEVICES"],
                "env_plugin": env["VLLM_PLUGINS"],
                "stdout_name": stdout.name,
                "stderr": stderr,
                "text": text,
            }
        )
        return FakeProcess()

    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        ManagedBackendLauncher,
        "_wait_until_healthy",
        lambda self: None,
    )

    launcher = ManagedBackendLauncher(
        specs=[
            ManagedBackendSpec(device_id="0", port=8091, base_url="http://127.0.0.1:8091", hardware_profile="910B2"),
            ManagedBackendSpec(device_id="1", port=8092, base_url="http://127.0.0.1:8092", hardware_profile="910B3"),
        ],
        model="Qwen/Qwen-Image",
        backend_args=["--omni", "--vae-use-slicing", "--vae-use-tiling"],
        backend_env={"VLLM_PLUGINS": "ascend"},
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        health_timeout_s=10.0,
        health_poll_interval_s=0.1,
        log_dir=str(tmp_path),
    )

    launcher.start_all()
    launcher.stop_all()

    assert [call["cmd"] for call in popen_calls] == [
        [
            "vllm",
            "serve",
            "Qwen/Qwen-Image",
            "--port",
            "8091",
            "--super-p95-hardware-profile",
            "910B2",
            "--omni",
            "--vae-use-slicing",
            "--vae-use-tiling",
        ],
        [
            "vllm",
            "serve",
            "Qwen/Qwen-Image",
            "--port",
            "8092",
            "--super-p95-hardware-profile",
            "910B3",
            "--omni",
            "--vae-use-slicing",
            "--vae-use-tiling",
        ],
    ]
    assert [call["env_device"] for call in popen_calls] == ["0", "1"]
    assert all(call["env_plugin"] == "ascend" for call in popen_calls)


def test_build_dispatcher_from_args_applies_backend_hardware_profiles():
    args = Namespace(
        backend_urls=None,
        num_servers=2,
        device_ids="0,1",
        model="Qwen/Qwen-Image",
        backend_host="127.0.0.1",
        backend_start_port=8091,
        backend_hardware_profiles="910B2,910B3",
        backend_args="--omni",
        backend_env=[],
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        backend_health_timeout_s=900.0,
        backend_health_poll_interval_s=5.0,
        backend_log_dir="/tmp/super_p95_test_logs",
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=600.0,
    )

    dispatcher = build_dispatcher_from_args(args)

    assert [backend.hardware_profile for backend in dispatcher.backends] == ["910B2", "910B3"]


def test_dispatcher_estimates_service_time_per_backend_profile():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=["910B2", "910B3"],
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )
    dispatcher.backends[0].normal_load_s = 30.0
    dispatcher.backends[1].normal_load_s = 0.0

    async def _run():
        return await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 1024, "height": 1024, "num_inference_steps": 25}},
        )

    decision = asyncio.run(_run())

    assert decision.backend_index == 1
    assert decision.estimated_service_s == pytest.approx(22.11)
