import asyncio
import io
from argparse import Namespace

import pytest
from starlette.datastructures import UploadFile

from benchmarks.diffusion.super_p95_dispatcher import (
    ManagedBackendLauncher,
    ManagedBackendSpec,
    SuperP95Dispatcher,
    _form_to_estimation_dict,
    _parse_npu_smi_bus_ids,
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


def test_normal_request_uses_weighted_total_load_when_routing():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )
    dispatcher.backends[0].normal_load_s = 5.0
    dispatcher.backends[0].sacrificial_load_s = 1000.0
    dispatcher.backends[1].normal_load_s = 6.0
    dispatcher.backends[1].sacrificial_load_s = 0.0

    async def _run():
        return await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 768, "height": 768, "num_inference_steps": 20}},
        )

    decision = asyncio.run(_run())

    assert decision.is_sacrificial is False
    assert decision.backend_index == 1


def test_sacrificial_request_uses_weighted_total_load_when_routing():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0", "http://backend-1"],
        backend_hardware_profiles=None,
        quota_every=1,
        quota_amount=1,
        threshold_ratio=0.0,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )
    dispatcher.backends[0].normal_load_s = 5.0
    dispatcher.backends[0].sacrificial_load_s = 1000.0
    dispatcher.backends[1].normal_load_s = 6.0
    dispatcher.backends[1].sacrificial_load_s = 0.0

    async def _run():
        return await dispatcher._choose_backend(
            "/v1/chat/completions",
            {"extra_body": {"width": 1536, "height": 1536, "num_inference_steps": 35}},
        )

    decision = asyncio.run(_run())

    assert decision.is_sacrificial is True
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
    assert dispatcher._backend_launcher.numa_membind is True


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
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.shutil.which", lambda cmd: None)

    launcher = ManagedBackendLauncher(
        specs=[
            ManagedBackendSpec(device_id="0", port=8091, base_url="http://127.0.0.1:8091", hardware_profile="910B2"),
            ManagedBackendSpec(device_id="1", port=8092, base_url="http://127.0.0.1:8092", hardware_profile="910B3"),
        ],
        model="Qwen/Qwen-Image",
        backend_args=["--omni", "--vae-use-slicing", "--vae-use-tiling"],
        backend_env={"VLLM_PLUGINS": "ascend"},
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        numa_membind=False,
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


def test_parse_npu_smi_bus_ids_extracts_mapping():
    output = """
| 0     910B3               | OK            | 92.5        30                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3429 / 65536         |
| 1     910B3               | OK            | 87.5        31                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3414 / 65536         |
"""

    assert _parse_npu_smi_bus_ids(output) == {
        "0": "0000:c1:00.0",
        "1": "0000:c2:00.0",
    }


def test_managed_backend_launcher_prefixes_numactl_when_numa_is_resolved(tmp_path, monkeypatch):
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
        popen_calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ManagedBackendLauncher, "_wait_until_healthy", lambda self: None)
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.shutil.which", lambda cmd: "/usr/bin/numactl")
    monkeypatch.setattr(ManagedBackendLauncher, "_get_device_bus_id", lambda self, device_id: "0000:c1:00.0")
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher._read_pci_numa_node", lambda bus_id: 6)

    launcher = ManagedBackendLauncher(
        specs=[ManagedBackendSpec(device_id="0", port=8091, base_url="http://127.0.0.1:8091", hardware_profile="910B3")],
        model="Qwen/Qwen-Image",
        backend_args=["--omni"],
        backend_env={"VLLM_PLUGINS": "ascend"},
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        numa_membind=False,
        health_timeout_s=10.0,
        health_poll_interval_s=0.1,
        log_dir=str(tmp_path),
    )

    launcher.start_all()
    launcher.stop_all()

    assert popen_calls == [
        [
            "numactl",
            "--cpunodebind",
            "6",
            "vllm",
            "serve",
            "Qwen/Qwen-Image",
            "--port",
            "8091",
            "--super-p95-hardware-profile",
            "910B3",
            "--omni",
        ]
    ]


def test_managed_backend_launcher_logs_launch_details(tmp_path, monkeypatch):
    popen_calls = []
    log_messages = []

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
        popen_calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ManagedBackendLauncher, "_wait_until_healthy", lambda self: None)
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.shutil.which", lambda cmd: "/usr/bin/numactl")
    monkeypatch.setattr(ManagedBackendLauncher, "_get_device_bus_id", lambda self, device_id: "0000:c1:00.0")
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher._read_pci_numa_node", lambda bus_id: 6)
    monkeypatch.setattr(
        "benchmarks.diffusion.super_p95_dispatcher.logger.info",
        lambda msg, *args: log_messages.append(msg % args if args else msg),
    )

    launcher = ManagedBackendLauncher(
        specs=[ManagedBackendSpec(device_id="0", port=8091, base_url="http://127.0.0.1:8091", hardware_profile="910B3")],
        model="Qwen/Qwen-Image",
        backend_args=["--omni"],
        backend_env={"VLLM_PLUGINS": "ascend"},
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        numa_membind=False,
        health_timeout_s=10.0,
        health_poll_interval_s=0.1,
        log_dir=str(tmp_path),
    )

    launcher.start_all()
    launcher.stop_all()

    assert popen_calls
    assert any("Launching backend port=8091 device_id=0 hardware_profile=910B3 numa_node=6 membind=False" in msg for msg in log_messages)
    assert any("Starting 1 managed super_p95 backends." in msg for msg in log_messages)


def test_managed_backend_launcher_can_opt_in_numa_membind(tmp_path, monkeypatch):
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
        popen_calls.append(cmd)
        return FakeProcess()

    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.subprocess.Popen", fake_popen)
    monkeypatch.setattr(ManagedBackendLauncher, "_wait_until_healthy", lambda self: None)
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher.shutil.which", lambda cmd: "/usr/bin/numactl")
    monkeypatch.setattr(ManagedBackendLauncher, "_get_device_bus_id", lambda self, device_id: "0000:c1:00.0")
    monkeypatch.setattr("benchmarks.diffusion.super_p95_dispatcher._read_pci_numa_node", lambda bus_id: 6)

    launcher = ManagedBackendLauncher(
        specs=[ManagedBackendSpec(device_id="0", port=8091, base_url="http://127.0.0.1:8091", hardware_profile="910B3")],
        model="Qwen/Qwen-Image",
        backend_args=["--omni"],
        backend_env={},
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        numa_membind=True,
        health_timeout_s=10.0,
        health_poll_interval_s=0.1,
        log_dir=str(tmp_path),
    )

    launcher.start_all()
    launcher.stop_all()

    assert popen_calls == [
        [
            "numactl",
            "--cpunodebind",
            "6",
            "--membind",
            "6",
            "vllm",
            "serve",
            "Qwen/Qwen-Image",
            "--port",
            "8091",
            "--super-p95-hardware-profile",
            "910B3",
            "--omni",
        ]
    ]


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


def test_build_dispatcher_from_args_disables_membind_for_multi_device_usp():
    args = Namespace(
        backend_urls=None,
        num_servers=1,
        device_ids="0",
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        backend_host="127.0.0.1",
        backend_start_port=8091,
        backend_hardware_profiles="910B3",
        backend_args="--omni --usp 8 --enable-layerwise-offload --vae-use-slicing --vae-use-tiling",
        backend_env=[],
        device_env_var="ASCEND_RT_VISIBLE_DEVICES",
        numa_membind=False,
        no_numa_membind=False,
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

    assert dispatcher._backend_launcher is not None
    assert dispatcher._backend_launcher.numa_membind is False


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
    assert decision.estimated_service_s == pytest.approx(14.22)


def test_dispatcher_estimates_wan22_videos_request():
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://backend-0"],
        backend_hardware_profiles=["910B2"],
        quota_every=20,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=30.0,
    )

    estimate = dispatcher._estimate_service_s(
        "/v1/videos",
        {
            "width": "854",
            "height": "480",
            "num_inference_steps": "3",
            "num_frames": "80",
            "fps": "16",
        },
        "910B2",
    )

    assert estimate == pytest.approx(38.07)


def test_form_to_estimation_dict_ignores_uploads():
    form = type(
        "FakeForm",
        (),
        {
            "multi_items": lambda self: [
                ("prompt", "test"),
                ("width", "854"),
                ("height", "480"),
                ("input_reference", UploadFile(file=io.BytesIO(b"png"), filename="x.png")),
            ]
        },
    )()

    body = _form_to_estimation_dict(form)

    assert body == {"prompt": "test", "width": "854", "height": "480"}
