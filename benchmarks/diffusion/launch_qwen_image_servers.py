#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Launch multiple single-device Qwen-Image servers and an optional dispatcher."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LOG_ROOT = Path("benchmarks/diffusion/logs")


@dataclass
class ManagedProcess:
    name: str
    command: list[str]
    process: subprocess.Popen[str]
    log_path: Path
    log_file: object


def _stream_output(name: str, stream: object, log_file: object) -> None:
    assert hasattr(stream, "readline")
    for line in iter(stream.readline, ""):
        formatted = f"[{name}] {line}"
        sys.stdout.write(formatted)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def _parse_devices(devices_arg: str | None, num_servers: int) -> list[str]:
    if devices_arg:
        devices = [item.strip() for item in devices_arg.split(",") if item.strip()]
    else:
        devices = [str(index) for index in range(num_servers)]

    if len(devices) < num_servers:
        raise ValueError(f"Requested {num_servers} servers but only {len(devices)} devices provided")
    return devices[:num_servers]


def _parse_cpu_range_list(value: str) -> list[int]:
    nodes: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            nodes.extend(range(start, end + 1))
        else:
            nodes.append(int(part))
    return nodes


def _get_online_numa_nodes() -> list[int]:
    online_path = Path("/sys/devices/system/node/online")
    try:
        return _parse_cpu_range_list(online_path.read_text(encoding="utf-8").strip())
    except Exception:
        return [0]


def _pick_numa_node(device: str, index: int, numa_nodes: list[int]) -> int | None:
    if len(numa_nodes) <= 1:
        return None

    try:
        device_id = int(device)
    except ValueError:
        device_id = index

    if device_id in numa_nodes:
        return device_id
    return numa_nodes[index % len(numa_nodes)]


def _require_numa_binding_mode(numa_nodes: list[int], numactl_path: str | None) -> str:
    if len(numa_nodes) <= 1:
        return "disabled"
    if numactl_path is None:
        raise RuntimeError(
            f"Detected NUMA nodes {numa_nodes} but `numactl` is not installed. "
            "Install `numactl` and restart the launcher."
        )

    probe_node = str(numa_nodes[0])

    membind_probe = subprocess.run(
        [numactl_path, f"--cpunodebind={probe_node}", f"--membind={probe_node}", "/bin/true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if membind_probe.returncode == 0:
        return "cpu+membind"

    raise RuntimeError(
        f"Detected NUMA nodes {numa_nodes} but `numactl --cpunodebind={probe_node} --membind={probe_node}` "
        "is not permitted in the current environment. Fix the container/host permissions "
        "(for example `--privileged` or the required capabilities/cpuset settings) before launching."
    )


def _wrap_with_numa_binding(command: list[str], numa_node: int | None, binding_mode: str) -> list[str]:
    if numa_node is None:
        return command
    if binding_mode == "cpu+membind":
        return [
            "numactl",
            f"--cpunodebind={numa_node}",
            f"--membind={numa_node}",
            *command,
        ]
    return command


def _wait_for_health(base_url: str) -> bool:
    health_url = f"{base_url.rstrip('/')}/health"
    while True:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(1)
            continue
        time.sleep(1)


def _start_process(
    name: str,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    threading.Thread(
        target=_stream_output,
        args=(name, process.stdout, log_file),
        daemon=True,
    ).start()
    return ManagedProcess(name=name, command=command, process=process, log_path=log_path, log_file=log_file)


def _terminate_process(proc: ManagedProcess, grace_period_s: int = 10) -> None:
    if proc.process.poll() is not None:
        proc.log_file.close()
        return
    try:
        os.killpg(proc.process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + grace_period_s
    while time.time() < deadline:
        if proc.process.poll() is not None:
            proc.log_file.close()
            return
        time.sleep(0.2)

    try:
        os.killpg(proc.process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        proc.log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 8 single-device Qwen-Image servers plus a dispatcher.")
    parser.add_argument("--model", default="Qwen/Qwen-Image", help="Model name passed to vllm serve.")
    parser.add_argument("--host", default="127.0.0.1", help="Backend listen host.")
    parser.add_argument("--num-servers", type=int, default=8, help="Number of backend servers to start.")
    parser.add_argument("--devices", default=None, help="Comma-separated device ids. Default: 0..num-servers-1")
    parser.add_argument("--base-port", type=int, default=8091, help="First backend port.")
    parser.add_argument("--dispatcher-port", type=int, default=8090, help="Dispatcher port.")
    parser.add_argument("--dispatcher-host", default="0.0.0.0", help="Dispatcher listen host.")
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Directory for server and dispatcher logs. Default: benchmarks/diffusion/logs/<timestamp>",
    )
    parser.add_argument(
        "--vllm-command",
        default="vllm",
        help="CLI used to start serving. Default assumes `vllm` is on PATH.",
    )
    parser.add_argument(
        "--extra-server-args",
        default="",
        help="Extra arguments appended to each `vllm serve` command.",
    )
    parser.add_argument(
        "--disable-diffusion-preemption",
        action="store_true",
        help="Disable step-level diffusion preemption on every backend server.",
    )
    parser.add_argument(
        "--diffusion-request-aging-alpha",
        type=float,
        default=0.0,
        help="Anti-starvation aging strength forwarded to every backend server. "
        "0 keeps pure shortest-remaining-time-first behavior.",
    )
    parser.add_argument(
        "--diffusion-request-aging-cap",
        type=float,
        default=8.0,
        help="Maximum normalized waited-work units used by diffusion request aging.",
    )
    parser.add_argument(
        "--diffusion-request-aging-cost-ref",
        type=float,
        default=float(1024 * 1024 * 25),
        help="Reference work unit for diffusion request aging normalization. "
        "Default is the cost of 1024x1024 with 25 inference steps.",
    )
    parser.add_argument(
        "--no-dispatcher",
        action="store_true",
        help="Only start backend servers, do not start the dispatcher.",
    )
    parser.add_argument(
        "--print-commands-only",
        action="store_true",
        help="Print commands and exit without launching processes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    devices = _parse_devices(args.devices, args.num_servers)
    numa_nodes = _get_online_numa_nodes()
    numactl_path = shutil.which("numactl")
    numa_binding_mode = _require_numa_binding_mode(numa_nodes, numactl_path)
    enable_numa_binding = numa_binding_mode != "disabled"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else DEFAULT_LOG_ROOT / f"multi-server-{timestamp}"
    extra_server_args = shlex.split(args.extra_server_args)

    backend_urls = []
    backend_commands: list[tuple[str, list[str], str, str, Path, int | None]] = []
    for index, device in enumerate(devices):
        port = args.base_port + index
        base_url = f"http://{args.host}:{port}"
        backend_urls.append(base_url)
        base_command = [
            args.vllm_command,
            "serve",
            args.model,
            "--omni",
            "--port",
            str(port),
            "--host",
            args.host,
            *extra_server_args,
        ]
        if args.disable_diffusion_preemption:
            base_command.append("--disable-diffusion-preemption")
        if args.diffusion_request_aging_alpha != 0.0:
            base_command.extend(
                [
                    "--diffusion-request-aging-alpha",
                    str(args.diffusion_request_aging_alpha),
                    "--diffusion-request-aging-cap",
                    str(args.diffusion_request_aging_cap),
                    "--diffusion-request-aging-cost-ref",
                    str(args.diffusion_request_aging_cost_ref),
                ]
            )
        numa_node = _pick_numa_node(device, index, numa_nodes) if enable_numa_binding else None
        command = _wrap_with_numa_binding(base_command, numa_node, numa_binding_mode)
        log_path = log_dir / f"server-{index}.log"
        backend_commands.append((f"server-{index}", command, device, base_url, log_path, numa_node))

    dispatcher_command = [
        sys.executable,
        str(Path(__file__).with_name("dispatch_qwen_image.py")),
        "--host",
        args.dispatcher_host,
        "--port",
        str(args.dispatcher_port),
        "--backend-urls",
        *backend_urls,
    ]

    if args.print_commands_only:
        for name, command, device, _base_url, log_path, numa_node in backend_commands:
            numa_suffix = "" if numa_node is None else f" numa={numa_node}"
            print(f"[{name}] device={device}{numa_suffix} log={log_path}")
            print(" ".join(shlex.quote(part) for part in command))
        if not args.no_dispatcher:
            print(f"[dispatcher] log={log_dir / 'dispatcher.log'}")
            print(" ".join(shlex.quote(part) for part in dispatcher_command))
        return 0

    processes: list[ManagedProcess] = []
    stop_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}, shutting down...")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        for name, command, device, base_url, log_path, numa_node in backend_commands:
            env = os.environ.copy()
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            env["ASCEND_RT_VISIBLE_DEVICES"] = device
            env["PYTHONUNBUFFERED"] = "1"
            process = _start_process(name=name, command=command, env=env, log_path=log_path)
            processes.append(process)
            numa_suffix = "" if numa_node is None else f" numa={numa_node}"
            print(f"Started {name} on device {device}{numa_suffix} at {base_url} -> log {log_path}")

        for index, base_url in enumerate(backend_urls):
            if stop_requested:
                break
            print(f"Waiting for server-{index} health: {base_url}/health")
            healthy = _wait_for_health(base_url)
            if not healthy:
                raise RuntimeError(f"server-{index} failed health check")
            print(f"server-{index} is healthy")

        if stop_requested:
            return 1

        if not args.no_dispatcher:
            dispatcher_log = log_dir / "dispatcher.log"
            dispatcher_env = os.environ.copy()
            dispatcher_env["PYTHONUNBUFFERED"] = "1"
            dispatcher = _start_process(
                name="dispatcher",
                command=dispatcher_command,
                env=dispatcher_env,
                log_path=dispatcher_log,
            )
            processes.append(dispatcher)
            print(f"Dispatcher listening at http://{args.dispatcher_host}:{args.dispatcher_port}")
            print(f"Dispatcher log: {dispatcher_log}")

        print("Backend URLs:")
        for backend_url in backend_urls:
            print(f"  - {backend_url}")
        if not args.no_dispatcher:
            print(f"Benchmark base URL: http://127.0.0.1:{args.dispatcher_port}")

        while not stop_requested:
            for proc in processes:
                return_code = proc.process.poll()
                if return_code is not None:
                    raise RuntimeError(f"{proc.name} exited early with code {return_code}. Check log: {proc.log_path}")
            time.sleep(1)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        for proc in reversed(processes):
            _terminate_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
