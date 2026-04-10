# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.diffusion.super_p95_dispatcher import SuperP95Dispatcher
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.super_p95_scheduler import SuperP95RequestScheduler
from vllm_omni.diffusion.super_p95 import apply_super_p95_request_headers
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_body(width: int, height: int, steps: int) -> dict:
    return {
        "model": "Qwen/Qwen-Image",
        "messages": [{"role": "user", "content": "prompt"}],
        "extra_body": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
        },
    }


def test_dispatcher_marks_large_request_as_sacrificial_when_quota_available() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://127.0.0.1:8091", "http://127.0.0.1:8092"],
        backend_hardware_profiles="910B3,910B3",
        quota_every=1,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=10.0,
    )

    decision = dispatcher._choose_backend("/v1/chat/completions", _make_body(1536, 1536, 35))

    assert decision.is_sacrificial is True
    assert decision.estimated_service_s > 0.0


def test_dispatcher_headers_flow_into_scheduler_ordering() -> None:
    dispatcher = SuperP95Dispatcher(
        backend_urls=["http://127.0.0.1:8091", "http://127.0.0.1:8092"],
        backend_hardware_profiles="910B3,910B3",
        quota_every=1,
        quota_amount=1,
        threshold_ratio=0.8,
        sacrificial_load_factor=0.1,
        request_timeout_s=10.0,
    )
    scheduler = SuperP95RequestScheduler()
    scheduler.initialize(object())

    normal_decision = dispatcher._choose_backend("/v1/chat/completions", _make_body(512, 512, 20))
    normal_headers = dispatcher._build_forward_headers({}, normal_decision)
    normal_params = OmniDiffusionSamplingParams(num_inference_steps=20)
    apply_super_p95_request_headers(normal_params, normal_headers)
    normal_request = OmniDiffusionRequest(
        prompts=["normal"],
        sampling_params=normal_params,
        request_ids=["normal"],
    )

    sacrificial_decision = dispatcher._choose_backend("/v1/chat/completions", _make_body(1536, 1536, 35))
    sacrificial_headers = dispatcher._build_forward_headers({}, sacrificial_decision)
    sacrificial_params = OmniDiffusionSamplingParams(num_inference_steps=35)
    apply_super_p95_request_headers(sacrificial_params, sacrificial_headers)
    sacrificial_request = OmniDiffusionRequest(
        prompts=["sacrificial"],
        sampling_params=sacrificial_params,
        request_ids=["sacrificial"],
    )

    scheduler.add_request(sacrificial_request)
    scheduler.add_request(normal_request)

    sched_output = scheduler.schedule()
    scheduled_ids = [req.sched_req_id for req in sched_output.scheduled_new_reqs]

    assert scheduled_ids == ["normal"]
