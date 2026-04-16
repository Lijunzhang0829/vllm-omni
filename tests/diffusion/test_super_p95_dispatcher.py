# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.diffusion.super_p95_dispatcher import SuperP95Dispatcher
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import estimate_service_time_s
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_estimate_uses_seconds_and_fps_when_num_frames_missing() -> None:
    body = {
        "size": "854x480",
        "seconds": "4",
        "fps": "24",
        "num_inference_steps": "3",
    }

    estimated = SuperP95Dispatcher._estimate_service_s("/v1/videos", body, "910B3")
    expected = estimate_service_time_s(
        OmniDiffusionRequest(
            prompts=["video"],
            sampling_params=OmniDiffusionSamplingParams(
                width=854,
                height=480,
                num_inference_steps=3,
                num_frames=96,
            ),
            request_ids=["video"],
        ),
        hardware_profile="910B3",
    )

    assert estimated == pytest.approx(expected)


def test_video_estimate_defaults_to_24_fps_for_seconds_only_requests() -> None:
    body = {
        "size": "854x480",
        "seconds": "4",
        "num_inference_steps": "3",
    }

    estimated = SuperP95Dispatcher._estimate_service_s("/v1/videos", body, "910B3")
    expected = estimate_service_time_s(
        OmniDiffusionRequest(
            prompts=["video"],
            sampling_params=OmniDiffusionSamplingParams(
                width=854,
                height=480,
                num_inference_steps=3,
                num_frames=96,
            ),
            request_ids=["video"],
        ),
        hardware_profile="910B3",
    )

    assert estimated == pytest.approx(expected)
