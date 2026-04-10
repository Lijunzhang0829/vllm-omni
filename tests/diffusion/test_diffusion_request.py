# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.super_p95 import (
    EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S,
    EXTRA_ARG_SUPER_P95_SACRIFICIAL,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee on a table"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )


def test_tp_seed_same_across_ranks_and_varies_across_requests():
    random.seed(0)
    n_requests = 5
    seeds = [_make_request().sampling_params.seed for _ in range(n_requests)]

    # Seed must be auto-assigned (not None) so every TP rank can use it.
    assert all(s is not None for s in seeds)

    # Seeds must vary across requests (non-determinism preserved).
    assert len(set(seeds)) == n_requests, f"Expected {n_requests} unique seeds but got {len(set(seeds))}: {seeds}"


def test_super_p95_metadata_is_normalized_from_extra_args():
    request = OmniDiffusionRequest(
        prompts=[{"prompt": "test prompt"}],
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            extra_args={
                EXTRA_ARG_SUPER_P95_SACRIFICIAL: True,
                EXTRA_ARG_SUPER_P95_ESTIMATED_SERVICE_S: 12.5,
            },
        ),
        request_ids=["meta-1"],
    )

    assert request.super_p95_sacrificial is True
    assert request.super_p95_estimated_service_s == 12.5
