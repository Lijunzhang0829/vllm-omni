# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_wan22_scheduler_state_roundtrip_preserves_multistep_history():
    pipeline = object.__new__(Wan22Pipeline)

    model_output_0 = torch.randn(1, 2, 3)
    model_output_1 = torch.randn(1, 2, 3)
    timestep_0 = torch.tensor(900)
    timestep_1 = torch.tensor(800)
    last_sample = torch.randn(1, 4, 5, 6, 7)

    scheduler = SimpleNamespace(
        config=SimpleNamespace(solver_order=2),
        model_outputs=[model_output_0, model_output_1],
        timestep_list=[timestep_0, timestep_1],
        lower_order_nums=2,
        last_sample=last_sample,
        _step_index=7,
        _begin_index=None,
        this_order=2,
    )
    pipeline.scheduler = scheduler

    state = pipeline._snapshot_scheduler_state()

    scheduler.model_outputs = [None, None]
    scheduler.timestep_list = [None, None]
    scheduler.lower_order_nums = 0
    scheduler.last_sample = None
    scheduler._step_index = None
    scheduler.this_order = 1

    pipeline._restore_scheduler_state(state)

    assert pipeline.scheduler.model_outputs == [model_output_0, model_output_1]
    assert pipeline.scheduler.timestep_list == [timestep_0, timestep_1]
    assert pipeline.scheduler.lower_order_nums == 2
    assert pipeline.scheduler.last_sample is last_sample
    assert pipeline.scheduler._step_index == 7
    assert pipeline.scheduler.this_order == 2


def test_wan22_scheduler_state_restore_handles_missing_history():
    pipeline = object.__new__(Wan22Pipeline)
    pipeline.scheduler = SimpleNamespace(
        config=SimpleNamespace(solver_order=3),
        model_outputs=[],
        timestep_list=[],
        lower_order_nums=0,
        last_sample=None,
        _step_index=None,
        _begin_index=None,
        this_order=1,
    )

    pipeline._restore_scheduler_state({"step_index": 4, "lower_order_nums": 1})

    assert pipeline.scheduler.model_outputs == [None, None, None]
    assert pipeline.scheduler.timestep_list == [None, None, None]
    assert pipeline.scheduler.lower_order_nums == 1
    assert pipeline.scheduler._step_index == 4
    assert pipeline.scheduler.this_order == 1
