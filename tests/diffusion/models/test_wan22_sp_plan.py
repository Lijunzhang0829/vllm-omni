# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_wan22_sp_plan_includes_rope_by_default():
    plan = WanTransformer3DModel._build_sp_plan(enable_rope_split=True)

    assert "rope" in plan
    assert "blocks.0" in plan
    assert "proj_out" in plan


def test_wan22_sp_plan_can_disable_rope_split():
    plan = WanTransformer3DModel._build_sp_plan(enable_rope_split=False)

    assert "rope" not in plan
    assert "blocks.0" in plan
    assert "proj_out" in plan
