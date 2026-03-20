# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_wan22_sp_plan_shards_patch_rope_prepare_outputs_together():
    plan = WanTransformer3DModel._sp_plan

    assert "patch_rope_prepare" in plan
    assert "proj_out" in plan
    patch_rope_plan = plan["patch_rope_prepare"]
    assert set(patch_rope_plan.keys()) == {0, 1, 2}
