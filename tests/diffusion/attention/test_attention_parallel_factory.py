# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.parallel.factory as factory_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_factory_enables_sync_for_ulysses_on_npu(monkeypatch):
    monkeypatch.setattr(
        factory_module,
        "get_forward_context",
        lambda: SimpleNamespace(omni_diffusion_config=SimpleNamespace(parallel_config=SimpleNamespace(ulysses_degree=2, ring_degree=1))),
    )
    monkeypatch.setattr(factory_module, "get_sp_group", lambda: "sp-group")
    monkeypatch.setattr(factory_module, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(factory_module, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))

    recorded = {}

    class _Ulysses:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(factory_module, "UlyssesParallelAttention", _Ulysses)

    factory_module.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert recorded["use_sync"] is True
