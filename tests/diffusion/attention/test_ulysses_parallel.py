# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.attention.parallel.ulysses as ulysses_module
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_ulysses_packs_qkv_for_self_attention(monkeypatch):
    calls = {"4d": 0, "5d": 0}

    class _A2A5D:
        @staticmethod
        def apply(group, x, scatter_idx, gather_idx, use_sync):
            del group, scatter_idx, gather_idx, use_sync
            calls["5d"] += 1
            return x

    class _A2A4D:
        @staticmethod
        def apply(group, x, scatter_idx, gather_idx, use_sync):
            del group, scatter_idx, gather_idx, use_sync
            calls["4d"] += 1
            return x

    monkeypatch.setattr(ulysses_module, "SeqAllToAll5D", _A2A5D)
    monkeypatch.setattr(ulysses_module, "SeqAllToAll4D", _A2A4D)

    strategy = UlyssesParallelAttention(
        sp_group=SimpleNamespace(ulysses_group="pg", ulysses_world_size=2, ulysses_rank=0, ring_world_size=1),
        scatter_idx=2,
        gather_idx=1,
        use_sync=False,
    )
    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 4, 16)
    v = torch.randn(1, 8, 4, 16)

    q, k, v, _, _ = strategy.pre_attention(q, k, v, None)

    assert q.shape == (1, 8, 4, 16)
    assert k.shape == (1, 8, 4, 16)
    assert v.shape == (1, 8, 4, 16)
    assert calls["5d"] == 1
    assert calls["4d"] == 0
