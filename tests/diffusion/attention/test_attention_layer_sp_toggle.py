# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.layer as attention_layer_module
from vllm_omni.diffusion.attention.layer import Attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_attention_can_disable_sequence_parallel(monkeypatch):
    monkeypatch.setattr(attention_layer_module, "get_attn_backend", lambda _: SimpleNamespace(get_impl_cls=lambda: _Impl))
    monkeypatch.setattr(
        attention_layer_module,
        "build_parallel_attention_strategy",
        lambda **kwargs: SimpleNamespace(pre_attention=lambda q, k, v, m: (q, k, v, m, None), post_attention=lambda o, c: o),
    )
    monkeypatch.setattr(
        attention_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            omni_diffusion_config=SimpleNamespace(parallel_config=SimpleNamespace(ring_degree=1)),
            sp_active=True,
        ),
    )
    monkeypatch.setattr(attention_layer_module, "is_forward_context_available", lambda: True)

    attn = Attention(
        num_heads=1,
        head_size=1,
        causal=False,
        softmax_scale=1.0,
        enable_sequence_parallel=False,
    )

    assert attn._get_active_parallel_strategy() is attn._no_parallel_strategy


class _Impl:
    def __init__(self, **kwargs):
        del kwargs

    def forward(self, query, key, value, attn_metadata):
        del key, value, attn_metadata
        return query
