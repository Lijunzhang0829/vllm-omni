# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler


def test_solve_small_linear_system_matches_torch_linalg_for_1x1():
    R = torch.tensor([[2.5]], dtype=torch.float32)
    b = torch.tensor([5.0], dtype=torch.float32)

    actual = FlowUniPCMultistepScheduler._solve_small_linear_system(R, b, torch.float32)
    expected = torch.linalg.solve(R.to(torch.float64), b.to(torch.float64)).to(torch.float32)

    assert torch.allclose(actual.cpu(), expected.cpu(), atol=1e-6, rtol=1e-6)


def test_solve_small_linear_system_matches_torch_linalg_for_2x2():
    R = torch.tensor([[1.5, 0.25], [0.125, 2.0]], dtype=torch.float32)
    b = torch.tensor([0.75, -1.25], dtype=torch.float32)

    actual = FlowUniPCMultistepScheduler._solve_small_linear_system(R, b, torch.float32)
    expected = torch.linalg.solve(R.to(torch.float64), b.to(torch.float64)).to(torch.float32)

    assert torch.allclose(actual.cpu(), expected.cpu(), atol=1e-6, rtol=1e-6)


def test_solve_small_linear_system_cpu_fallback_matches_torch_linalg_for_3x3():
    R = torch.tensor(
        [
            [1.0, 0.2, -0.1],
            [0.0, 2.0, 0.3],
            [0.0, 0.1, 1.5],
        ],
        dtype=torch.float32,
    )
    b = torch.tensor([1.0, -0.5, 0.75], dtype=torch.float32)

    actual = FlowUniPCMultistepScheduler._solve_small_linear_system(R, b, torch.float32)
    expected = torch.linalg.solve(R.to(torch.float64), b.to(torch.float64)).to(torch.float32)

    assert torch.allclose(actual.cpu(), expected.cpu(), atol=1e-6, rtol=1e-6)
