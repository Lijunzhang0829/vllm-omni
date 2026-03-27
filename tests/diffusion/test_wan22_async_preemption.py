from types import SimpleNamespace
import threading

import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline


def _make_pipeline():
    pipe = object.__new__(Wan22Pipeline)
    pipe.transformer = SimpleNamespace(dtype=torch.float32)
    pipe.transformer_2 = None
    pipe._current_timestep = None
    pipe._interrupt = False
    pipe._interrupt_event = None
    pipe.predict_noise_maybe_with_cfg = lambda **kwargs: kwargs["positive_kwargs"]["hidden_states"]
    pipe.scheduler_step_maybe_with_cfg = lambda noise_pred, _t, latents, _do_true_cfg: latents + 1
    return pipe


def _make_state():
    return {
        "timesteps": torch.tensor([1, 2, 3], dtype=torch.float32),
        "next_step_index": 0,
        "boundary_timestep": None,
        "guidance_low": 1.0,
        "guidance_high": 1.0,
        "prompt_embeds": torch.zeros(1, 1, 1),
        "negative_prompt_embeds": None,
        "attention_kwargs": {},
        "latents": torch.zeros(1, 1, 1, 2, 2),
        "num_latent_frames": 1,
        "latent_height": 2,
        "latent_width": 2,
    }


def test_async_preemption_yields_after_one_step_when_flag_is_pre_set():
    pipe = _make_pipeline()
    pipe._interrupt = True

    state, finished, completed_steps = pipe._run_generation_steps(_make_state())

    assert finished is False
    assert completed_steps == 1
    assert state["next_step_index"] == 1


def test_async_preemption_stops_at_next_boundary_after_signal_arrives():
    pipe = _make_pipeline()
    call_count = {"steps": 0}

    def _scheduler_step(noise_pred, _t, latents, _do_true_cfg):
        call_count["steps"] += 1
        if call_count["steps"] == 1:
            pipe._interrupt = True
        return latents + 1

    pipe.scheduler_step_maybe_with_cfg = _scheduler_step

    state, finished, completed_steps = pipe._run_generation_steps(_make_state())

    assert finished is False
    assert completed_steps == 1
    assert state["next_step_index"] == 1


def test_interrupt_property_checks_shared_event():
    pipe = _make_pipeline()
    event = threading.Event()
    pipe._interrupt_event = event

    assert pipe.interrupt is False
    event.set()
    assert pipe.interrupt is True
