from types import SimpleNamespace
import threading

import torch

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline


def _make_pipeline():
    pipe = object.__new__(QwenImagePipeline)
    pipe.scheduler = SimpleNamespace(set_begin_index=lambda _: None)
    pipe.transformer = SimpleNamespace(do_true_cfg=False)
    pipe._current_timestep = None
    pipe._interrupt = False
    pipe._interrupt_event = None
    pipe._num_timesteps = 0
    pipe.predict_noise_maybe_with_cfg = lambda *args, **kwargs: args[2]["hidden_states"]
    pipe.scheduler_step_maybe_with_cfg = lambda noise_pred, _t, latents, _do_true_cfg: latents + 1
    pipe.prepare_timesteps = lambda num_inference_steps, sigmas, image_seq_len: (
        torch.tensor([1, 2, 3], dtype=torch.float32),
        num_inference_steps,
    )
    return pipe


def _make_state():
    return {
        "timesteps": torch.tensor([1, 2, 3], dtype=torch.float32),
        "next_step_index": 0,
        "do_true_cfg": False,
        "guidance": None,
        "true_cfg_scale": 1.0,
        "prompt_embeds_mask": None,
        "prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "negative_prompt_embeds": None,
        "img_shapes": [],
        "txt_seq_lens": None,
        "negative_txt_seq_lens": None,
        "attention_kwargs": {},
        "latents": torch.zeros(1),
        "num_inference_steps": 3,
        "sigmas": None,
        "image_seq_len": 16,
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


def test_resume_rebuilds_scheduler_state_before_setting_begin_index():
    begin_indices = []
    pipe = _make_pipeline()
    pipe.scheduler = SimpleNamespace(set_begin_index=lambda idx: begin_indices.append(idx))
    restored = torch.tensor([10, 20, 30], dtype=torch.float32)
    pipe.prepare_timesteps = lambda num_inference_steps, sigmas, image_seq_len: (restored, num_inference_steps)

    state = _make_state()
    state["timesteps"] = torch.tensor([999], dtype=torch.float32)
    state["next_step_index"] = 1

    next_state, finished, completed_steps = pipe._run_generation_steps(state)

    assert torch.equal(next_state["timesteps"], restored)
    assert pipe._num_timesteps == 3
    assert begin_indices == [1]
    assert finished is True
    assert completed_steps == 2
