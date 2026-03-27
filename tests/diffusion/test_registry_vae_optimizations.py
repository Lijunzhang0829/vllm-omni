from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import initialize_model


class _DummyVAE:
    def __init__(self):
        self.use_slicing = False
        self.use_tiling = False
        self.enable_slicing_calls = 0
        self.enable_tiling_calls = 0

    def enable_slicing(self):
        self.enable_slicing_calls += 1
        self.use_slicing = True

    def enable_tiling(self):
        self.enable_tiling_calls += 1
        self.use_tiling = True


class _DummyPipeline:
    def __init__(self, od_config):
        self.od_config = od_config
        self.vae = _DummyVAE()


def test_initialize_model_auto_enables_wan_vae_tiling(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry.DiffusionModelRegistry._try_load_model_cls",
        lambda _name: _DummyPipeline,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry._apply_sequence_parallel_if_enabled",
        lambda model, od_config: None,
    )

    od_config = OmniDiffusionConfig(model_class_name="Wan22Pipeline")
    model = initialize_model(od_config)

    assert od_config.vae_use_tiling is True
    assert model.vae.use_tiling is True
    assert model.vae.enable_tiling_calls == 1


def test_initialize_model_keeps_non_wan_vae_tiling_disabled(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry.DiffusionModelRegistry._try_load_model_cls",
        lambda _name: _DummyPipeline,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry._apply_sequence_parallel_if_enabled",
        lambda model, od_config: None,
    )

    od_config = OmniDiffusionConfig(model_class_name="QwenImagePipeline")
    model = initialize_model(od_config)

    assert od_config.vae_use_tiling is False
    assert model.vae.use_tiling is False
    assert model.vae.enable_tiling_calls == 0


def test_initialize_model_uses_enable_slicing_when_requested(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry.DiffusionModelRegistry._try_load_model_cls",
        lambda _name: _DummyPipeline,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.registry._apply_sequence_parallel_if_enabled",
        lambda model, od_config: None,
    )

    od_config = OmniDiffusionConfig(
        model_class_name="QwenImagePipeline",
        vae_use_slicing=True,
    )
    model = initialize_model(od_config)

    assert model.vae.use_slicing is True
    assert model.vae.enable_slicing_calls == 1
