import dataclasses

import tyro

from openpi.models import pi0_config
from openpi.training import config as _config


def test_piper_finetune_configs():
    full = _config.get_config("pi05_piper_full_finetune")
    lora = _config.get_config("pi05_piper_lora_finetune")

    assert isinstance(full.model, pi0_config.Pi0Config)
    assert full.model.pi05
    assert full.model.action_horizon == 30
    assert full.batch_size == 32
    assert isinstance(full.data, _config.LeRobotPiperDataConfig)
    assert full.data.use_delta_actions
    assert full.data.repo_id is tyro.MISSING
    assert full.data.dataset_root is None
    assert full.data.norm_stats_dir is None
    assert full.data.assets.asset_id is None

    assert isinstance(lora.model, pi0_config.Pi0Config)
    assert lora.model.pi05
    assert lora.model.action_horizon == 30
    assert "lora" in lora.model.paligemma_variant
    assert "lora" in lora.model.action_expert_variant
    assert lora.batch_size == 32
    assert lora.ema_decay is None
    assert lora.data.repo_id is tyro.MISSING
    assert lora.data.dataset_root is None
    assert lora.data.norm_stats_dir is None
    assert lora.data.assets.asset_id is None


def test_checkpoint_dir_override(tmp_path):
    config = _config.get_config("pi05_piper_full_finetune")
    config = dataclasses.replace(config, exp_name="test", checkpoint_dir_override=str(tmp_path))
    assert config.checkpoint_dir == tmp_path
