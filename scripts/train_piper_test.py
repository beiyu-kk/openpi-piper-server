import argparse
import sys

import pytest

from scripts import compute_norm_stats
from scripts import train_piper


def test_resolve_base_params_path(tmp_path):
    params_dir = tmp_path / "pi05_base" / "params"
    params_dir.mkdir(parents=True)

    assert train_piper._resolve_base_params_path(str(params_dir.parent)) == str(params_dir)  # noqa: SLF001
    assert (
        train_piper._resolve_base_params_path("gs://bucket/pi05_base")  # noqa: SLF001
        == "gs://bucket/pi05_base/params"
    )


def test_build_config_with_custom_paths(tmp_path):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "meta").mkdir(parents=True)
    (dataset_dir / "meta" / "info.json").write_text("{}")
    (dataset_dir / "data").mkdir()
    params_dir = tmp_path / "pi05_base" / "params"
    params_dir.mkdir(parents=True)
    norm_stats_dir = tmp_path / "normalization"
    checkpoint_dir = tmp_path / "output"
    args = argparse.Namespace(
        config="pi05_piper_lora_finetune",
        dataset_dir=str(dataset_dir),
        dataset_repo_id="local/piper",
        base_model_dir=str(params_dir.parent),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        checkpoint_dir=str(checkpoint_dir),
        norm_stats_dir=str(norm_stats_dir),
        exp_name="test",
        overwrite=False,
        resume=False,
        batch_size=None,
        num_workers=None,
        num_train_steps=None,
        fsdp_devices=None,
        wandb_enabled=False,
    )

    config = train_piper._build_config(args)  # noqa: SLF001
    data_config = config.data.create_base_config(config.assets_dirs, config.model)

    assert config.batch_size == 32
    assert config.checkpoint_dir == checkpoint_dir
    assert data_config.dataset_root == str(dataset_dir)
    assert data_config.asset_id == "local/piper"
    assert compute_norm_stats.get_norm_stats_dir(config, data_config) == norm_stats_dir


@pytest.mark.parametrize(
    ("provided_argument", "value", "missing_argument"),
    [
        ("--dataset-repo-id", "local/piper", "--dataset-dir"),
        ("--dataset-dir", "/tmp/piper", "--dataset-repo-id"),
    ],
)
def test_dataset_arguments_are_required(monkeypatch, capsys, provided_argument, value, missing_argument):
    monkeypatch.setattr(sys, "argv", ["train_piper.py", provided_argument, value, "--exp-name", "test"])

    with pytest.raises(SystemExit, match="2"):
        train_piper._parse_args()  # noqa: SLF001

    assert missing_argument in capsys.readouterr().err
