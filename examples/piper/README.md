# Piper pi0.5 fine-tuning

The `pi05_piper_full_finetune` and `pi05_piper_lora_finetune` configs define Piper transforms and training
hyperparameters without embedding a dataset location or ID. Both configs use a global batch size of 32 and an action
horizon of 30 frames (one second at 30 FPS). They convert all seven absolute action dimensions to values relative to
the current state. The seventh gripper dimension remains continuous; it is not binarized.

Use the reusable wrapper to select the local dataset, pi0.5 base checkpoint, and exact checkpoint output path:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
  --config pi05_piper_full_finetune \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi05_base \
  --checkpoint-dir ./checkpoints/piper/full \
  --exp-name piper_full \
  --compute-norm-stats \
  --overwrite
```

For LoRA, change the config and output directory:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_piper.py \
  --config pi05_piper_lora_finetune \
  --dataset-dir /path/to/lerobot/dataset \
  --dataset-repo-id piper_dataset \
  --base-model-dir ./checkpoints/pi05_base \
  --checkpoint-dir ./checkpoints/piper/lora \
  --exp-name piper_lora \
  --compute-norm-stats \
  --overwrite
```

`--dataset-dir` and `--dataset-repo-id` are required. The ID is a stable logical name used to store normalization
assets inside checkpoints; it does not need to be a Hugging Face Hub repository.

`--base-model-dir` accepts either a checkpoint root containing `params` or the `params` directory itself. It also
accepts a `gs://` checkpoint URL; when omitted it uses `gs://openpi-assets/checkpoints/pi05_base`.

By default, OpenPI normalization statistics are read from `<dataset-dir>/norm_stats.json`. With
`--compute-norm-stats`, the wrapper computes that file only when it is missing and reuses it on subsequent runs.
Use `--norm-stats-dir` to keep it elsewhere. Without `--compute-norm-stats`, a missing file is reported before
model initialization.

When `--checkpoint-dir` is omitted, the standard OpenPI layout is used:
`<checkpoint-base-dir>/<config>/<exp-name>`. Use `--resume` instead of `--overwrite` to continue an existing run.
