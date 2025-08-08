# Experiments

Config-driven training runs for Nightingale models.

## Layout
- `configs/` — example experiment configs (edit or add your own)
- `run.py` — entrypoint to launch training
- `results/` — output directory per experiment name

## Quickstart
```bash
python -m src.experiments.run \
  --config_name transformer_base \
  --experiment_name exp_001
```

- Configs are loaded from `src/experiments/configs/<config_name>.yaml`.
- Outputs are written to `src/experiments/results/<experiment_name>/`:
  - `config.yaml` (resolved copy), `training.log`, `loss.log`
  - `model.pth`, `vocab.csv`
  - optional `simulations/` (created by simulation tools)

## Minimal config notes
Required top-level sections:
- `model` — e.g., type (`lstm` | `transformer`), sizes
- `optimiser` — e.g., `{ type: adam, lr: 0.0001 }`
- `loss_function` — e.g., `{ type: cross_entropy }`
- `training` — `{ epochs: <int>, device: "cuda"|"cpu" }`
- `data` — paths and loader params: `train_dataset_dir`, `val_dataset_dir`, `vocab_path`, `sequence_length`, `batch_size`, `shuffle`

See examples in `configs/`:
- `transformer_base.yaml`, `transformer_med.yaml`, `lstm_base.yaml`, `template.yaml`

## Visualize
After training, start the visualization server to browse losses and run inference:
```bash
python -m src.evaluation.visualisation_server.main
```
This reads from `src/experiments/results/` automatically. 