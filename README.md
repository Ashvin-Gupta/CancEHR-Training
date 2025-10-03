<h1 align="center">
  <!-- <img src="src/evaluation/visualisation_server/static/images/logo.png" width="64" style="vertical-align:middle;margin-right:8px;"> -->
  CancEHR
</h1>

This repo is for training sequence models on tokenized medical event datasets produced the `CancEHR-tokenization` repo.

### Contents
- **Data**: PyTorch datasets/dataloaders for tokenized EHR sequences (`src/data/`)
- **Training**: Config-driven experiments with logging and checkpointing (`src/experiments/`, `src/training/`)
- **Models**: Sequence models (LSTM, Transformer decoder, ...) trained to auto-regressively predict tokens (`src/models/`)
- **Evaluation**: Analyse and compare trained models (`src/evaluation/`)

### Quickstart
1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure data and model
- Copy/modify a config in `src/experiments/configs/` (e.g., `transformer_base.yaml`).
- Update paths under `data:` to point to your tokenized dataset (`train_dataset_dir`, `val_dataset_dir`, `vocab_path`).
- Set `training.device` to `cuda` or `cpu`.

3) Train
```bash
python -m src.experiments.run --config_name transformer_base --experiment_name exp_001
```
Outputs are saved to `src/experiments/results/exp_001/`:
- `config.yaml`, `training.log`, `loss.log`
- `model.pth`, `vocab.csv`

4) Visualize (losses, simulations, inference playground)
```bash
python -m src.evaluation.visualisation_server.main
```

### Notes
- The code expects tokenized EHR data produced by the `ehr-tokenization` pipeline (see `src/data/`). Update config paths accordingly.
- Available example configs: `transformer_base.yaml`, `transformer_med.yaml`, `lstm_base.yaml`, `template.yaml` in `src/experiments/configs/`.
