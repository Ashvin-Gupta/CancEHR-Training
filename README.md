# Nightingale

Deep learning for electronic health records (EHR) - organized by data format for cancer prediction tasks.

## Overview

This repository provides three distinct pipelines for training models on EHR data, organized by the format of the input data:

1. **Token-Based Pipeline** (`src/pipelines/token_based/`): Custom transformer decoder, LSTM, and GPT-2 models trained directly on integer token sequences
2. **Text-Based Pipeline** (`src/pipelines/text_based/`): LLM and BERT models fine-tuned on natural language narratives
3. **Embedded-Based Pipeline** (`src/pipelines/embedded_based/`): Transformer encoder/decoder models trained on pre-embedded event sequences

Each pipeline supports both pretraining and fine-tuning for classification tasks.

## Architecture

```
src/
├── pipelines/
│   ├── token_based/       # Custom models on token sequences
│   │   ├── models/        # LSTM, Transformer Decoder, GPT-2
│   │   ├── configs/       # Configuration files
│   │   └── pretrain.py    # Training script
│   │
│   ├── text_based/        # LLM & BERT on text narratives
│   │   ├── configs/       # Configuration files
│   │   ├── pretrain_llm.py      # Unsloth LLM pretraining
│   │   ├── finetune_bert.py     # PubMed BERT fine-tuning
│   │   └── finetune_llm.py      # LLM fine-tuning
│   │
│   ├── embedded_based/    # Models on pre-embedded events
│   │   ├── models/        # Transformer Encoder/Decoder (embedded)
│   │   ├── configs/       # Configuration files
│   │   ├── create_embeddings.py       # Step 1: Create embeddings
│   │   ├── create_vocab_embeddings.py # Precompute vocab embeddings
│   │   ├── pretrain.py                # Step 2: Pretrain decoder
│   │   └── finetune.py                # Step 3: Fine-tune for classification
│   │
│   └── shared/            # Shared components
│       ├── base_models.py
│       ├── blocks/        # Attention mechanisms, etc.
│       └── note_models/   # Clinical notes models
│
├── data/                  # Dataset loaders (work with all pipelines)
├── training/              # Training loops and utilities
└── evaluation/            # Evaluation and visualization tools
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Token-Based Pipeline

Train custom models (LSTM, Transformer, GPT-2) on integer token sequences:

```bash
# Edit config to point to your tokenized data
vim src/pipelines/token_based/configs/encoder_lstm.yaml

# Run training
./run_token_pretrain.sh src/pipelines/token_based/configs/encoder_lstm.yaml my_lstm_exp

# Or use python directly
python -m src.pipelines.token_based.pretrain \
    --config src/pipelines/token_based/configs/encoder_lstm.yaml \
    --experiment_name my_lstm_exp
```

**Configuration Requirements:**
- Point `data.train_dataset_dir` and `data.val_dataset_dir` to your tokenized EHR data
- Set `data.vocab_path` to your vocabulary CSV
- Configure model architecture in `model` section
- Results saved to `results/token_based/{experiment_name}/`

See [Token-Based Pipeline README](src/pipelines/token_based/README.md) for details.

### 2. Text-Based Pipeline

#### Option A: LLM Pretraining (Unsloth)

```bash
# Edit config
vim src/pipelines/text_based/configs/llm_pretrain.yaml

# Run pretraining
./run_text_pretrain.sh src/pipelines/text_based/configs/llm_pretrain.yaml

# Or use python directly
python -m src.pipelines.text_based.pretrain_llm \
    --config_filepath src/pipelines/text_based/configs/llm_pretrain.yaml
```

#### Option B: BERT Fine-Tuning

```bash
# Edit config
vim src/pipelines/text_based/configs/fine-tune-bert.yaml

# Run fine-tuning
./run_text_finetune_bert.sh src/pipelines/text_based/configs/fine-tune-bert.yaml

# Or use python directly
python -m src.pipelines.text_based.finetune_bert \
    --config_filepath src/pipelines/text_based/configs/fine-tune-bert.yaml
```

**Configuration Requirements:**
- Point `data.data_dir` to your tokenized EHR data
- Set vocabulary and lookup file paths
- Configure model (e.g., `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
- Results saved to path specified in `training.output_dir`

See [Text-Based Pipeline README](src/pipelines/text_based/README.md) for details.

### 3. Embedded-Based Pipeline

This pipeline requires three steps:

```bash
# Full pipeline (all 3 steps)
./run_embedded_pipeline.sh \
    src/pipelines/embedded_based/configs/create_embeddings.yaml \
    src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml \
    src/pipelines/embedded_based/configs/finetune_encoder_embedded.yaml

# Or run steps individually:

# Step 1: Create embeddings from events
python -m src.pipelines.embedded_based.create_embeddings \
    --config_filepath src/pipelines/embedded_based/configs/create_embeddings.yaml

# Step 2: Pretrain decoder (optional)
python -m src.pipelines.embedded_based.pretrain \
    --config src/pipelines/embedded_based/configs/pretrain_decoder_embedded.yaml

# Step 3: Fine-tune for classification
python -m src.pipelines.embedded_based.finetune \
    --config src/pipelines/embedded_based/configs/finetune_encoder_embedded.yaml
```

**Configuration Requirements:**
- **Step 1:** Create vocabulary embeddings first using `create_vocab_embeddings.py`
- Point configs to your tokenized EHR data and embedding paths
- Embeddings saved to `data.embedding_output_dir`
- Model checkpoints saved to `training.output_dir`

See [Embedded-Based Pipeline README](src/pipelines/embedded_based/README.md) for details.

## Hyperparameter Sweeps

Use Weights & Biases for hyperparameter optimization:

```bash
# Edit sweep config
vim src/pipelines/text_based/configs/sweep.yaml

# Update the program path to point to your training script
# Launch sweep
wandb sweep src/pipelines/text_based/configs/sweep.yaml

# Run agents
wandb agent <sweep_id>
```

## Evaluation

Visualize training results and run inference:

```bash
# Start visualization server
python -m src.evaluation.visualisation_server.main

# Open browser to http://localhost:5000
```

Features:
- Training loss curves
- Model inference playground
- Rollout simulations
- Benchmark comparisons

## Pipeline Comparison

| Pipeline | Input Format | Models | Use Case |
|----------|-------------|--------|----------|
| **Token-Based** | Integer tokens | LSTM, Transformer Decoder, GPT-2 | Fast iteration, custom architectures |
| **Text-Based** | Natural language | BERT, LLMs (Mistral, Llama) | Leverage pretrained language models |
| **Embedded-Based** | Pre-embedded events | Transformer Encoder/Decoder | Efficient representation learning |

## Data Format

All pipelines expect data from the `CancEHR-tokenization` repository:

- **Train/Val/Test splits**: Separate directories with `.pkl` files
- **Vocabulary**: CSV with token IDs and string representations
- **Labels**: CSV with `subject_id`, `is_case`, `cancerdate`, etc.
- **Lookup tables**: Medical code and lab test descriptions

## Migration from Old Structure

If you have code using the old structure (`src/experiments/`, `src/models/core_models/`):

1. **Update imports:**
   - `src.models.core_models.lstm` → `src.pipelines.token_based.models.lstm`
   - `src.models.core_models.transformer_encoder_embedded` → `src.pipelines.embedded_based.models.transformer_encoder_embedded`

2. **Update config paths:**
   - Old: `src/experiments/configs/encoder_lstm.yaml`
   - New: `src/pipelines/token_based/configs/encoder_lstm.yaml`

3. **Update entry points:**
   - Old: `python -m src.experiments.run --config_name encoder_lstm --experiment_name exp`
   - New: `python -m src.pipelines.token_based.pretrain --config path/to/config.yaml --experiment_name exp`

See `src/experiments/_DEPRECATED_README.md` for full migration guide.

## Development

Each pipeline is self-contained and independently testable:

```bash
# Test token-based pipeline
python -m src.pipelines.token_based.pretrain --config <config> --experiment_name test

# Test text-based pipeline
python -m src.pipelines.text_based.finetune_bert --config_filepath <config>

# Test embedded pipeline
python -m src.pipelines.embedded_based.create_embeddings --config_filepath <config>
```

## Contributing

When adding new models or features:
- Place models in the appropriate pipeline's `models/` directory
- Add configuration examples in the pipeline's `configs/` directory
- Update the pipeline-specific README
- Ensure all scripts have proper `if __name__ == "__main__":` blocks for testing

## License

[Your License Here]

## Citation

[Your Citation Here]

## Contact

[Your Contact Info Here]
