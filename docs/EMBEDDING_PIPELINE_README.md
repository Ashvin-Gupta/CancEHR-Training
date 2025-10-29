# Embedding-Based Model Pipeline

This document explains how to use the new embedding-based pipeline for training transformer models on pre-computed E5 embeddings of EHR data.

## Overview

The pipeline supports:
- **Encoder models**: For classification tasks using bidirectional attention
- **Decoder models**: For autoregressive next-event prediction using causal attention
- **Two-stage training**: Pretraining (unsupervised) → Fine-tuning (supervised classification)

## Pipeline Architecture

```
EHR Data → UnifiedEHRDataset (events_with_ids) → E5 Embeddings → 
.pt files (embeddings + token_ids + labels) → PreEmbeddedDataset →
Transformer Models (Encoder/Decoder) → Classification/Prediction
```

## Step-by-Step Usage

### Step 1: Create Embedding Corpus

First, create E5 embeddings for all your EHR data:

```bash
python -m src.experiments.create_embedding_corpus \
    --config_filepath src/experiments/configs/embed_text.yaml
```

**What this does:**
- Loads EHR data using `UnifiedEHRDataset` with `format='events_with_ids'`
- Translates medical codes to natural language text
- Creates E5 embeddings for each event (768-dimensional)
- Saves `.pt` files with: embeddings (N×768), token_ids (N,), labels

**Output structure:**
```
/path/to/embedded/data/
  train/
    patient_0.pt
    patient_1.pt
    ...
  tuning/
    patient_0.pt
    ...
  held_out/
    patient_0.pt
    ...
```

**Config file** (`src/experiments/configs/embed_text.yaml`):
```yaml
model:
  model_name: "intfloat/e5-large-v2"
  device: "cuda"

data:
  format: "events_with_ids"  # Now uses new format!
  embedding_output_dir: "/path/to/output"
  data_dir: "/path/to/tokenized/data"
  vocab_filepath: "/path/to/vocab.csv"
  labels_filepath: "/path/to/labels.csv"
  medical_lookup_filepath: "/path/to/medical_dict.csv"
  lab_lookup_filepath: "/path/to/lab_lookup.csv"
```

### Step 2A: Pretrain Decoder Model (Optional)

Pretrain a transformer decoder using autoregressive next-event prediction:

```bash
python -m src.experiments.pretrain_embedded \
    --config src/experiments/configs/pretrain_decoder_embedded.yaml
```

**What this does:**
- Trains the model to predict next token ID given previous embeddings
- Uses causal (unidirectional) attention
- No labels used (unsupervised learning)
- Saves checkpoints to `output_dir`

**Config file** (`src/experiments/configs/pretrain_decoder_embedded.yaml`):
```yaml
model:
  type: "transformer_decoder_embedded"
  embedding_dim: 768
  model_dim: 512
  n_layers: 6
  n_heads: 8
  vocab_size: 10000  # UPDATE based on your vocab

training:
  task: "autoregressive"
  batch_size: 16
  epochs: 50
  learning_rate: 1e-4

data:
  embedding_output_dir: "/path/to/embedded/data"
```

### Step 2B: Fine-tune Encoder for Classification

Train an encoder model directly for classification (no pretraining needed):

```bash
python -m src.experiments.finetune_embedded \
    --config src/experiments/configs/finetune_encoder_embedded.yaml
```

**What this does:**
- Trains a transformer encoder for binary classification
- Uses bidirectional attention (can see future events)
- Supervised learning with cancer labels
- Saves best model based on F1 score

**Config file** (`src/experiments/configs/finetune_encoder_embedded.yaml`):
```yaml
model:
  type: "transformer_encoder_embedded"
  embedding_dim: 768
  model_dim: 512
  n_layers: 6
  n_heads: 8
  num_classes: 2
  pooling: "mean"  # 'mean', 'cls', or 'max'

training:
  task: "classification"
  batch_size: 16
  epochs: 30
  learning_rate: 5e-5
  pretrained_checkpoint: null  # Optional

data:
  embedding_output_dir: "/path/to/embedded/data"
```

### Step 3: Fine-tune Pretrained Decoder (Optional)

If you pretrained a decoder, fine-tune it for classification:

```bash
python -m src.experiments.finetune_embedded \
    --config src/experiments/configs/finetune_decoder_embedded.yaml
```

**Config file** (`src/experiments/configs/finetune_decoder_embedded.yaml`):
```yaml
model:
  type: "transformer_decoder_embedded"
  add_classification_head: true
  num_classes: 2
  # ... other params match pretraining

training:
  pretrained_checkpoint: "./outputs/pretrain_decoder_embedded/best_checkpoint.pt"
  # ... other params
```

## Key Components

### 1. Data Components

#### `UnifiedEHRDataset` (Modified)
- **New format**: `'events_with_ids'`
- Returns: `{"events": List[str], "token_ids": Tensor, "label": Tensor}`
- Use this format when creating embeddings

#### `PreEmbeddedDataset`
- Loads `.pt` files created by `create_embedding_corpus.py`
- **Task parameter**: `'classification'` or `'autoregressive'`
- Returns appropriate data for each task

#### `embedding_collator.py`
- **`classification_collate_fn`**: Pads embeddings, creates padding masks
- **`autoregressive_collate_fn`**: Creates input/target pairs for next-token prediction
- Handles variable-length sequences dynamically

### 2. Model Components

#### `TransformerEncoderEmbedded`
- **Architecture**: Embeddings → Projection → Pos Encoding → Encoder Blocks → Pooling → Classifier
- **Attention**: Bidirectional (can see all positions)
- **Use case**: Classification tasks
- **Pooling options**: mean, cls, max

#### `TransformerDecoderEmbedded`
- **Architecture**: Embeddings → Projection → Pos Encoding → Decoder Blocks → LM Head
- **Attention**: Causal (only sees past positions)
- **Use case**: Autoregressive prediction, or classification with pretrained weights
- **Optional**: Can add classification head on top

### 3. Training Scripts

#### `pretrain_embedded.py`
- Unsupervised pretraining
- Next-token prediction for decoders
- Tracks: loss, perplexity
- Saves: best checkpoint, latest checkpoint, periodic checkpoints

#### `finetune_embedded.py`
- Supervised classification
- Can load pretrained checkpoints
- Tracks: loss, accuracy, precision, recall, F1, AUC
- Early stopping based on F1 score

## Model Comparison

| Feature | Encoder | Decoder |
|---------|---------|---------|
| **Attention** | Bidirectional | Causal |
| **Pretraining** | Optional (MLM) | Autoregressive |
| **Fine-tuning** | Direct classification | Classification head |
| **Use Case** | Best for classification | Best after pretraining |
| **Context** | Sees future events | Only past events |

## Workflow Examples

### Workflow 1: Quick Classification (Encoder)
```bash
# 1. Create embeddings
python -m src.experiments.create_embedding_corpus --config configs/embed_text.yaml

# 2. Train encoder directly
python -m src.experiments.finetune_embedded --config configs/finetune_encoder_embedded.yaml
```

### Workflow 2: Pretrain → Fine-tune (Decoder)
```bash
# 1. Create embeddings
python -m src.experiments.create_embedding_corpus --config configs/embed_text.yaml

# 2. Pretrain decoder
python -m src.experiments.pretrain_embedded --config configs/pretrain_decoder_embedded.yaml

# 3. Fine-tune decoder
python -m src.experiments.finetune_embedded --config configs/finetune_decoder_embedded.yaml
```

## Important Notes

### Temporal Cutoff
- Apply temporal cutoff (e.g., 12 months) when creating embeddings
- This removes recent events before cancer diagnosis to prevent label leakage
- Currently handled in `UnifiedEHRDataset` via `cutoff_months` parameter

### Padding
- Sequences are dynamically padded to the longest in each batch
- Padding masks ensure models don't attend to padding positions
- Autoregressive loss ignores padding tokens (index 0)

### Class Imbalance
- Use `class_weights` in config to handle imbalanced datasets
- Example: `class_weights: [1.0, 3.0]` weights positive class 3x

### Memory Management
- Reduce `batch_size` if you run out of GPU memory
- Use `num_workers` for faster data loading
- Embeddings are pre-computed, so training is fast!

## Extending the Pipeline

### Add New Model Architectures
1. Create new model in `src/models/core_models/`
2. Inherit from `BaseNightingaleModel`
3. Register with `@register_model("your_model_name")`
4. Implement required methods: `required_config_keys()`, `required_input_keys()`, `forward()`

### Add New Tasks
1. Add new task type in `PreEmbeddedDataset.__getitem__()`
2. Create new collate function in `embedding_collator.py`
3. Update training scripts to handle new task

### Add State Space Models (SSMs)
- SSMs can use the same data pipeline
- Create new model class (e.g., `MambaEmbedded`)
- Use `classification_collate_fn` for inputs
- No need for causal masks (SSMs handle sequence order differently)

## Troubleshooting

### "Split directory does not exist"
- Check `embedding_output_dir` path in config
- Ensure you ran `create_embedding_corpus.py` first

### "Unknown model type"
- Check `model.type` in config matches registered model name
- Ensure model file is imported in training script

### Out of memory
- Reduce `batch_size`
- Reduce `model_dim` or `n_layers`
- Use gradient accumulation

### Poor performance
- Try pretraining first (especially for small datasets)
- Adjust learning rate (lower for fine-tuning)
- Check class balance and use class weights
- Increase model capacity (more layers/heads)

## File Structure

```
src/
├── data/
│   ├── unified_dataset.py          # Modified with events_with_ids format
│   ├── embedded_dataset.py         # Loads pre-embedded data
│   └── embedding_collator.py       # Collate functions for batching
├── models/
│   └── core_models/
│       ├── transformer_encoder_embedded.py  # Encoder for classification
│       └── transformer_decoder_embedded.py  # Decoder for autoregressive
└── experiments/
    ├── create_embedding_corpus.py  # Step 1: Create embeddings
    ├── pretrain_embedded.py        # Step 2: Pretrain (optional)
    ├── finetune_embedded.py        # Step 3: Fine-tune
    └── configs/
        ├── embed_text.yaml
        ├── pretrain_decoder_embedded.yaml
        ├── finetune_encoder_embedded.yaml
        └── finetune_decoder_embedded.yaml
```

## Questions?

- Check config files for all available options
- Look at model `__main__` blocks for usage examples
- Review docstrings in model classes for architecture details

