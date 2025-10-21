# LLM Pretraining Format Guide

## Overview

The `pretrain` format in `UnifiedEHRDataset` is designed for continued pretraining of large language models (LLMs) on EHR data. It follows the Nightingale training approach with patient-level shuffling and random window sampling.

## Key Features

1. **1-month temporal cutoff for cancer patients**: Automatically removes the last month before diagnosis to prevent the model from learning obvious late-stage signals
2. **Random window sampling**: Each epoch samples different random windows from patient narratives for data augmentation
3. **Patient-level shuffling**: DataLoader shuffles patients at the start of each epoch
4. **100% data coverage**: All patient data is eventually used across multiple epochs
5. **Causal language modeling**: Standard next-token prediction setup

## How It Works

### Data Flow

1. **Load patient records** → Full event sequences from pickle files
2. **Apply temporal cutoff** → Remove last 1 month for cancer patients (label > 0)
3. **Translate to text** → Convert medical codes to natural language
4. **Tokenize** → Use LLM tokenizer to get subword tokens
5. **Random sampling** → Sample a window of C tokens from the full narrative
6. **Collate & batch** → Pad sequences and create attention masks

### Random Window Sampling

```python
# If patient narrative has 3000 LLM tokens and max_sequence_length=2048:
# Random start: anywhere from 0 to 952
# Sample: tokens[start_idx:start_idx+2048]

# Each __getitem__ call → different random window
# Over many epochs → all parts of the narrative are seen
```

## Usage

### Basic Setup

```python
from transformers import AutoTokenizer
from src.data.unified_dataset import UnifiedEHRDataset
from src.data.pretrain_collator import PretrainCollator
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create dataset
dataset = UnifiedEHRDataset(
    data_dir="/path/to/data",
    vocab_file="/path/to/vocab.csv",
    labels_file="/path/to/labels.csv",
    medical_lookup_file="/path/to/medical_lookup.csv",
    lab_lookup_file="/path/to/lab_lookup.csv",
    split='train',
    format='pretrain',
    max_sequence_length=2048,  # Context window C
    tokenizer=tokenizer,
    cutoff_months=None  # Uses 1-month cutoff automatically
)

# Create collator
collator = PretrainCollator(tokenizer)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,  # Patient-level shuffling
    collate_fn=collator,
    num_workers=4
)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # batch["input_ids"]: [B, C] - input token IDs
        # batch["attention_mask"]: [B, C] - attention mask
        # batch["labels"]: [B, C] - labels for causal LM (same as input_ids)
        
        outputs = model(**batch)
        loss = outputs.loss
        # ... training step
```

### With Hugging Face Trainer

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
)

trainer.train()
```

## Format Comparison

| Feature | `tokens` | `text` | `pretrain` |
|---------|----------|--------|------------|
| **Output** | Integer token IDs | Raw text strings | LLM-tokenized sequences |
| **Use case** | Custom models | Classification fine-tuning | Continued pretraining |
| **Temporal cutoff** | Configurable | Configurable | Fixed 1-month for cancer |
| **Sequence handling** | Truncate at max_length | Full narrative | Random window sampling |
| **Tokenization** | None (already tokens) | None (done downstream) | LLM tokenizer |
| **Data augmentation** | No | No | Yes (random windows) |
| **Return format** | `{"tokens", "label"}` | `{"text", "label"}` | `{"input_ids", "label", "subject_id"}` |

## Design Decisions

### Why 1-month cutoff for cancer patients?

During pretraining, we want the model to learn general EHR patterns without overfitting to obvious late-stage cancer symptoms. The 1-month cutoff removes the immediate pre-diagnosis period while preserving earlier signals.

### Why random window sampling?

- **Data augmentation**: Each epoch sees different parts of each patient's history
- **100% coverage**: Over many epochs, all data is used
- **Memory efficiency**: Only need to store one window per patient in memory
- **Consistent with Nightingale**: Matches the proven training approach

### Why sample in LLM-token-space vs event-space?

Simpler implementation and works naturally with LLM tokenization. You can control coverage by adjusting `max_sequence_length`:
- 512 tokens ≈ 100-200 medical events
- 2048 tokens ≈ 400-800 medical events
- 4096 tokens ≈ 800-1600 medical events

## Next Steps: Classification Fine-tuning

After pretraining, you can add a classification head:

```python
# Switch to 'text' format for classification
dataset = UnifiedEHRDataset(
    format='text',
    cutoff_months=6,  # Variable cutoff for classification
    ...
)

# Add classification head
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    "./output/pretrained_model",
    num_labels=5  # Number of cancer types + control
)

# Fine-tune with classification loss
# ... (see run_hf_finetune.py for example)
```

## Testing

Run the verification script to test the pretrain format:

```bash
python -m src.data.unified_dataloader
```

This will:
1. Test all three formats (tokens, text, pretrain)
2. Verify temporal truncation is working
3. Verify random window sampling produces different batches
4. Show sample outputs

## Troubleshooting

### "tokenizer must be provided for format='pretrain'"
Make sure to pass a tokenizer instance when creating the dataset:
```python
dataset = UnifiedEHRDataset(..., tokenizer=tokenizer)
```

### Pad token not set
Some tokenizers (like GPT-2) don't have a pad token by default:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### Out of memory
Reduce `max_sequence_length` or `batch_size`:
```python
dataset = UnifiedEHRDataset(..., max_sequence_length=1024)  # Instead of 2048
dataloader = DataLoader(..., batch_size=2)  # Instead of 4
```

## References

- Nightingale dataset: `src/data/dataset.py` (NightingaleTrainingDataset)
- Random window sampling: Lines 219-235 in dataset.py
- Unified dataset: `src/data/unified_dataset.py`
- Pretrain collator: `src/data/pretrain_collator.py`

