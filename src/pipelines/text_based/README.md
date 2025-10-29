# Text-Based Pipeline

Fine-tune large language models (LLMs) and BERT on natural language EHR narratives.

## Overview

This pipeline converts tokenized EHR sequences into human-readable text narratives and fine-tunes pretrained language models. It supports:

1. **PubMed BERT fine-tuning** - Fast, efficient fine-tuning for classification
2. **LLM continued pretraining** - Adapt foundation models (Mistral, Llama) to medical domain
3. **LLM fine-tuning** - Task-specific fine-tuning with Unsloth or PEFT

## Workflows

### Workflow 1: BERT Fine-Tuning

Best for: Quick experiments, classification tasks, limited compute

```bash
python -m src.pipelines.text_based.finetune_bert \
    --config_filepath src/pipelines/text_based/configs/fine-tune-bert.yaml
```

### Workflow 2: LLM Continued Pretraining → Fine-Tuning

Best for: Maximum performance, domain adaptation, generative tasks

```bash
# Step 1: Pretrain on medical narratives
python -m src.pipelines.text_based.pretrain_llm \
    --config_filepath src/pipelines/text_based/configs/llm_pretrain.yaml

# Step 2: Fine-tune for specific task
python -m src.pipelines.text_based.finetune_llm \
    --config_filepath src/pipelines/text_based/configs/finetune_llm.yaml
```

## Models

### BERT (`finetune_bert.py`)

- **Supported models**: Any HuggingFace BERT-style model
- **Recommended**: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Method**: Standard fine-tuning with classification head
- **Speed**: Fast (few hours on single GPU)

### LLM (`pretrain_llm.py`, `finetune_llm.py`)

- **Supported models**: Mistral, Llama, Phi, Gemma (via Unsloth)
- **Method**: LoRA/QLoRA for parameter-efficient fine-tuning
- **Frameworks**:
  - **Unsloth** (default): 2x faster, lower memory
  - **PEFT**: Standard HuggingFace implementation

## Configuration

### BERT Fine-Tuning Config

```yaml
model:
  model_name: microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
  num_classes: 2
  max_length: 512

data:
  data_dir: /path/to/tokenized/data/
  vocab_filepath: /path/to/vocab.csv
  labels_filepath: /path/to/labels.csv
  medical_lookup_filepath: /path/to/medical_lookup.csv
  lab_lookup_filepath: /path/to/lab_lookup.csv
  cutoff_months: 1  # Temporal cutoff before diagnosis

training:
  output_dir: results/text_based/bert_finetune/
  overwrite_output_dir: true
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  weight_decay: 0.01

wandb:
  project: cancer-ehr-bert
  run_name: pubmedbert_1month_cutoff
```

### LLM Pretraining Config

```yaml
model:
  model_name: unsloth/mistral-7b-v0.3
  max_length: 2048

data:
  data_dir: /path/to/tokenized/data/
  vocab_filepath: /path/to/vocab.csv
  labels_filepath: /path/to/labels.csv
  medical_lookup_filepath: /path/to/medical_lookup.csv
  lab_lookup_filepath: /path/to/lab_lookup.csv
  cutoff_months: 1  # Always 1 for pretraining (cancer patients)

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
  use_rslora: true

training:
  output_dir: results/text_based/llm_pretrain/
  framework: unsloth  # or 'peft'
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  weight_decay: 0.01
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  load_in_4bit: true
  fp16: false
  bf16: true
  warmup_steps: 500
  logging_steps: 10
  eval_steps: 500
  save_steps: 1000

wandb:
  enabled: true
  project: cancer-ehr-llm
```

## Data Format

### UnifiedEHRDataset (Text Mode)

The dataset automatically converts token sequences to natural language:

**Input**: Tokenized EHR
```
[AGE_decile_5, GENDER//FEMALE, MEDICAL//C10, LAB//GLUCOSE_5, ...]
```

**Output**: Natural language narrative
```
"AGE_decile 5, FEMALE, Diseases of the circulatory system, Glucose 5, ..."
```

**Temporal Cutoff**: Removes events within X months before cancer diagnosis (cancer patients only)

### Text Generation

Medical codes are translated using lookup tables:
- `MEDICAL//C10` → "Diseases of the circulatory system"
- `LAB//GLUCOSE` → "Glucose"
- `AGE_decile_5` → "AGE_decile 5"

## Output

### BERT Fine-Tuning

```
results/text_based/bert_finetune/
├── config.json
├── pytorch_model.bin
├── training_args.bin
└── trainer_state.json
```

### LLM Pretraining

```
results/text_based/llm_pretrain/
├── checkpoint-1000/
├── checkpoint-2000/
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── runs/  # Tensorboard logs
```

## Hyperparameter Sweeps

Use Weights & Biases for automated hyperparameter optimization:

```yaml
# sweep.yaml
method: bayes
metric:
  name: eval/loss
  goal: minimize

parameters:
  training.learning_rate:
    distribution: log_uniform_values
    min: 1e-6
    max: 5e-5
  
  lora.r:
    values: [8, 16, 32]
  
  lora.lora_dropout:
    values: [0.05, 0.1, 0.2]
```

```bash
# Launch sweep
wandb sweep src/pipelines/text_based/configs/sweep.yaml

# Run agents
wandb agent <sweep_id>
```

## Inference

### BERT

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("results/text_based/bert_finetune/")
model = AutoModelForSequenceClassification.from_pretrained("results/text_based/bert_finetune/")

text = "AGE_decile 5, FEMALE, Hypertension, Diabetes, ..."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1)
```

### LLM

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="results/text_based/llm_pretrain/final_model/",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

prompt = "AGE_decile 5, FEMALE, Hypertension, Diabetes"
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

## Tips

### BERT Fine-Tuning

- **Sequence length**: Start with 512, increase if needed (up to model's max)
- **Batch size**: Use largest that fits in memory (16-32 typical)
- **Learning rate**: 2e-5 to 5e-5 works well for BERT
- **Epochs**: 3-10 usually sufficient (watch for overfitting)

### LLM Training

- **Memory optimization**:
  - Enable `load_in_4bit: true` (QLoRA)
  - Use `gradient_checkpointing: true`
  - Increase `gradient_accumulation_steps`
  - Reduce `batch_size`

- **Speed optimization**:
  - Use Unsloth (2x faster than PEFT)
  - Enable `bf16: true` on Ampere GPUs
  - Increase `batch_size` if memory allows

- **Quality**:
  - Higher `lora.r` = more parameters = better quality (but slower)
  - Lower `lora_dropout` for smaller datasets
  - Use `use_rslora: true` for stable training

## Frameworks

### Unsloth (Recommended)

- **Pros**: 2x faster, 50% less memory, easy to use
- **Cons**: Limited model support
- **Best for**: Mistral, Llama, Phi models

### PEFT

- **Pros**: Works with any HuggingFace model, standard implementation
- **Cons**: Slower, more memory
- **Best for**: Unusual models, maximum compatibility

Switch between frameworks in config:

```yaml
training:
  framework: unsloth  # or 'peft'
```

## Troubleshooting

**Issue**: CUDA out of memory (LLM)
- Enable `load_in_4bit: true`
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing`
- Reduce `model.max_length`

**Issue**: Training is slow
- Use Unsloth instead of PEFT
- Enable `bf16: true` (Ampere GPU)
- Increase `batch_size`
- Reduce `logging_steps` and `eval_steps`

**Issue**: Model not improving
- Check `cutoff_months` - may be leaking label information
- Increase `warmup_steps`
- Try different `learning_rate`
- Check for class imbalance

**Issue**: Text narratives look wrong
- Verify lookup table paths in config
- Check `medical_lookup_filepath` and `lab_lookup_filepath`
- Ensure medical codes match vocabulary

## Resources

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PubMed BERT Paper](https://arxiv.org/abs/2007.15779)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

