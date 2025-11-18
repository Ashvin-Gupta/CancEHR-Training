# src/pipelines/text_based/finetune_llm_classifier.py

"""
Main script for fine-tuning a pretrained LLM for binary classification.

Loads a pretrained LLM with LoRA adapters and extended tokenizer,
freezes the LLM, and trains only a classification head on top.
"""

import argparse
import yaml
import os
import wandb
from huggingface_hub import login
from unsloth import FastLanguageModel

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier, run_classification_training
from transformers import AutoTokenizer

def load_model(config: dict):
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    wandb_config = config.get('wandb', {})
    
    print("\n" + "=" * 80)
    print(f"Loading pretrained model from: {model_config['pretrained_checkpoint']}")
    print("=" * 80)

    # STEP A: Explicitly load the correct Base Model (Qwen 2.5)
    # We force the model_name to be the base model, NOT the checkpoint path yet.
    base_model_name = model_config['unsloth_model']
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name, 
        max_seq_length=data_config['max_length'],
        dtype=None,
        load_in_4bit=training_config.get('load_in_4bit', True),
    )
    print(f'Original tokenizer size: {len(tokenizer)}')

    # STEP B: Load the tokenizer from your checkpoint to get the new vocab size
    # This ensures we have the 151673 size including your 4 special tokens
    checkpoint_tokenizer = AutoTokenizer.from_pretrained(model_config['pretrained_checkpoint'])
    print(f'Checkpoint tokenizer size: {len(checkpoint_tokenizer)}')
    # Replace the standard tokenizer with your extended one
    tokenizer = checkpoint_tokenizer
    print(f'New tokenizer size: {len(tokenizer)}')
    # STEP C: Resize the model embeddings to match the checkpoint (151673)
    model.resize_token_embeddings(len(tokenizer))
    
    # STEP D: Load the adapters (PeftModel)
    # Since FastLanguageModel wraps the model, we access the internal model to load adapters if needed,
    # but usually, we can just load the adapter on top.
    model.load_adapter(model_config['pretrained_checkpoint'])

    print(f"  - Loaded model with {len(tokenizer)} tokens in vocabulary")
    print(f"  - Model type: {type(model).__name__}")
    return model, tokenizer

def main(config_path: str):
    print("=" * 80)
    print("LLM Binary Classification Fine-tuning")
    print("=" * 80)
    
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    wandb_config = config.get('wandb', {})
    
    # 2. Set up WandB
    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "llm-classification")
        run_name = wandb_config.get("run_name")
        if run_name is None:
            # Auto-generate run name
            run_name = f"classifier_{config.get('name', 'default')}"
        wandb_config['run_name'] = run_name
        print(f"\nWandB enabled - Project: {wandb_config['project']}, Run: {run_name}")
    
    # 3. HuggingFace Login
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                hf_token = f.readline().split('=')[1].strip('"')
            login(token=str(hf_token))
            print("HuggingFace login successful.")
        except Exception as e:
            print(f"Failed to read or login with HF token: {e}")
    
    model, tokenizer = load_model(config)
    
    # 5. Wrap model with classification head
    print("\n" + "=" * 80)
    print("Creating LLM Classifier wrapper...")
    print("=" * 80)
    
    classifier_model = LLMClassifier(
        base_model=model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=model_config.get('freeze_llm', True)
    )
    
    # 6. Load Datasets
    print("\n" + "=" * 80)
    print("Loading datasets...")
    print("=" * 80)
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "region_lookup_file": data_config["region_lookup_filepath"],
        "time_lookup_file": data_config["time_lookup_filepath"],
        "format": 'text',  # Return text narratives
        "cutoff_months": data_config.get("cutoff_months", 1),
        "max_sequence_length": None,  # No truncation at dataset level
        "tokenizer": None  # Not needed for 'text' format
    }
    
    train_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    val_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    
    print(f"  - Train dataset: {len(train_dataset)} patients")
    print(f"  - Validation dataset: {len(val_dataset)} patients")
    
    # Print a few examples
    print("\n" + "=" * 80)
    print("Sample data (last 500 chars):")
    print("=" * 80)
    for i in range(min(2, len(train_dataset))):
        sample = train_dataset[i]
        if sample is not None:
            text_preview = sample['text'][-500:] if len(sample['text']) > 500 else sample['text']
            label = sample['label'].item()
            binary_label = 1 if label > 0 else 0
            print(f"\nPatient {i}:")
            print(f"  Label: {label} (binary: {binary_label})")
            print(f"  Text: ...{text_preview}")
    
    # 7. Create Data Collator
    print("\n" + "=" * 80)
    print("Creating data collator...")
    print("=" * 80)
    
    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config['max_length'],
        binary_classification=True
    )
    print(f"  - Max sequence length: {data_config['max_length']}")
    print(f"  - Binary classification: True")
    
    # 8. Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer, eval_results = run_classification_training(
        config=config,
        model=classifier_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn
    )
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal model saved to: {training_config['output_dir']}/final_model")
    
    return trainer, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Binary Classification Fine-tuning")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    main(args.config_filepath)


