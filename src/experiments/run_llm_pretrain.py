# src/experiments/run_llm_pretrain.py

"""
LLM Continued Pretraining Script

This script performs continued pretraining of a language model on EHR data using:
1. UnifiedEHRDataset in 'text' format for medical code translation
2. HuggingFace's ConstantLengthDataset for efficient sequence packing
3. 1-month temporal cutoff for cancer patients to avoid late-stage signals
4. Standard causal language modeling (next-token prediction)

Usage:
    python -m src.experiments.run_llm_pretrain --config_filepath configs/llm_pretrain.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

from src.data.unified_dataset import UnifiedEHRDataset

class TextDatasetForCLM:
    """
    Wraps UnifiedEHRDataset (text format) for causal language modeling.
    Tokenizes patient narratives on-the-fly and adds EOS tokens as separators.
    """
    def __init__(self, base_dataset, tokenizer, max_length=2048, add_eos=True):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_eos = add_eos
        
        # Pre-filter valid samples
        print(f"Filtering valid samples from {len(base_dataset)} patients...")
        self.valid_indices = []
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item is not None:
                self.valid_indices.append(i)
        print(f"  - Found {len(self.valid_indices)} valid patients")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the actual patient record
        actual_idx = self.valid_indices[idx]
        item = self.base_dataset[actual_idx]
        
        # Add EOS token as separator between patients
        text = item['text']
        if self.add_eos and self.tokenizer.eos_token:
            text = text + self.tokenizer.eos_token
        
        # Tokenize (don't truncate here - let DataCollatorForLanguageModeling handle it)
        # The collator will group texts and chunk them efficiently
        tokenized = self.tokenizer(
            text,
            truncation=False,  
            max_length=self.max_length,
            return_attention_mask=False,  # Collator will create this
            return_token_type_ids=False,
        )
        
        # Return just input_ids - collator handles the rest
        return {
            "input_ids": tokenized["input_ids"]
        }


def create_text_generator(dataset, tokenizer, add_eos=True):
    """
    Creates a generator that yields patient narratives with optional EOS tokens.
    
    Args:
        dataset: UnifiedEHRDataset in 'text' format
        tokenizer: HuggingFace tokenizer
        add_eos: Whether to add EOS token between patients (recommended)
    
    Yields:
        Patient narratives as strings
    """
    eos_token = tokenizer.eos_token if add_eos else ""
    
    for i in range(len(dataset)):
        item = dataset[i]
        if item is not None:
            # Yield patient narrative with separator
            yield item['text'] + eos_token


def create_clm_dataset(base_dataset, tokenizer, max_length):
    """
    Creates a dataset for causal language modeling.
    
    Args:
        base_dataset: UnifiedEHRDataset in 'text' format
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        TextDatasetForCLM
    """
    return TextDatasetForCLM(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        add_eos=True
    )


def main(config_path: str):
    """
    Main function to run LLM continued pretraining.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # 1. Load Config
    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    # 2. Set up WandB (optional)
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "ehr-llm-pretraining")
        run_name = wandb_config.get("run_name", f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        report_to = "wandb"
    else:
        run_name = f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_to = "none"
    
    print(f"Run name: {run_name}")
    
    # 3. Load Tokenizer
    print(f"\nLoading tokenizer: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  - Set pad_token to eos_token: '{tokenizer.eos_token}'")
    
    print(f"  - Vocabulary size: {len(tokenizer)}")
    print(f"  - EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    
    # 4. Create Base Datasets (text format)
    print("\n" + "=" * 80)
    print("Creating datasets in 'text' format...")
    print("=" * 80)
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "format": 'text',  # Use existing text format!
        "cutoff_months": data_config.get("cutoff_months", 1),  # Default 1-month cutoff
        "max_sequence_length": None  # No truncation - we'll pack sequences
    }
    
    print("\nLoading training data...")
    train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    print(f"  - Loaded {len(train_base_dataset)} training patients")
    
    print("\nLoading validation data...")
    val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  - Loaded {len(val_base_dataset)} validation patients")
    
    # 5. Create CLM Datasets
    print("\n" + "=" * 80)
    print("Creating datasets for causal language modeling...")
    print("=" * 80)

    print("\nWrapping training data...")
    train_dataset = create_clm_dataset(
        train_base_dataset, 
        tokenizer, 
        config['model']['max_length']
    )

    print("\nWrapping validation data...")
    val_dataset = create_clm_dataset(
        val_base_dataset, 
        tokenizer, 
        config['model']['max_length']
    )
    
    # 6. Load Model
    print("\n" + "=" * 80)
    print(f"Loading model: {model_config['model_name']}")
    print("=" * 80)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print(f"  - CUDA not available, using CPU")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        torch_dtype=torch.bfloat16 if training_config.get('use_bf16', False) else torch.float32,
    )
    
    print(f"  - Model parameters: {model.num_parameters():,}")
    print(f"  - Model dtype: {model.dtype}")
    print(f"  - Target device: {device}")
    
    # 7. Set Up Training Arguments
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        overwrite_output_dir=training_config.get('overwrite_output_dir', True),
        
        # Training hyperparameters
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=training_config.get('warmup_steps', 500),
        
        # Logging and evaluation
        logging_steps=training_config.get('logging_steps', 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500),
        
        # Saving
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 1000),
        save_total_limit=training_config.get('save_total_limit', 3),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        
        # Performance
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', False),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        
        # Reporting
        report_to=report_to,
        run_name=run_name,
        
        # Other
        remove_unused_columns=False,
    )
    
    print(f"  - Output directory: {training_args.output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  - FP16: {training_args.fp16}, BF16: {training_args.bf16}")
    
    # 8. Create Data Collator (handles batching and label creation)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # 9. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        data_collator=data_collator,
    )
    
    # 10. Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 11. Save Final Model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"  - Model saved to: {final_model_path}")
    
    # 12. Final Evaluation
    if val_dataset:
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        
        eval_results = trainer.evaluate()
        print("\nValidation Results:")
        for key, value in eval_results.items():
            print(f"  - {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Continued Pretraining on EHR Data")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config_filepath)