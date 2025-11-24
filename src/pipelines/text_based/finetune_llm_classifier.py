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
import torch
import pprint
from huggingface_hub import login

from src.data.unified_dataset import UnifiedEHRDataset
from src.data.classification_collator import ClassificationCollator
from src.training.classification_trainer import LLMClassifier, run_classification_training
from src.utils.load_LoRA_model import load_LoRA_model
from src.pipelines.text_based.token_adaption2 import EHRTokenExtensionStaticTokenizer

EXPERIMENT_NO_PRETRAIN = "no_pretrain"
EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER = "pretrained_cls"
EXPERIMENT_PRETRAIN_CLASSIFIER_LORA = "pretrained_cls_lora"


def load_model_for_mode(config: dict, experiment_mode: str):
    """
    Load the correct model/tokenizer pair based on the experiment mode.
    """
    data_config = config['data']
    training_config = config['training']
    model_config = config['model']
    
    if experiment_mode == EXPERIMENT_NO_PRETRAIN:
        translator = EHRTokenExtensionStaticTokenizer()
        model, tokenizer = translator.extend_tokenizer(
            model_name=model_config['unsloth_model'],
            max_seq_length=data_config['max_length'],
            load_in_4bit=training_config.get('load_in_4bit', True)
        )
        print("\nLoaded base model without continued pretraining. Only the classifier head will train.")
        return model, tokenizer
    
    if experiment_mode in (EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER, EXPERIMENT_PRETRAIN_CLASSIFIER_LORA):
        if not model_config.get('pretrained_checkpoint'):
            raise ValueError(
                f"'model.pretrained_checkpoint' must be set for experiment mode '{experiment_mode}'."
            )
        return load_LoRA_model(config)
    
    raise ValueError(f"Unknown experiment mode '{experiment_mode}'.")


def main(config_path: str):
    print("=" * 80)
    print("LLM Binary Classification Fine-tuning")
    print("=" * 80)
    
    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print('Loaded configuration')
    pprint.pprint(config)
    print("=" * 80)
    
    model_config = config['model']
    experiment_config = config.get('experiment', {})
    experiment_mode = experiment_config.get('mode', EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER)
    data_config = config['data']
    training_config = config['training']
    wandb_config = config.get('wandb', {})
    
    mode_msg = {
        EXPERIMENT_NO_PRETRAIN: "No continued pretraining - classifier head only.",
        EXPERIMENT_PRETRAIN_ONLY_CLASSIFIER: "Using continued-pretrained checkpoint - classifier head only.",
        EXPERIMENT_PRETRAIN_CLASSIFIER_LORA: "Using continued-pretrained checkpoint - training classifier head + LoRA adapters."
    }[experiment_mode]
    print(f"Experiment mode: {experiment_mode} -> {mode_msg}")
    
    # 2. Set up WandB
    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "llm-classification")
        run_name = wandb_config.get("run_name")
        if run_name is None:
            # Auto-generate run name
            run_name = f"classifier_{config.get('name', 'default')}"
        wandb_config['run_name'] = run_name
        print(f"\nWandB enabled - Project: {wandb_config['project']}, Run: {run_name}")
    
    # 3. HuggingFace Login (skip for pure classifier-only mode unless forced)
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    hf_token = None
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                hf_token = f.readline().split('=')[1].strip('"')
        except Exception as e:
            print(f"Failed to read HF token: {e}")
    
    require_login = experiment_config.get('force_hf_login', False) or experiment_mode != EXPERIMENT_NO_PRETRAIN
    if hf_token and require_login:
        try:
            login(token=str(hf_token))
            print("HuggingFace login successful.")
        except Exception as e:
            print(f"Failed to login to HuggingFace: {e}")
    elif require_login:
        print("No HuggingFace token available but login required for this mode.")
    else:
        print("Skipping HuggingFace login for classifier-only experiment.")
    
    model, tokenizer = load_model_for_mode(config, experiment_mode)
    
    # 5. Wrap model with classification head
    print("\n" + "=" * 80)
    print("Creating LLM Classifier wrapper...")
    print("=" * 80)
    
    multi_label_task = bool(training_config.get('multi_label', False))
    if multi_label_task:
        print("Multi-label flag detected. Ensure datasets/collators emit multi-hot labels. Current metrics remain binary.")
    
    train_lora_adapters = bool(model_config.get('train_lora', False))
    if 'freeze_lora' in model_config:
        train_lora_adapters = not bool(model_config['freeze_lora'])
    if experiment_mode == EXPERIMENT_PRETRAIN_CLASSIFIER_LORA:
        train_lora_adapters = True
    
    freeze_llm = bool(model_config.get('freeze_llm', True))
    
    trainable_keywords = ["lora_"] if train_lora_adapters else None
    
    classifier_model = LLMClassifier(
        base_model=model,
        hidden_size=model_config['hidden_size'],
        num_labels=model_config['num_labels'],
        freeze_base=freeze_llm,
        trainable_param_keywords=trainable_keywords,
        multi_label=multi_label_task
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
            label_tensor = sample['label']
            scalar_label = None
            label_value = None
            if torch.is_tensor(label_tensor):
                if label_tensor.dim() == 0:
                    scalar_label = label_tensor.item()
                    label_value = scalar_label
                else:
                    label_value = label_tensor.tolist()
            else:
                scalar_label = label_tensor
                label_value = scalar_label
            
            if label_value is None:
                label_value = label_tensor
            binary_label = 1 if (scalar_label is not None and scalar_label > 0) else label_value
            print(f"\nPatient {i}:")
            print(f"  Label: {label_value} (binary view: {binary_label})")
            print(f"  Text: ...{text_preview}")
    
    # 7. Create Data Collator
    print("\n" + "=" * 80)
    print("Creating data collator...")
    print("=" * 80)
    
    binary_classification = not multi_label_task
    collate_fn = ClassificationCollator(
        tokenizer=tokenizer,
        max_length=data_config['max_length'],
        binary_classification=binary_classification
    )
    print(f"  - Max sequence length: {data_config['max_length']}")
    print(f"  - Binary classification: {binary_classification}")
    
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


