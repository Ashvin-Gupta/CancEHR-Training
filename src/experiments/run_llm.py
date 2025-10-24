# src/experiments/run_llm.py
import argparse
import yaml
import os
import wandb
from huggingface_hub import login
from transformers import AutoTokenizer
from datasets import Dataset

# Import the new, separated trainer functions
from src.training.unsloth_trainer import run_unsloth_training
from src.training.peft_trainer import run_peft_training
from src.data.unified_dataset import UnifiedEHRDataset # Your dataset

def extract_text(base_dataset, tokenizer):
    """Extracts all valid text narratives from the base dataset and adds EOS token."""
    text_list = []
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
    
    print(f"  - Processing {len(base_dataset)} patients...")
    for i in range(len(base_dataset)):
        item = base_dataset[i]
        if item is not None:
            text_list.append(item['text'] + eos_token)
    print(f"  - Extracted {len(text_list)} valid narratives.")
    return text_list

def main(config_path: str):
    # 1. Load Config
    print("=" * 80)
    print("LLM Continued Pretraining - Main Controller")
    print("=" * 80)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
    # 2. Set up WandB and Hugging Face token
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "ehr-llm-pretraining")
        run_name = wandb_config.get("run_name")
        report_to = "wandb"
    else:
        run_name = "local-run"
        report_to = "none"
    
    # (Your HF login logic)
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f: hf_token = f.readline().split('=')[1].strip('"')
            login(token=str(hf_token))
            print("HuggingFace login successful.")
        except Exception as e:
            print(f"Failed to read or login with HF token: {e}")
            
    # 3. Load Tokenizer (Common to both frameworks)
    print(f"\nLoading tokenizer: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  - Set pad_token to eos_token: '{tokenizer.eos_token}'")

    # 4. Load Base Datasets (Common to both frameworks)
    print("\n" + "=" * 80)
    print("Loading base UnifiedEHRDataset...")
    print("=" * 80)
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "format": 'text',
        "cutoff_months": data_config.get("cutoff_months", 1),
        "max_sequence_length": None
    }
    train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  - Loaded {len(train_base_dataset)} train, {len(val_base_dataset)} val patients.")

    # 5. Extract Text & Convert to HF Dataset (Common to both frameworks)
    print("\n" + "=" * 80)
    print("Extracting text and building HuggingFace Datasets...")
    print("=" * 80)
    
    train_text_list = extract_text(train_base_dataset, tokenizer)
    val_text_list = extract_text(val_base_dataset, tokenizer)

    # (Your sanity check print)
    for i in range(min(3, len(train_text_list))):
        print(f"\n--- PATIENT {i} (Last 1000 chars) ---")
        print(f"{train_text_list[i][-1000:]}...")

    train_dataset = Dataset.from_dict({ "text": train_text_list })
    val_dataset = Dataset.from_dict({ "text": val_text_list })
    print(f"\n  - Created HF training dataset: {train_dataset}")
    print(f"  - Created HF validation dataset: {val_dataset}")
    
    # 6. Branch to the correct trainer
    framework = training_config.get('framework', 'unsloth').lower()
    print("\n" + "=" * 80)
    print(f"Starting training with framework: {framework}")
    print("=" * 80)
    
    if framework == 'unsloth':
        run_unsloth_training(config, tokenizer, train_dataset, val_dataset)
    elif framework == 'peft':
        run_peft_training(config, tokenizer, train_dataset, val_dataset)
    else:
        raise ValueError(f"Unknown framework '{framework}'. Must be 'unsloth' or 'peft'.")
        
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Continued Pretraining Controller")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    main(args.config_filepath)