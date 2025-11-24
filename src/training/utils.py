import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import yaml
import math
from torch.optim.lr_scheduler import LambdaLR


def get_nested_value(data: dict, key: str):
    """
    Get a nested value from a dictionary using dot notation.
    Supports any level of nesting: 'simple_key', 'ehr.input_token_ids', 'deep.nested.key.path'
    """
    keys = key.split('.')
    current = data
    
    for k in keys:
        current = current[k]
    
    return current


def create_nested_dict(flat_dict: dict) -> dict:
    """
    Convert a flat dictionary with dot-notation keys to a nested dictionary.
    Example: {'ehr.input_token_ids': tensor, 'clinical_notes.token_ids': tensor}
    becomes: {'ehr': {'input_token_ids': tensor}, 'clinical_notes': {'token_ids': tensor}}
    """
    nested = {}
    
    for key, value in flat_dict.items():
        keys = key.split('.')
        current = nested
        
        # Navigate/create the nested structure
        for k in keys[:-1]:  # All keys except the last one
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
    
    return nested

def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=None, lr_min_ratio=0.1):
    """
    Linear warmup from 0 -> lr_max over warmup_steps, then cosine decay to lr_min_ratio * lr_max.
    Call sched.step() **once per optimizer step**.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        total_steps (int): The total number of steps to schedule.
        warmup_steps (int): The number of steps to warmup.
        lr_min_ratio (float): The minimum ratio of the learning rate to the maximum learning rate.

    Returns:
        LambdaLR: The scheduler.
    """
    if warmup_steps is None:
        warmup_steps = max(int(0.02 * total_steps), 2000)

    def lr_lambda(step):  # step: 0,1,2,... (after each sched.step())
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1 -> 0
        return lr_min_ratio + (1.0 - lr_min_ratio) * cosine   # 1 -> floor

    return LambdaLR(optimizer, lr_lambda)

def load_LoRA_model(config: dict):
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    
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

def load_model_for_inference(config_path: str, checkpoint_path: str):
    """
    Loads a trained Unsloth LoRA (PEFT) model and its tokenizer from a checkpoint,
    correctly handling the resized vocabulary.

    Args:s
        config_path: Path to the original YAML config file (to get model settings).
        checkpoint_path: Path to the specific checkpoint directory 
                         (e.g., "outputs/final_model" or "outputs/checkpoint-1000").

    Returns:
        (model, tokenizer): The loaded model and tokenizer ready for inference.
    """
    print(f"\n" + "=" * 80)
    print(f"Loading model for inference from: {checkpoint_path}")
    print(f"Using config for settings from: {config_path}")
    print("=" * 80)

    # 1. Load config to get model parameters (like 4bit, max_length)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    model_config = config.get('model', {})

    load_in_4bit = training_config.get('load_in_4bit', True)
    max_seq_length = model_config.get('max_length', 512)
    
    # 2. Load the model and tokenizer from the checkpoint directory
    # Unsloth's from_pretrained is smart:
    # 1. It loads the tokenizer from checkpoint_path.
    # 2. It reads adapter_config.json to find the base_model.
    # 3. It loads the base_model (e.g., "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit").
    # 4. It sees the tokenizer vocab size is LARGER than the base model's.
    # 5. It automatically calls model.resize_token_embeddings(len(tokenizer)).
    # 6. It THEN loads the LoRA adapter weights from the checkpoint.
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path, # This is the key
        max_seq_length = max_seq_length,
        dtype = None, # Autodetect
        load_in_4bit = load_in_4bit,
    )

    print(f"\nSuccessfully loaded model from {checkpoint_path}")
    print(f"  - Tokenizer vocab size: {len(tokenizer)}")
    print(f"  - Model input embed size:  {model.get_input_embeddings().weight.shape[0]}")
    print(f"  - Model output embed size: {model.get_output_embeddings().weight.shape[0]}")

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        print("\n🚨 WARNING: Tokenizer and model embedding size mismatch!")
    else:
        print("  - Tokenizer and model embedding sizes match. ✅")
    
    return model, tokenizer