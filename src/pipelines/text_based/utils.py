import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import os
import yaml
import textwrap
from threading import Thread
from transformers import TextIteratorStreamer
import argparse



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