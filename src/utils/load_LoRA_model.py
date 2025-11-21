import unsloth
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

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