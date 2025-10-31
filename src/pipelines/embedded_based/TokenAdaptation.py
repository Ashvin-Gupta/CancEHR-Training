import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unsloth
from unsloth import FastLanguageModel

original_model_name = "Qwen/Qwen3-0.6B"
unsloth_model_name = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"

new_concepts = ['Current smoker', 'Non-smoker', 'Former smoker', 'Never smoker']

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(unsloth_model_name)

# Add new tokens
num_new_tokens = tokenizer.add_tokens(new_concepts)

# Add PAD token if needed
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print(f"Added {num_new_tokens} new tokens. New vocab size = {len(tokenizer)}")

# Load the model FIRST (without modifying vocab_size)
model, _ = FastLanguageModel.from_pretrained(
    unsloth_model_name,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Now resize embeddings to match the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Initialize the new embeddings (average of sub-token embeddings)
original_tokenizer = AutoTokenizer.from_pretrained(original_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    low_cpu_mem_usage=True,
    dtype=torch.float16,
)
original_weights = base_model.get_input_embeddings().weight.data.cpu().float()

new_embeddings = model.get_input_embeddings().weight.data
with torch.no_grad():
    for concept in new_concepts:
        new_token_id = tokenizer.convert_tokens_to_ids(concept)
        sub_token_ids = original_tokenizer.encode(concept, add_special_tokens=False)
        sub_embs = original_weights[sub_token_ids]
        new_embeddings[new_token_id] = sub_embs.mean(dim=0).to(new_embeddings.device, dtype=new_embeddings.dtype)

print("All new token embeddings initialized successfully!")