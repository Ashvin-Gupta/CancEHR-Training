import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

model_name = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"

new_concepts = ['Current smoker', 'Non-smoker', 'Former smoker', 'Never smoker']

print('Phase1: Extracting token IDs of strings')
# Load original tokeniser
original_tokeniser = AutoTokenizer.from_pretrained(model_name)

# Create new map: {new token string -> list of sub token ids}
new_token_to_sub_token_ids = {}
for concept in new_concepts:
    sub_token_ids = original_tokeniser.encode(concept, add_special_tokens=False)
    new_token_to_sub_token_ids[concept] = sub_token_ids

# Get original embedding matrix (weights of the model)
base_model = FastLanguageModel.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    dtype=torch.float16,
)
original_weights = base_model.get_input_embeddings().weight.data.cpu().float()

# Clean up memory
del base_model
del original_tokeniser

print('Phase2: Creating new embedding matrix')

