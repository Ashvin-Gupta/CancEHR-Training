import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import unsloth
from unsloth import FastLanguageModel

original_model_name = "Qwen/Qwen3-0.6B"
unsloth_model_name = "unsloth/Qwen3-0.6B-Base-unsloth-bnb-4bit"

new_concepts = ['Current smoker', 'Non-smoker', 'Former smoker', 'Never smoker']

print('Phase1: Extracting token IDs of strings')
# Load original tokeniser
original_tokeniser = AutoTokenizer.from_pretrained(original_model_name)

# Create new map: {new token string -> list of sub token ids}
new_token_to_sub_token_ids = {}
for concept in new_concepts:
    sub_token_ids = original_tokeniser.encode(concept, add_special_tokens=False)
    new_token_to_sub_token_ids[concept] = sub_token_ids

# Get original embedding matrix (weights of the model)
base_model = AutoModelForCausalLM.from_pretrained(
    original_model_name,
    low_cpu_mem_usage=True,
    dtype=torch.float16,
)
original_weights = base_model.get_input_embeddings().weight.data.cpu().float()

# Clean up memory
del base_model
del original_tokeniser

print('Phase2: Creating new embedding matrix')

# Load tokensier
tokeniser = AutoTokenizer.from_pretrained(unsloth_model_name)

# Add new tokens to tokeniser
tokeniser.add_tokens(new_concepts)

# Add PAD token to tokeniser
if tokeniser.pad_token is None:
    tokeniser.add_special_tokens({'pad_token': '[PAD]'})
else:
    print(f"PAD token already exists: {tokeniser.pad_token}")

new_vocab_size = len(tokeniser)
print(f"New vocabulary size: {new_vocab_size}")

# Load model 
model, _ = FastLanguageModel.from_pretrained(
    unsloth_model_name,
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
    vocab_size=new_vocab_size,
)

print('Phase3: Overwriting random inits with averaged vectors')

new_embeddings = model.get_input_embeddings()
new_weights = new_embeddings.weight.data

with torch.no_grad():
    for concept, sub_token_ids in new_token_to_sub_token_ids.items():
        # Get ID of the new single token
        new_token_id = tokeniser.convert_tokens_to_ids(concept)
        # Get embeddings of the sub tokens
        sub_token_embeddings = original_weights[sub_token_ids]
        # Average the embeddings
        new_token_embedding = sub_token_embeddings.mean(dim=0)
        # Overwrite the random init with the averaged vector
        new_weights[new_token_id] = new_token_id.to(new_weights.device, dtype=new_weights.dtype)


print('All new token embeddings have been initialised')