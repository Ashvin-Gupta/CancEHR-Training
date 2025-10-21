import torch


class PretrainCollator:
    """
    Collator for causal language modeling with EHR narratives.
    Handles padding and creates labels (labels = input_ids for next-token prediction).
    """
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # All sequences are exactly max_sequence_length, so just stack
        input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
        
        # No padding needed, so all tokens are valid
        attention_mask = torch.ones_like(input_ids)
        
        # Labels = input_ids for causal LM
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

