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
        # Filter out None values
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # Extract input_ids from each sample
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        
        # Pad sequences to the same length
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
        
        # For causal LM: labels = input_ids, but mask out padding in loss computation
        labels = input_ids_padded.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # -100 is ignored by CrossEntropyLoss
        
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels
        }

