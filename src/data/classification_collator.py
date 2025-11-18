# src/data/classification_collator.py

"""
Data collator for LLM-based classification tasks.

Handles tokenization, padding, attention masks, and binary label conversion
for EHR text classification.
"""

import torch
from typing import List, Dict, Any


class ClassificationCollator:
    """
    Collate function for LLM classification tasks.
    
    Takes raw text from UnifiedEHRDataset and tokenizes it for batch processing.
    Also converts multi-class labels to binary (cancer vs control).
    
    Args:
        tokenizer: HuggingFace tokenizer (should be the extended tokenizer from pretraining)
        max_length: Maximum sequence length for truncation
        binary_classification: If True, converts labels > 0 to 1 (cancer vs control)
    """
    
    def __init__(self, tokenizer, max_length: int = 2048, binary_classification: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.binary_classification = binary_classification
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of dicts with keys 'text' (str) and 'label' (torch.Tensor)
        
        Returns:
            Dict with:
                - input_ids: (batch_size, seq_len) tokenized text
                - attention_mask: (batch_size, seq_len) mask for padding
                - labels: (batch_size,) classification labels
        """
        # Filter out None values (patients without labels)
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        
        # Extract text and labels
        texts = [item['text'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        
        # Convert to binary labels if needed
        if self.binary_classification:
            # Label 0 = Control, Label > 0 = Cancer
            labels = (labels > 0).long()
        
        # Tokenize the text
        # This handles padding and creates attention masks automatically
        encoded = self.tokenizer(
            texts,
            padding=True,  # Pad to longest sequence in batch
            truncation=True,  # Truncate to max_length
            max_length=self.max_length,
            return_tensors='pt',  # Return PyTorch tensors
            return_attention_mask=True  # Return attention masks
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels
        }


