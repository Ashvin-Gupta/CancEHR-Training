# src/data/classification_collator.py

"""
Data collator for LLM-based classification tasks.

Handles tokenization, padding, attention masks, and binary label conversion
for EHR text classification.
"""

import torch
from typing import List, Dict, Any
import warnings


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
    
    def __init__(self, tokenizer, max_length: int = 2048, binary_classification: bool = True, truncation: bool = False,handle_long_sequences: str = 'warn'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.binary_classification = binary_classification
        self.truncation = truncation
        self.handle_long_sequences = handle_long_sequences
        self._warned_once = False  # Only warn once to avoid spam
        self._long_sequence_count = 0
        self._total_sequences = 0
    
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
        
        # Pre-check sequence lengths if we have a max_length and we're not truncating
        if self.max_length is not None and not self.truncation:
            filtered_texts = []
            filtered_labels = []
            
            for text, label in zip(texts, labels):
                # Quick tokenization to check length
                token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
                self._total_sequences += 1
                
                if token_count > self.max_length:
                    self._long_sequence_count += 1
                    
                    if self.handle_long_sequences == 'error':
                        raise ValueError(
                            f"Sequence length {token_count} exceeds max_length {self.max_length}. "
                            f"Set truncation=True or increase max_length in model loading."
                        )
                    elif self.handle_long_sequences == 'warn':
                        if not self._warned_once:
                            warnings.warn(
                                f"Found sequence with length {token_count} exceeding max_length {self.max_length}. "
                                f"Truncating from the start (keeping most recent events). "
                                f"This warning will only show once."
                            )
                            self._warned_once = True
                        # Truncate from the start to keep the end (most recent medical events)
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        # Keep the last (max_length - 2) tokens, leaving room for special tokens
                        truncated_tokens = tokens[-(self.max_length - 2):]
                        text = self.tokenizer.decode(truncated_tokens)
                        filtered_texts.append(text)
                        filtered_labels.append(label)
                    elif self.handle_long_sequences == 'skip':
                        if not self._warned_once:
                            warnings.warn(
                                f"Skipping sequences longer than {self.max_length}. "
                                f"This warning will only show once."
                            )
                            self._warned_once = True
                        continue  # Skip this sample
                    elif self.handle_long_sequences == 'truncate':
                        # Truncate from the start to keep the end
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)
                        truncated_tokens = tokens[-(self.max_length - 2):]
                        text = self.tokenizer.decode(truncated_tokens)
                        filtered_texts.append(text)
                        filtered_labels.append(label)
                else:
                    filtered_texts.append(text)
                    filtered_labels.append(label)
            
            texts = filtered_texts
            if len(filtered_labels) > 0:
                labels = torch.stack(filtered_labels)
            else:
                return None  # All sequences were too long and skipped
        
        # Tokenize the text with dynamic padding
        tokenizer_kwargs = {
            'padding': True,  # Pad to longest sequence in batch
            'truncation': self.truncation,
            'return_tensors': 'pt',
            'return_attention_mask': True
        }
        
        # Only add max_length if truncation is enabled
        if self.truncation and self.max_length is not None:
            tokenizer_kwargs['max_length'] = self.max_length
        
        encoded = self.tokenizer(texts, **tokenizer_kwargs)
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'labels': labels
        }
    
    def get_stats(self):
        """Return statistics about long sequences encountered."""
        return {
            'total_sequences': self._total_sequences,
            'long_sequences': self._long_sequence_count,
            'percentage_long': (self._long_sequence_count / self._total_sequences * 100) 
                               if self._total_sequences > 0 else 0
        }


