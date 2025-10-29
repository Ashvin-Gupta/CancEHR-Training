# src/data/embedded_dataset.py
import torch
import os
from torch.utils.data import Dataset

class PreEmbeddedDataset(Dataset):
    def __init__(self, data_dir, split='train', task='classification'):
        """
        Load pre-embedded data for a specific split.
        
        Args:
            data_dir: Base directory containing train/, tuning/, held_out/ subdirectories
            split: Which split to load ('train', 'tuning', 'held_out')
            task: Task type - 'classification' (encoder), 'autoregressive' (decoder), or 'both'
        """
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory {self.split_dir} does not exist")
        
        self.task = task
        self.data_files = [os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) if f.endswith('.pt')]
        self.data_files.sort()  # Ensure consistent ordering
    
    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        # Load the pre-computed file
        data = torch.load(self.data_files[idx])
        
        if self.task == 'classification':
            # For encoder models: only need embeddings and label
            return {
                "embeddings": data['embeddings'],  # (N, 768)
                "label": data['label']
            }
        elif self.task == 'autoregressive':
            # For decoder models: need embeddings, token_ids, and label
            return {
                "embeddings": data['embeddings'],  # (N, 768)
                "token_ids": data['token_ids'],    # (N,)
                "label": data['label']
            }
        else:  # 'both'
            # Return everything
            return data