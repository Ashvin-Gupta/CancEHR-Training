# In a new file, e.g., src/data/embedded_dataset.py
import torch
import os
from torch.utils.data import Dataset

class PreEmbeddedDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Load pre-embedded data for a specific split.
        
        Args:
            data_dir: Base directory containing train/, tuning/, held_out/ subdirectories
            split: Which split to load ('train', 'tuning', 'held_out')
        """
        self.split_dir = os.path.join(data_dir, split)
        if not os.path.exists(self.split_dir):
            raise ValueError(f"Split directory {self.split_dir} does not exist")
            
        self.data_files = [os.path.join(self.split_dir, f) for f in os.listdir(self.split_dir) if f.endswith('.pt')]
        self.data_files.sort()  # Ensure consistent ordering
    
    def __len__(self):
        return len(self.data_files)
        
    def __getitem__(self, idx):
        # Just load the pre-computed file
        data = torch.load(self.data_files[idx])
        return {
            "embeddings": data['embeddings'],
            "label": data['label']
        }