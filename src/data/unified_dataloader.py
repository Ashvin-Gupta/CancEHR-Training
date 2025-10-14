import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .unified_dataset import UnifiedEHRDataset 

def pad_collate(batch):
    """
    A collate function to handle padding for token sequences and batching for text data.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    if 'text' in batch[0]:
        # For text data, the LLM's tokenizer will handle padding later.
        # We just batch the text and labels.
        return {
            'text': [item['text'] for item in batch],
            'labels': torch.stack([item['label'] for item in batch])
        }
    else:
        # For token data, we pad the sequences to the same length in the batch.
        tokens = [item['tokens'] for item in batch]
        labels = [item['label'] for item in batch]
        lengths = [len(s) for s in tokens]
        
        padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
        
        return {
            'tokens': padded_tokens,
            'lengths': torch.tensor(lengths),
            'labels': torch.stack(labels)
        }

def get_dataloader(config: dict, split: str) -> DataLoader:
    """
    Creates a dataloader using the UnifiedEHRDataset.
    
    Args:
        config (dict): The data configuration dictionary.
        split (str): The dataset split to load ('train', 'tuning', or 'held_out').

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    print(f"Creating {split} dataloader...")
    
    dataset = UnifiedEHRDataset(
        data_dir=config["data_dir"],
        vocab_file=config["vocab_filepath"],
        labels_file=config["labels_filepath"],
        medical_lookup_file=config["medical_lookup_filepath"],
        lab_lookup_file=config["lab_lookup_filepath"],
        cutoff_months=config.get("cutoff_months"),
        format=config.get("format", "tokens"),
        split=split
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=(split == 'train'), 
        num_workers=config.get("num_workers", 4),
        collate_fn=pad_collate # <-- Use our custom collate function
    )
    return dataloader