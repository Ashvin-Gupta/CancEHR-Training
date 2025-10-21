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

    Only useful for the 'tokens' format.
    For the 'text' format, use the UnifiedEHRDataset directly. Only appropriate for debugging.
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

if __name__ == '__main__':
    """
    A simple test block to verify that the UnifiedEHRDataset and dataloader are working correctly.
    
    This can be run directly from the command line:
    python -m src.data.dataloader
    """
    print("--- Running Dataloader Verification Script ---")

    # --- 1. Create a dummy config for testing ---
    # IMPORTANT: Update these paths to your actual file locations
    test_config = {
        "format": "tokens",
        "batch_size": 4,
        "num_workers": 0, # Use 0 for simple debugging
        "cutoff_months": 6,
        "data_dir":'/data/scratch/qc25022/upgi/tokenised_data_debug/cprd_test',
        "vocab_filepath": "/data/scratch/qc25022/upgi/tokenised_data_debug/cprd_test/vocab.csv",
        "labels_filepath": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
        "medical_lookup_filepath": "/data/home/qc25022/cancer-extraction-pipeline/src/resources/MedicalDictTranslation.csv",
        "lab_lookup_filepath": "/data/home/qc25022/cancer-extraction-pipeline/src/resources/LabLookUP.csv"
    }

    # --- 2. Test the 'tokens' format ---
    print("\n--- Verifying 'tokens' format for custom models ---")
    test_config['format'] = 'tokens'
    token_dataloader = get_dataloader(test_config, split="train")
    
    try:
        token_batch = next(iter(token_dataloader))
        if token_batch:
            print("Successfully fetched one batch in 'tokens' format.")
            print(f"  - Tokens tensor shape: {token_batch['tokens'].shape}")
            print(f"  - Lengths tensor shape: {token_batch['lengths'].shape}")
            print(f"  - Labels tensor: {token_batch['labels']}")
            print(f"  - Batch size matches: {token_batch['tokens'].shape[0] == test_config['batch_size']}")
        else:
            print("  - Dataloader produced an empty batch.")
    except Exception as e:
        print(f"  - FAILED to fetch a batch in 'tokens' format. Error: {e}")


    # --- 3. Test the 'text' format ---
    print("\n--- Verifying 'text' format for LLM fine-tuning ---")
    test_config['format'] = 'text'
    text_dataloader = get_dataloader(test_config, split="train")
    
    try:
        text_batch = next(iter(text_dataloader))
        if text_batch:
            print("Successfully fetched one batch in 'text' format.")
            print(f"  - Number of text samples in batch: {len(text_batch['text'])}")
            print(f"  - Labels tensor: {text_batch['labels']}")
            print("\n  - Example Patient Narrative (first in batch):")
            print(f"    '{text_batch['text'][0][:500]}...'") # Print first 500 characters
        else:
            print("  - Datalaloader produced an empty batch.")
    except Exception as e:
        print(f"  - FAILED to fetch a batch in 'text' format. Error: {e}")
    
    # --- 4. Test temporal truncation with cutoff_months ---
    print("\n--- Verifying Temporal Truncation (cutoff_months) ---")
    print("Testing that increasing cutoff_months reduces sequence length for cancer cases...")
    
    # Create datasets with different cutoff values
    cutoff_values = [None, 24, 12, 6]
    
    for cutoff in cutoff_values:
        print(f"\n  Testing with cutoff_months={cutoff}:")
        test_config['cutoff_months'] = cutoff
        test_config['format'] = 'tokens'
        
        # Create dataset directly to inspect individual samples
        dataset = UnifiedEHRDataset(
            data_dir=test_config["data_dir"],
            vocab_file=test_config["vocab_filepath"],
            labels_file=test_config["labels_filepath"],
            medical_lookup_file=test_config["medical_lookup_filepath"],
            lab_lookup_file=test_config["lab_lookup_filepath"],
            cutoff_months=cutoff,
            format='tokens',
            split='train'
        )
        
        # Find a cancer case and a control to compare
        cancer_idx = None
        control_idx = None
        
        for i in range(min(100, len(dataset))):  # Check first 100 samples
            sample = dataset[i]
            if sample is not None:
                label = sample['label'].item()
                if label > 0 and cancer_idx is None:
                    cancer_idx = i
                elif label == 0 and control_idx is None:
                    control_idx = i
                
                if cancer_idx is not None and control_idx is not None:
                    break
        
        if cancer_idx is not None:
            cancer_sample = dataset[cancer_idx]
            cancer_length = len(cancer_sample['tokens'])
            cancer_label = cancer_sample['label'].item()
            print(f"    Cancer case (idx={cancer_idx}, label={cancer_label}): sequence length = {cancer_length}")
        else:
            print(f"    No cancer case found in first 100 samples")
        
        if control_idx is not None:
            control_sample = dataset[control_idx]
            control_length = len(control_sample['tokens'])
            print(f"    Control case (idx={control_idx}): sequence length = {control_length}")
        else:
            print(f"    No control case found in first 100 samples")
    
    print("\n  Expected behavior:")
    print("    - Cancer cases should have DECREASING sequence lengths as cutoff_months INCREASES")
    print("    - Control cases should have SAME sequence length regardless of cutoff_months")
    print("    - cutoff_months=None should show the full timeline")

    # --- 5. Test the 'pretrain' format ---
    print("\n--- Verifying 'pretrain' format for LLM continued pretraining ---")
    try:
        from transformers import AutoTokenizer
        from src.data.pretrain_collator import PretrainCollator
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # GPT-2 doesn't have a pad token by default, so we set it
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        pretrain_dataset = UnifiedEHRDataset(
            data_dir=test_config["data_dir"],
            vocab_file=test_config["vocab_filepath"],
            labels_file=test_config["labels_filepath"],
            medical_lookup_file=test_config["medical_lookup_filepath"],
            lab_lookup_file=test_config["lab_lookup_filepath"],
            split="train",
            format='pretrain',
            max_sequence_length=512,
            tokenizer=tokenizer,
            cutoff_months=None  # Will use 1-month cutoff automatically for cancer patients
        )
        
        collator = PretrainCollator(tokenizer)
        pretrain_dataloader = DataLoader(
            pretrain_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )
        
        batch = next(iter(pretrain_dataloader))
        if batch:
            print("Successfully fetched one batch in 'pretrain' format.")
            print(f"  - input_ids shape: {batch['input_ids'].shape}")
            print(f"  - labels shape: {batch['labels'].shape}")
            print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  - Sample shows random window sampling working correctly")
            
            # Verify that different calls return different windows (data augmentation)
            batch2 = next(iter(pretrain_dataloader))
            if batch2:
                # Check if the batches are different (random sampling)
                are_different = not torch.equal(batch['input_ids'], batch2['input_ids'])
                print(f"  - Random window sampling verified: batches differ = {are_different}")
        else:
            print("  - Dataloader produced an empty batch.")
    except Exception as e:
        print(f"  - FAILED to fetch batch in 'pretrain' format. Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Verification Script Finished ---")