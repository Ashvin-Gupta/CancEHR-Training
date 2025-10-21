"""
Example usage of the 'pretrain' format for LLM continued pretraining.

This script demonstrates how to:
1. Create a dataset in pretrain format
2. Use the PretrainCollator for batching
3. Set up a DataLoader with patient shuffling
4. Train a causal language model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from src.data.unified_dataset import UnifiedEHRDataset
from src.data.pretrain_collator import PretrainCollator


def create_pretrain_dataloader(config, tokenizer, split='train'):
    """
    Creates a DataLoader for LLM continued pretraining.
    
    Args:
        config: Dictionary with data configuration
        tokenizer: Hugging Face tokenizer
        split: Dataset split ('train', 'tuning', or 'held_out')
    
    Returns:
        DataLoader configured for pretraining
    """
    # Create dataset in pretrain format
    dataset = UnifiedEHRDataset(
        data_dir=config["data_dir"],
        vocab_file=config["vocab_filepath"],
        labels_file=config["labels_filepath"],
        medical_lookup_file=config["medical_lookup_filepath"],
        lab_lookup_file=config["lab_lookup_filepath"],
        split=split,
        format='pretrain',
        max_sequence_length=config.get("max_sequence_length", 2048),
        tokenizer=tokenizer,
        cutoff_months=None  # Automatically uses 1-month cutoff for cancer patients
    )
    
    # Create collator
    collator = PretrainCollator(tokenizer)
    
    # Create DataLoader with patient shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=(split == 'train'),  # Shuffle patients for training
        collate_fn=collator,
        num_workers=config.get("num_workers", 4)
    )
    
    return dataloader


def example_usage():
    """Example of how to use the pretrain format."""
    
    # Configuration
    config = {
        "data_dir": "/data/scratch/qc25022/upgi/tokenised_data/",
        "vocab_filepath": "/data/scratch/qc25022/upgi/tokenised_data/vocab.csv",
        "labels_filepath": "/data/scratch/qc25022/upgi/master_subject_labels.csv",
        "medical_lookup_filepath": "/data/home/qc25022/cancer-extraction-pipeline/src/resources/MedicalDictTranslation.csv",
        "lab_lookup_filepath": "/data/home/qc25022/cancer-extraction-pipeline/src/resources/LabLookUP.csv",
        "max_sequence_length": 2048,
        "batch_size": 4,
        "num_workers": 4
    }
    
    # Load tokenizer
    model_name = "gpt2"  # or "meta-llama/Llama-2-7b-hf", etc.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    train_dataloader = create_pretrain_dataloader(config, tokenizer, split='train')
    val_dataloader = create_pretrain_dataloader(config, tokenizer, split='tuning')
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output/pretrain",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
    )
    
    # Note: For using HF Trainer, you'd need to convert DataLoader to Dataset
    # For now, you can use the dataset directly:
    train_dataset = train_dataloader.dataset
    val_dataset = val_dataloader.dataset
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PretrainCollator(tokenizer),
    )
    
    # Train
    trainer.train()
    
    print("Pretraining complete!")


if __name__ == "__main__":
    example_usage()

