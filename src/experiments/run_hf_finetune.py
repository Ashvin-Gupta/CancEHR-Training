import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import torch
import os

# Import your custom dataloader function
# from src.data.unified_dataloader import get_dataloader
from src.data.unified_dataset import UnifiedEHRDataset
from torch.utils.data import Dataset

def compute_metrics(eval_pred):
    """
    Computes and returns a dictionary of metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class TokenizedDatasetWrapper(Dataset):
    """Wrapper that tokenizes text on-the-fly."""
    def __init__(self, base_dataset, tokenizer, max_length=512):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if item is None:
            return None
        
        # Tokenize the text
        tokenized = self.tokenizer(
            item["text"], 
            truncation=True, 
            max_length=self.max_length,
            # Don't pad here - let the data collator handle it
        )
        
        # Return format expected by Trainer
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": item["label"].item()  # Convert tensor to int
        }

def main(config_path: str):
    """
    Main function to run the Hugging Face fine-tuning pipeline using the UnifiedEHRDataset.
    """
    # 1. Load Configuration
    print("Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']
    training_config = config['training']

    # 2. Instantiate UnifiedEHRDataset directly
    print("Initializing UnifiedEHRDataset in 'text' mode...")
    
    # Common dataset arguments
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "cutoff_months": data_config.get("cutoff_months"),
        "format": 'text', # Ensure format is set to text
        "max_sequence_length": model_config.get('max_length', 512)
    }

    train_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    validation_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    test_dataset = UnifiedEHRDataset(split="held_out", **dataset_args)

    # # 2. Get the Dataset objects directly from your custom dataloader
    # #    We set format to 'text' to get the natural language narratives.
    # print("Initializing UnifiedEHRDataset in 'text' mode...")
    # data_config['format'] = 'text' # Ensure format is set to text
    # train_dataset = get_dataloader(data_config, split="train").dataset
    # validation_dataset = get_dataloader(data_config, split="tuning").dataset
    # test_dataset = get_dataloader(data_config, split="held_out").dataset
    
    # 3. Load Pre-trained Tokenizer
    print(f"Loading tokenizer for model: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])

    # 4. Wrap datasets with tokenisation 
    max_length = model_config.get('max_length', 512)
    train_tokenized_dataset = TokenizedDatasetWrapper(train_dataset, tokenizer, max_length)
    validation_tokenized_dataset = TokenizedDatasetWrapper(validation_dataset, tokenizer, max_length)
    test_tokenized_dataset = TokenizedDatasetWrapper(test_dataset, tokenizer, max_length)

    # 5. Load Pre-trained Model
    print(f"Loading model: {model_config['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config['model_name'], 
        num_labels=model_config['num_classes']
    )

    # 6. Set Up the Trainer from Hugging Face not custom trainer
    print("Setting up the Trainer...")
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        overwrite_output_dir=training_config['overwrite_output_dir'],
        learning_rate=float(training_config['learning_rate']),
        per_device_train_batch_size=int(training_config['batch_size']),
        per_device_eval_batch_size=int(training_config['batch_size']),
        num_train_epochs=int(training_config['epochs']),
        weight_decay=float(training_config['weight_decay']),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    #  Use DataCollatorWithPadding for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding_side="right")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Run Training
    print("Starting fine-tuning...")
    trainer.train()

    # 8. Run Final Evaluation on the Test Set
    print("\n--- Evaluating on the test set ---")
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)
    print(test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file.")
    args = parser.parse_args()
    main(args.config_filepath)