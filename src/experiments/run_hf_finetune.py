import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
import torch
import os

# Import your custom dataloader function
from src.data.unified_dataloader import get_dataloader

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

    # 2. Get the Dataset objects directly from your custom dataloader
    #    We set format to 'text' to get the natural language narratives.
    print("Initializing UnifiedEHRDataset in 'text' mode...")
    data_config['format'] = 'text' # Ensure format is set to text
    train_dataset = get_dataloader(data_config, split="train").dataset
    validation_dataset = get_dataloader(data_config, split="tuning").dataset
    test_dataset = get_dataloader(data_config, split="held_out").dataset
    
    # 3. Load Pre-trained Tokenizer
    print(f"Loading tokenizer for model: {model_config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])

    # 4. Wrap datasets with tokenisation 
    max_length = model_config.get('max_length', 512)
    train_dataset = TokenizedDatasetWrapper(train_dataset, tokenizer, max_length)
    validation_dataset = TokenizedDatasetWrapper(validation_dataset, tokenizer, max_length)
    test_dataset = TokenizedDatasetWrapper(test_dataset, tokenizer, max_length)

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
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Run Training
    print("Starting fine-tuning...")
    trainer.train()

    # 8. Run Final Evaluation on the Test Set
    print("\n--- Evaluating on the test set ---")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(test_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the experiment config YAML file.")
    args = parser.parse_args()
    main(args.config_filepath)