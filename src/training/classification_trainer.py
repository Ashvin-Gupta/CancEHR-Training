# src/training/classification_trainer.py

"""
Classification trainer for LLM-based binary classification.

Provides a wrapper model that adds a classification head on top of a frozen LLM,
and functions to train only the classification head.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from typing import Dict, Any


class LLMClassifier(nn.Module):
    """
    Wrapper that adds a classification head on top of a (frozen) LLM.
    
    The LLM extracts hidden states, and we use the last non-padding token's
    hidden state as input to a simple linear classification head.
    
    Args:
        base_model: The pretrained LLM (with or without LoRA adapters)
        hidden_size: Hidden dimension of the LLM
        num_labels: Number of output classes (2 for binary classification)
        freeze_base: Whether to freeze the base LLM parameters
    """
    
    def __init__(self, base_model, hidden_size: int, num_labels: int = 2, freeze_base: bool = True):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            print("  - Froze all base model parameters")
        
        # Classification head: hidden_size -> num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)
        print(f"  - Added classification head: {hidden_size} -> {num_labels}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for classification.
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            attention_mask: (batch_size, seq_len) attention mask
            labels: (batch_size,) ground truth labels
        
        Returns:
            Dict with:
                - loss: scalar loss (if labels provided)
                - logits: (batch_size, num_labels) classification logits
                - hidden_states: (batch_size, hidden_size) extracted features
        """
        # Get hidden states from the LLM
        # Note: Most LLMs return a tuple or a special object
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Extract the last layer's hidden states
        # Shape: (batch_size, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]
        
        # Get the last non-padding token's hidden state for each sequence
        # The attention_mask is 1 for real tokens, 0 for padding
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_size = hidden_states.size(0)
        
        # Gather the hidden states at the last valid position
        # Shape: (batch_size, hidden_size)
        last_hidden_states = hidden_states[range(batch_size), sequence_lengths]
        
        # Pass through classification head
        # Shape: (batch_size, num_labels)
        logits = self.classifier(last_hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': last_hidden_states
        }
    
    def print_trainable_parameters(self):
        """Print the number of trainable vs total parameters."""
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"\n{'='*80}")
        print("Model Parameters:")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - Total params: {all_params:,}")
        print(f"  - Trainable %: {100 * trainable_params / all_params:.2f}%")
        print(f"{'='*80}\n")


def compute_metrics(eval_pred):
    """
    Compute classification metrics for evaluation.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
    
    Returns:
        Dict of metric names to values
    """
    predictions, labels = eval_pred
    
    # Handle case where predictions is a tuple/dict (model returns multiple outputs)
    # Extract just the logits (first element if tuple, or 'logits' key if dict)
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Usually logits are first
    elif isinstance(predictions, dict):
        predictions = predictions['logits']
    
    # Now predictions should be shape (batch_size, num_labels)
    # Get predicted class (argmax of logits)
    preds = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # Calculate AUROC using softmax probabilities
    probs = torch.softmax(torch.tensor(predictions), dim=1).numpy()[:, 1]
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        # If only one class present in labels
        auroc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc
    }


def run_classification_training(
    config: Dict[str, Any],
    model: nn.Module,
    tokenizer,
    train_dataset,
    val_dataset,
    collate_fn
):
    """
    Run the classification training loop.
    
    Args:
        config: Configuration dict with model, training, and data settings
        model: LLMClassifier model
        tokenizer: Tokenizer (for saving with model)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        collate_fn: Data collator function
    """
    training_config = config['training']
    model_config = config['model']
    wandb_config = config.get('wandb', {})
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        run_name=wandb_config.get('run_name', 'llm-classifier'),
        report_to="wandb" if wandb_config.get('enabled', False) else "none",
        
        # Training hyperparameters
        num_train_epochs=int(training_config.get('epochs', 10)),
        per_device_train_batch_size=int(training_config.get('batch_size', 8)),
        per_device_eval_batch_size=int(training_config.get('eval_batch_size', 8)),
        learning_rate=float(training_config.get('learning_rate', 1e-3)),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=int(training_config.get('warmup_steps', 100)),
        
        # Gradient settings
        gradient_accumulation_steps=int(training_config.get('gradient_accumulation_steps', 1)),
        
        # Precision
        fp16=bool(training_config.get('fp16', False)),
        bf16=bool(training_config.get('bf16', True)),
        
        # Logging and evaluation
        logging_steps=int(training_config.get('logging_steps', 10)),
        eval_strategy="steps",
        eval_steps=int(training_config.get('eval_steps', 100)),
        save_strategy="steps",
        save_steps=int(training_config.get('save_steps', 500)),
        save_total_limit=int(training_config.get('save_total_limit', 2)),
        
        # Best model tracking
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        
        # Other
        remove_unused_columns=False,  # Keep our custom columns
    )
    
    # Create trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting classification training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    print(f"  - Model saved to: {final_model_path}")
    
    # Run final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    print("\nFinal Evaluation Results:")
    print("="*80)
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    print("="*80)
    
    return trainer, eval_results


