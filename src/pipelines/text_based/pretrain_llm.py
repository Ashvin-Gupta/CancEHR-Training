# src/experiments/run_llm_pretrain.py

"""
LLM Continued Pretraining Script

This script performs continued pretraining of a language model on EHR data using:
1. UnifiedEHRDataset in 'text' format for medical code translation
2. HuggingFace's ConstantLengthDataset for efficient sequence packing
3. 1-month temporal cutoff for cancer patients to avoid late-stage signals
4. Standard causal language modeling (next-token prediction)

Usage:
    python -m src.experiments.run_llm_pretrain --config_filepath configs/llm_pretrain.yaml
"""

import argparse
import yaml
import os
from datetime import datetime
import wandb
from collections import defaultdict
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TextIteratorStreamer
)
from threading import Thread
import textwrap

import torch
from huggingface_hub import login

from src.data.unified_dataset import UnifiedEHRDataset
from src.pipelines.text_based.token_adaptation import EHRTokenTranslator


def extract_text(base_dataset, tokenizer):
        """Extracts all valid text narratives and adds EOS token."""
        text_list = []
        # Use eos_token if it exists, otherwise use an empty string
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        
        print(f"  - Processing {len(base_dataset)} patients...")
        # We iterate through the base_dataset to get the text
        for i in range(len(base_dataset)):
            item = base_dataset[i]
            if item is not None:
                # item['text'] is the narrative from UnifiedEHRDataset
                text_list.append(item['text'] + ', ' + eos_token)
        print(f"  - Extracted {len(text_list)} valid narratives.")
        return text_list
        


def main(config_path: str):
    """
    Main function to run LLM continued pretraining.
    
    Args:
        config_path: Path to YAML configuration file
    """
    # 1. Load Config
    print("=" * 80)
    print("LLM Continued Pretraining")
    print("=" * 80)
    print(f"\nLoading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Set up WandB and hugging face token
    wandb_config = config.get('wandb', {})
    
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    lora_config = config['lora']

    # Build default run name from hyperparameters
    model_name = model_config['model_name'].split('/')[-1]
    default_run_name = (
        f"{model_name}-pretrain-{data_config['cutoff_months']}month-cutoff"
        f"_r{lora_config['r']}"
        f"_alpha{lora_config['lora_alpha']}"
        f"_dropout{lora_config['lora_dropout']}"
        f"_lr{training_config['learning_rate']}"
        f"_bs{training_config['batch_size']}"
        f"_wd{training_config['weight_decay']}"
        f"_ga{training_config['gradient_accumulation_steps']}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(default_run_name)

    if wandb_config.get('enabled', False):
        os.environ["WANDB_PROJECT"] = wandb_config.get("project", "ehr-llm-pretraining")
        run_name = wandb_config.get("run_name", default_run_name)
        report_to = "wandb"
        
        wandb.init(
            project=wandb_config.get("project", "ehr-llm-pretraining"), 
            config=config, # Pass entire YAML as defaults
            name=run_name
        )
    else:
        run_name = default_run_name
        report_to = "none"
    
    token_file = os.path.join("src", "resources", "API_Keys.txt")
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                line = f.readline().strip()
                hf_token = line.split('=')[1].strip('"')
                if hf_token:
                    print(f"Loaded HuggingFace token from {token_file}: {hf_token}")
        except Exception as e:
            print(f"Failed to read token from {token_file}: {e}")
    else:
        print(f"No API keys file found at {token_file}")
    
    print(f"Run name: {run_name}")
    
    login(token=str(hf_token))

    # 3. Load Model with Unsloth
    print("\n" + "=" * 80)
    print(f"Loading model: {model_config['model_name']}")
    print("=" * 80)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  - CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA devices: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        print(f"  - CUDA not available, using CPU")
    
    translator = EHRTokenTranslator(data_config["medical_lookup_filepath"], data_config["lab_lookup_filepath"])
    unique_concepts = translator.extract_translated_concepts(data_config["vocab_filepath"])
    
    # Perform token adaptation BEFORE applying LoRA
    model, tokenizer = translator.token_adaptation(
        original_model_name=model_config['base_model_name'],
        unsloth_model_name=model_config['model_name'],
        new_concepts=unique_concepts,
        max_seq_length=model_config['max_length'],  # Pass the max_length from config
        load_in_4bit=training_config.get('load_in_4bit', True)  # Pass load_in_4bit from config
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.get('r', 16),
        target_modules = lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]), # Modules for Mistral/Llama
        lora_alpha = lora_config.get('lora_alpha', 16),
        lora_dropout = lora_config.get('lora_dropout', 0.05),
        bias = lora_config.get('bias', "none"),
        use_gradient_checkpointing = training_config.get('gradient_checkpointing', 'unsloth'),
        random_state = 42,
        use_rslora = lora_config.get('use_rslora', True),
        loftq_config = lora_config.get('loftq_config', None),

    )
    print("  - Applied LoRA adapters (PEFT) to the model.")

    
    # 4. Create Base Datasets (text format)
    print("\n" + "=" * 80)
    print("Creating datasets in 'text' format...")
    print("=" * 80)
    
    dataset_args = {
        "data_dir": data_config["data_dir"],
        "vocab_file": data_config["vocab_filepath"],
        "labels_file": data_config["labels_filepath"],
        "medical_lookup_file": data_config["medical_lookup_filepath"],
        "lab_lookup_file": data_config["lab_lookup_filepath"],
        "format": 'text',  # Use existing text format!
        "cutoff_months": data_config.get("cutoff_months", 1),  # Default 1-month cutoff
        "max_sequence_length": None  # No truncation - we'll pack sequences
    }
    
    print("\nLoading training data...")
    train_base_dataset = UnifiedEHRDataset(split="train", **dataset_args)
    print(f"  - Loaded {len(train_base_dataset)} training patients")
    
    print("\nLoading validation data...")
    val_base_dataset = UnifiedEHRDataset(split="tuning", **dataset_args)
    print(f"  - Loaded {len(val_base_dataset)} validation patients")
    

    print("\n" + "=" * 80)
    print("Verifying data - First 3 patient narratives:")
    print("=" * 80)
    
    # 5. Extract text from datasets
    train_text_list = extract_text(train_base_dataset, tokenizer)
    val_text_list = extract_text(val_base_dataset, tokenizer)

    print("\nVerifying data - First 3 patient narratives:")
    # for i in range(min(3, len(train_text_list))):
    #     print(f"\n--- PATIENT {i} ---")
    #     # Print the last 1000 chars
    #     print(f"{train_text_list[i][:1000]}...")

    print("\n" + "=" * 80)
    print("Creating SFT datasets...")
    print("=" * 80)

    train_dataset = Dataset.from_dict({"text": train_text_list})
    val_dataset = Dataset.from_dict({"text": val_text_list})

    # 6. Set Up Training Arguments
    print("\n" + "=" * 80)
    print("Setting up training...")
    print("=" * 80)
    
    training_args = TrainingArguments(
        dataloader_num_workers=training_config.get('dataloader_num_workers', 8),

        output_dir=training_config['output_dir'],
        overwrite_output_dir=training_config.get('overwrite_output_dir', True),
        
        # Training hyperparameters
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        learning_rate=float(training_config['learning_rate']),
        weight_decay=float(training_config.get('weight_decay', 0.01)),
        warmup_steps=training_config.get('warmup_steps', 500),
        
        # Logging and evaluation
        logging_steps=training_config.get('logging_steps', 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=training_config.get('eval_steps', 500),
        
        # Saving
        save_strategy="steps",
        save_steps=training_config.get('save_steps', 1000),
        save_total_limit=training_config.get('save_total_limit', 2),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="loss" if val_dataset else None,
        
        # Performance
        fp16=training_config.get('fp16', False),
        bf16=training_config.get('bf16', True),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        
        # Reporting
        report_to=report_to,
        run_name=run_name,
        
        # Other
        remove_unused_columns=False,
    )
    
    print(f"  - Output directory: {training_args.output_dir}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    print(f"  - FP16: {training_args.fp16}, BF16: {training_args.bf16}")
    

    # 7. Create SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text", # Key from our TextDatasetForSFT
        max_seq_length = model_config['max_length'],
        args = training_args,
        packing = True, # --- THIS IS THE EFFICIENT PACKING! ---
    )
    
    # 10. Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # 11. Save Final Model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    
    print(f"  - Model saved to: {final_model_path}")
    
    # 12. Final Evaluation
    if val_dataset:
        print("\n" + "=" * 80)
        print("Final Evaluation")
        print("=" * 80)
        
        eval_results = trainer.evaluate()
        print("\nValidation Results:")
        for key, value in eval_results.items():
            print(f"  - {key}: {value:.4f}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    # 13. Run Inference
    print("\n" + "=" * 80)
    print("Running inference test...")
    print("=" * 80)

    # Set model to evaluation mode
    model.eval()
    
    # Call for_inference() to prepare the model
    FastLanguageModel.for_inference(model)

    # Define a sample prompt
    prompt = "Cough 2mt-4mt Drug therapy 12d-20d Ex-Smoker Cough 1d Lymphocytes: Normal Plasma proteins: High Urea serum: Low Test request : Thyroid Function Test: Normal Renal function tests: High Test request : Serum electrolytes: Normal Red blood cell count: High Cholesterol triglycerides: Normal Eosinophil count: High ALP: High Platelets: Normal Creatinine serum: Normal Bilirubin: Normal Hemoglobin: High MCV: Normal Albumin serum: High WBC: Normal MCHC - Mean corpuscular haemoglobin concentration: High Monocyte count: Normal Laboratory test observable Basophil count: Normal Neutrophils: Normal ALT: Normal Haematocrit - PCV: High Glucose 2mt-4mt Referral to surgeon Swelling 7d-12d Bp Diastolic High Blood Pressure Bp Systolic High 20d-30d"
    print(f"PROMPT: {prompt}\n")
    print("MODEL OUTPUT:")

    # Setup the streamer
    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    max_print_width = 100 # For text wrapping

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # 0. Create a list to capture all the generated text
    all_generated_chunks = []
    
    # 1. Setup generation kwargs and start the thread
    generation_kwargs = dict(
        inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 2. Print the LIVE (raw) output as it streams
    # (This will be a single, long line of space-separated tokens)
    print("MODEL OUTPUT (RAW STREAM):")
    length = 0
    for j, new_text in enumerate(text_streamer):
        print(new_text, end="")
        all_generated_chunks.append(new_text) # Capture the chunk
    
    # 3. Wait for the generation thread to finish
    thread.join()

    # 4. Now, create the FORMATTED readable output
    print("\n\n" + "=" * 80)
    print("--- Formatted Readable Output ---")
    print("=" * 80)

    # Combine all the chunks into one full string
    full_generated_text = "".join(all_generated_chunks)
    
    full_token_list = tokenizer.tokenize(full_generated_text)

    # Filter out the spacer tokens (which are ' ' for most models)
    ehr_events_only = [tok for tok in full_token_list if tok != " "]

    # Join the "real" tokens with a comma and space
    readable_output = ", ".join(ehr_events_only)

    # Use textwrap.fill to wrap the final, readable string
    print(textwrap.fill(readable_output, width=max_print_width))
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Continued Pretraining on EHR Data")
    parser.add_argument(
        "--config_filepath",
        type=str,
        required=True,
        help="Path to the experiment config YAML file"
    )
    args = parser.parse_args()
    
    main(args.config_filepath)