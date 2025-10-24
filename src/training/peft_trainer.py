# src/training/peft_trainer.py
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

def _tokenize_and_chunk(examples, tokenizer, max_length):
    """Helper function to tokenize and chunk the dataset."""
    # Tokenize all text
    tokenized_inputs = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
    
    concatenated_ids = []
    for ids in tokenized_inputs['input_ids']:
        concatenated_ids.extend(ids)
        
    total_length = len(concatenated_ids)
    # We drop the small remainder, just like `packing=True`
    total_length = (total_length // max_length) * max_length
    
    # Split into chunks
    result = [concatenated_ids[i : i + max_length] for i in range(0, total_length, max_length)]
    return {"input_ids": result}

def run_peft_training(config, tokenizer, train_dataset, val_dataset):
    """Handles the full standard PEFT training and inference pipeline."""
    
    model_config = config['model']
    training_config = config['training']

    # 1. Tokenize and Chunk Data (Required for standard Trainer)
    print("Tokenizing and chunking dataset for standard Trainer...")
    max_length = model_config['max_length']
    
    tokenized_train_dataset = train_dataset.map(
        _tokenize_and_chunk,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=["text"]
    )
    tokenized_val_dataset = val_dataset.map(
        _tokenize_and_chunk,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=["text"]
    )
    print(f"  - New tokenized training dataset: {tokenized_train_dataset}")
    print(f"  - New tokenized validation dataset: {tokenized_val_dataset}")

    # 2. Load Model with 4-bit Config
    print("Loading model with BitsAndBytesConfig...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_config.get('load_in_4bit', True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for T4/V100
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config['model_name'],
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # 3. Apply PEFT/LoRA
    print("Applying LoRA adapters (PEFT)...")
    lora_config = LoraConfig(
        r=training_config.get('lora_r', 16),
        lora_alpha=training_config.get('lora_alpha', 16),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 4. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        run_name=config.get('wandb', {}).get('run_name', 'peft-run'),
        report_to="wandb" if config.get('wandb', {}).get('enabled', False) else "none",
        **training_config
    )

    # 5. Create Trainer
    print("\nInitializing standard Trainer...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 6. Train
    print("Starting standard PEFT training...")
    trainer.train()
    
    # 7. Save Model
    print("\nSaving final PEFT model adapters...")
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    print(f"  - Model saved to: {final_model_path}")

    # 8. Run Inference Test
    _run_peft_inference_test(model, tokenizer)

def _run_peft_inference_test(model, tokenizer):
    """Runs a qualitative inference test on the trained PEFT model."""
    print("\n" + "=" * 80)
    print("Running standard PEFT inference test...")
    print("=" * 80)

    model.eval() # Standard eval mode

    prompt = "Patient presents with cough and fever. Past medical history includes"
    print(f"PROMPT: {prompt}\n")
    print("MODEL OUTPUT:")

    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    max_print_width = 100
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    generation_kwargs = dict(inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True)
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    length = 0
    for j, new_text in enumerate(text_streamer):
        # (Text wrapping logic)
        if j == 0:
            wrapped_text = textwrap.wrap(new_text, width=max_print_width)
            length = len(wrapped_text[-1]) if wrapped_text else 0
            wrapped_text = "\n".join(wrapped_text)
            print(wrapped_text, end="")
        else:
            length += len(new_text)
            if length >= max_print_width:
                wrap_point = new_text.rfind(' ', 0, max_print_width - (length - len(new_text)))
                if wrap_point != -1:
                    print(new_text[:wrap_point] + "\n" + new_text[wrap_point+1:], end="")
                    length = len(new_text[wrap_point+1:])
                else:
                    print("\n" + new_text, end="")
                    length = len(new_text)
            else:
                print(new_text, end="")
    print("\n")