# src/training/unsloth_trainer.py
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

def run_unsloth_training(config, tokenizer, train_dataset, val_dataset):
    """Handles the full Unsloth training and inference pipeline."""
    
    model_config = config['model']
    training_config = config['training']

    # 1. Load Unsloth Model
    print("Loading Unsloth FastLanguageModel...")
    model, _ = FastLanguageModel.from_pretrained(
        model_name = model_config['model_name'],
        max_seq_length = model_config['max_length'],
        dtype = torch.float16, # Use float16 for T4/V100
        load_in_4bit = training_config.get('load_in_4bit', True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = training_config.get('lora_r', 16),
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = training_config.get('lora_alpha', 16),
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = training_config.get('gradient_checkpointing', True),
        random_state = 42,
    )
    print("  - Applied LoRA adapters (PEFT) to Unsloth model.")
    
    # 2. Set up Training Arguments
    training_args = TrainingArguments(
        output_dir=training_config['output_dir'],
        run_name=config.get('wandb', {}).get('run_name', 'unsloth-run'),
        report_to="wandb" if config.get('wandb', {}).get('enabled', False) else "none",
        **training_config
    )

    # 3. Create SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = model_config['max_length'],
        args = training_args,
        packing = True, # Use efficient packing
    )
    
    # 4. Train
    print("Starting Unsloth training...")
    trainer.train()
    
    # 5. Save Model
    print("\nSaving final Unsloth model adapters...")
    final_model_path = os.path.join(training_config['output_dir'], "final_model")
    trainer.save_model(final_model_path)
    print(f"  - Model saved to: {final_model_path}")
    
    # 6. Run Inference Test
    _run_unsloth_inference_test(model, tokenizer)

def _run_unsloth_inference_test(model, tokenizer):
    """Runs a qualitative inference test on the trained Unsloth model."""
    print("\n" + "=" * 80)
    print("Running Unsloth inference test...")
    print("=" * 80)

    model.eval()
    FastLanguageModel.for_inference(model) # Prepare Unsloth model

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