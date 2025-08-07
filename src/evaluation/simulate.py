import torch
import os
import yaml
from src.models.utils import load_model
from tqdm import tqdm
import pandas as pd

def sample_from_distribution(probs: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Samples an index from a 1D softmax distribution with optional temperature scaling.

    Args:
        probs (torch.Tensor): A 1D tensor of shape [vocab_size] representing probabilities.
        temperature (float): Temperature to scale the distribution. Must be > 0.

    Returns:
        int: The sampled index.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")

    logits = torch.log(probs + 1e-9) / temperature
    scaled_probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(scaled_probs, num_samples=1).item()

def simulate_subject(
        save_dir: str,
        tokens: torch.Tensor, 
        stop_tokens_indices: list, 
        experiment_dir: str, 
        simulations_per_step: int = 10,
        max_steps: int = 2048,
        device: torch.device = torch.device("cpu"),
        batch_size: int = None
    ):
    """
    Simulates a subject's data using a trained model with batched processing.

    Args:
        tokens (torch.Tensor): The tokens for the subject.
        stop_tokens_indices (list): The indices of the stop tokens.
        experiment_dir (str): The directory containing the experiment.
        simulations_per_step (int): The number of simulations to run at each token step.
        max_steps (int): Maximum steps per simulation.
        device (torch.device): The device to run the model on.
        batch_size (int): Number of simulations to batch together. If None, uses simulations_per_step.
    """
    
    # load config
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # create save directory
    os.makedirs(save_dir, exist_ok=True)

    context_length = config['data']['sequence_length']

    # load model
    model = load_model(config['model'])
    model.load_state_dict(torch.load(os.path.join(experiment_dir, "model.pth"), map_location=device))
    model.to(device)
    model.eval()

    # Move input tokens to device once
    tokens = tokens.to(device)
    stop_tokens_indices = torch.tensor(stop_tokens_indices, device=device)
    
    # Set batch size
    if batch_size is None:
        batch_size = simulations_per_step
    
    # Collect all results
    all_simulation_results = []
    
    # Process each starting position
    for start_token_index in tqdm(range(1, len(tokens)), desc="Start Token Index", leave=True):
        input_tokens = tokens[:start_token_index]
        
        # Process simulations in batches
        batch_ranges = list(range(0, simulations_per_step, batch_size))
        for batch_start in batch_ranges:
            current_batch_size = min(batch_size, simulations_per_step - batch_start)
            batch_results = simulate_batch(
                input_tokens=input_tokens,
                start_token_index=start_token_index,
                simulation_start_idx=batch_start,
                batch_size=current_batch_size,
                max_steps=max_steps,
                context_length=context_length,
                stop_tokens_indices=stop_tokens_indices,
                model=model,
                device=device
            )
            all_simulation_results.extend(batch_results)
    
        # Write results
        results_df = pd.DataFrame(all_simulation_results)
        results_df.to_csv(os.path.join(save_dir, "simulation_results.csv"), index=False)
        print(f"Saved results for start token index {start_token_index} to {os.path.join(save_dir, 'simulation_results.csv')}")


def simulate_batch(
    input_tokens: torch.Tensor,
    start_token_index: int,
    simulation_start_idx: int,
    batch_size: int,
    max_steps: int,
    context_length: int,
    stop_tokens_indices: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device
) -> list:
    """
    Simulate a batch of sequences in parallel.
    
    Returns:
        list: List of simulation results for this batch
    """
    # Pre-allocate tensors for the entire batch
    # Shape: [batch_size, input_length + max_steps]
    max_sequence_length = len(input_tokens) + max_steps
    batch_sequences = torch.zeros(batch_size, max_sequence_length, dtype=torch.long, device=device)
    
    # Initialize all sequences with input tokens
    batch_sequences[:, :len(input_tokens)] = input_tokens.unsqueeze(0).repeat(batch_size, 1)
    
    # Track current length for each sequence
    sequence_lengths = torch.full((batch_size,), len(input_tokens), dtype=torch.long, device=device)
    
    # Track which sequences are still active (not stopped)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Track stop information for each sequence
    stop_reasons = ["max_steps"] * batch_size
    stop_values = [None] * batch_size
    stop_steps = [max_steps] * batch_size
    
    with torch.no_grad():
        for step in tqdm(range(max_steps), desc="Steps", leave=False):
            if not active_mask.any():
                break  # All sequences have stopped
            
            # Get active sequences only
            active_indices = torch.where(active_mask)[0]
            if len(active_indices) == 0:
                break
            
            # Prepare input contexts for active sequences
            batch_contexts = []
            for idx in active_indices:
                seq_len = sequence_lengths[idx]
                # Get the last context_length tokens (or all if shorter)
                start_pos = max(0, seq_len - context_length)
                context = batch_sequences[idx, start_pos:seq_len]
                batch_contexts.append(context)
            
            # Pad contexts to same length for batching
            max_context_len = max(len(ctx) for ctx in batch_contexts)
            padded_contexts = torch.zeros(len(active_indices), max_context_len, dtype=torch.long, device=device)
            
            for i, ctx in enumerate(batch_contexts):
                if len(ctx) > 0:
                    padded_contexts[i, -len(ctx):] = ctx
            
            # Get predictions for all active sequences at once
            logits = model(padded_contexts)  # Shape: [active_batch_size, seq_len, vocab_size]
            
            # Get the last token predictions for each sequence
            last_token_logits = logits[:, -1, :]  # Shape: [active_batch_size, vocab_size]
            next_token_distributions = torch.softmax(last_token_logits, dim=-1)
            
            # Sample next tokens for all active sequences
            next_tokens = torch.multinomial(next_token_distributions, num_samples=1).squeeze(-1)
            
            # Update sequences and check for stop conditions
            for i, global_idx in enumerate(active_indices):
                next_token = next_tokens[i].item()
                seq_len = sequence_lengths[global_idx]
                
                # Add the next token
                batch_sequences[global_idx, seq_len] = next_token
                sequence_lengths[global_idx] += 1
                
                # Check if this token is a stop token
                if next_token in stop_tokens_indices:
                    active_mask[global_idx] = False
                    stop_reasons[global_idx] = "stop_token"
                    stop_values[global_idx] = next_token
                    stop_steps[global_idx] = step
    
    # Extract results for this batch
    batch_results = []
    for i in range(batch_size):
        # Get the predicted tokens (everything after the input)
        predicted_tokens = batch_sequences[i, len(input_tokens):sequence_lengths[i]].cpu().tolist()
        
        result = {
            "start_token_index": start_token_index,
            "simulation_index": simulation_start_idx + i,
            "stop_reason": stop_reasons[i],
            "stop_value": stop_values[i],
            "stop_step": stop_steps[i],
            "predicted_tokens": predicted_tokens,
        }
        batch_results.append(result)
    
    return batch_results

if __name__ == "__main__":

    import argparse
    from src.data.dataset import NightingaleEvaluationDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of simulations to batch together")
    parser.add_argument("--simulations_per_step", type=int, default=10, help="Total number of simulations per starting position")
    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Simulations per step: {args.simulations_per_step}")

    # calculate stop tokens
    stop_tokens_str = [
        "MEDS_DEATH",
        "TRANSFER_TO//discharge//UNKNOWN",
        "HOSPITAL_DISCHARGE//HOME",
        "HOSPITAL_DISCHARGE//UNK",
        # "HOSPITAL_DISCHARGE//SKILLED",
        # "TRANSFER_TO//admit//Discharge",
        # "TRANSFER_TO//ED//Emergency",
    ]

    # load experiment directory
    experiment_dir = f"src/experiments/results/{args.experiment_name}"
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory {experiment_dir} does not exist")
        exit()

    # Create eval dataset
    data_dir = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/ethos_timetokens/"
    dataset_dir = os.path.join(data_dir, "tuning")
    vocab_path = os.path.join(data_dir, "vocab.csv")
    eval_dataset = NightingaleEvaluationDataset(dataset_dir, vocab_path)

    # Get subject data
    try:    
        subject_data = eval_dataset.get_data_by_subject_id(args.subject_id)
    except KeyError:
        print(f"Subject {args.subject_id} not found in evaluation dataset")
        exit()

    # get the indices of the stop tokens
    stop_tokens_indices = [eval_dataset.string_to_token(stop_token_str) for stop_token_str in stop_tokens_str]

    # Simulate subject
    simulate_subject(
        save_dir=os.path.join(experiment_dir, f"simulations/{args.subject_id}"),
        tokens=subject_data['tokens'],
        stop_tokens_indices=stop_tokens_indices,
        experiment_dir=experiment_dir,
        simulations_per_step=args.simulations_per_step,
        device=device,
        batch_size=args.batch_size
    )