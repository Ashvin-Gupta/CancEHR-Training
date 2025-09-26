from src.data.evaluation_datasets import RolloutEvaluationDataset
import torch
from tqdm import tqdm
from src.models.utils import load_model, sample_from_distribution
import os
import yaml
import pandas as pd
import json
from datetime import datetime

# Enable TensorFloat32 for better performance on supported GPUs
torch.set_float32_matmul_precision('high')

def rollout_benchmark(model: torch.nn.Module, dataset: RolloutEvaluationDataset, max_steps: int, num_rollouts: int = 1, num_subjects_per_batch: int = 1, temperature: float = 1.0, device: torch.device = torch.device("cpu"), save_dir: str = None):
    """
    Runs a rollout benchmark on a model and saves results in a csv file in the save_dir.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (RolloutEvaluationDataset): The dataset to evaluate on.
        max_steps (int): The maximum number of steps to take for each rollout.
        num_rollouts (int): The number of rollouts to run for each subject.
        num_subjects_per_batch (int): The number of subjects to process in each batch.
        temperature (float): The temperature to use for sampling.
        device (torch.device): The device to use for processing.
        save_dir (str): The directory to save the results to.

    Returns:
        pd.DataFrame: A dataframe containing the results of the benchmark.
    """
    model.to(device)
    model.eval()
    
    model = torch.compile(model)

    end_token_ids = dataset.end_token_ids
    end_token_tensor = torch.tensor(end_token_ids, device=device)
    
    # Create empty list to store results in long format
    os.makedirs(save_dir, exist_ok=True)
    all_results = []
    
    # Convert dataset to list for batching
    dataset_list = list(dataset)
    total_subjects = len(dataset_list)
    
    # Initialize results_df
    results_df = pd.DataFrame()
    
    # Process subjects in batches
    for batch_start in tqdm(range(0, total_subjects, num_subjects_per_batch), desc="Processing subject batches"):
        batch_end = min(batch_start + num_subjects_per_batch, total_subjects)
        batch_subjects = dataset_list[batch_start:batch_end]
        current_batch_size = len(batch_subjects)
        
        # Total rollouts in this batch
        total_batch_rollouts = current_batch_size * num_rollouts
        
        # Prepare batch data
        batch_input_tokens = []
        batch_metadata = []
        
        for subject_idx, x in enumerate(batch_subjects):
            input_tokens = x['input_tokens']
            # Replicate for num_rollouts and add to batch
            for rollout_idx in range(num_rollouts):
                batch_input_tokens.append(input_tokens)
                batch_metadata.append({
                    'subject_idx': subject_idx,
                    'rollout_idx': rollout_idx,
                    'subject_data': x
                })
        
        # Stack all tokens into single batch tensor [total_batch_rollouts, seq_len]
        current_tokens = torch.stack(batch_input_tokens).to(device)
        
        # Track active rollouts (not yet stopped)
        active_mask = torch.ones(total_batch_rollouts, dtype=torch.bool, device=device)
        
        # Track outcomes for each rollout
        stop_reasons = ["max_steps"] * total_batch_rollouts
        predicted_tokens = [None] * total_batch_rollouts
        steps_taken = [max_steps] * total_batch_rollouts
        
        with torch.no_grad():
            for step in tqdm(range(max_steps), desc=f"Batch {batch_start//num_subjects_per_batch + 1}", leave=False):
                if not active_mask.any():
                    break  # All rollouts have stopped
                
                # Get predictions for active rollouts only
                active_indices = torch.where(active_mask)[0]
                if len(active_indices) == 0:
                    break
                    
                active_tokens = current_tokens[active_indices]
                output = model({'ehr': {'input_token_ids': active_tokens}})
                
                # Sample next tokens with temperature
                logits = output[:, -1]  # Shape: [active_batch_size, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                next_tokens = sample_from_distribution(probs, temperature=temperature)
                
                # Check for end tokens (vectorized)
                is_end_token = torch.isin(next_tokens, end_token_tensor)
                
                # Update active mask and track results
                for i, global_idx in enumerate(active_indices):
                    token = next_tokens[i].item()
                    
                    if is_end_token[i]:
                        active_mask[global_idx] = False
                        stop_reasons[global_idx] = "end_token"
                        predicted_tokens[global_idx] = token
                        steps_taken[global_idx] = step + 1
                    
                    # Update the input sequence for next iteration
                    current_tokens[global_idx] = torch.cat([current_tokens[global_idx, 1:], torch.tensor([token], device=device)])

        # Aggregate results by subject and convert to long format (one row per subject outcome pair)
        for subject_idx in range(current_batch_size):
            x = batch_subjects[subject_idx]
            
            # Find rollouts for this subject
            subject_rollout_indices = [i for i, meta in enumerate(batch_metadata) if meta['subject_idx'] == subject_idx]
            
            # Aggregate results across rollouts for this subject
            outcome_counts = {}
            subject_steps_taken = []
            
            for rollout_global_idx in subject_rollout_indices:
                subject_steps_taken.append(steps_taken[rollout_global_idx])
                
                if stop_reasons[rollout_global_idx] == "end_token":
                    token_str = dataset.vocab[dataset.vocab["token"] == predicted_tokens[rollout_global_idx]]["str"].values[0]
                    outcome_counts[token_str] = outcome_counts.get(token_str, 0) + 1
                else:
                    outcome_counts["max_steps"] = outcome_counts.get("max_steps", 0) + 1
            
            # Calculate average steps taken
            avg_steps_taken = sum(subject_steps_taken) / num_rollouts
            
            # Get subject info
            subject_id = x["subject_id"].item()
            
            # Handle cases where no end token was found (end_token = -1)
            end_token_value = int(x['end_token'].item())
            if end_token_value == -1:
                real_end_token_str = "NO_END_TOKEN_FOUND"
                real_end_token_steps = -1  # Indicates no end token found
            else:
                real_end_token_str = dataset.vocab[dataset.vocab["token"] == end_token_value]["str"].values[0]
                real_end_token_steps = (x['end_token_idx'] - x['start_token_idx']).item()
            
            # Create one row per outcome (long format)
            for outcome, count in outcome_counts.items():
                proportion = count / num_rollouts
                
                result_row = {
                    "subject_id": subject_id,
                    "real_end_token_str": real_end_token_str,
                    "real_end_token_steps": real_end_token_steps,
                    "outcome": outcome,
                    "count": count,
                    "proportion": proportion,
                    "avg_steps_taken": avg_steps_taken
                }
                all_results.append(result_row)
    
        # Save dataframe after each batch (incremental saving)
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(save_dir, "rollout_results.csv")
        results_df.to_csv(results_path, index=False)
    
    print(f"Saved {len(results_df)} outcome records to {results_path}")
    if len(results_df) > 0:
        print(f"Summary: {len(results_df['subject_id'].unique())} subjects, {len(results_df)} total outcome records")
    else:
        print("No results to summarize - dataset was empty or no valid subjects processed")
    
    return results_df

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_rollouts", type=int, default=2)
    parser.add_argument("--num_subjects_per_batch", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include_patients_without_end_token", action="store_true", help="Include patients without valid end tokens in evaluation")

    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    dataset_dir = args.dataset_dir
    vocab_path = os.path.join(experiment_dir, "vocab.csv")

    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        experiment_config = yaml.safe_load(f)

    model = load_model(experiment_config["model"])

    start_token_str = "HOSPITAL_ADMISSION//EW EMER.//EMERGENCY ROOM"
    end_token_strs = ["MEDS_DEATH", "TRANSFER_TO//discharge//UNKNOWN", "HOSPITAL_DISCHARGE//HOME", "HOSPITAL_DISCHARGE//UNK"]

    dataset = RolloutEvaluationDataset(dataset_dir, vocab_path, sequence_length=experiment_config["model"]["context_length"], start_token_str=start_token_str, end_token_strs=end_token_strs, include_patients_without_end_token=args.include_patients_without_end_token, logger=None)

    save_dir = os.path.join(experiment_dir, "evaluations/stop_condition_rollout")

    # Create and save config before running benchmark
    os.makedirs(save_dir, exist_ok=True)
    
    benchmark_config = {
        "experiment_dir": experiment_dir,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_rollouts": args.num_rollouts,
            "num_subjects_per_batch": args.num_subjects_per_batch,
            "temperature": args.temperature,
            "max_steps": args.max_steps,
            "device": args.device,
            "include_patients_without_end_token": args.include_patients_without_end_token
        },
        "dataset": {
            "name": "ethos_timetokens",
            "start_token": start_token_str,
            "end_tokens": end_token_strs
        }
    }
    
    config_path = os.path.join(save_dir, "rollout_config.json")
    with open(config_path, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    
    print(f"Saved config to {config_path}")

    # Run the benchmark
    results_df = rollout_benchmark(model, dataset, max_steps=args.max_steps, num_rollouts=args.num_rollouts, num_subjects_per_batch=args.num_subjects_per_batch, temperature=args.temperature, device=torch.device(args.device), save_dir=save_dir)