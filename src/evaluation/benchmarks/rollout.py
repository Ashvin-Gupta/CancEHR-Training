from src.data.evaluation_datasets import RolloutEvaluationDataset
import torch
from tqdm import tqdm
from src.models.utils import load_model, sample_from_distribution
import os
import yaml
import pandas as pd
import json
from datetime import datetime
from collections import deque

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
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert dataset to list for processing
    dataset_list = list(dataset)
    total_subjects = len(dataset_list)
    
    # Create rollout queue - each item is (subject_data, rollout_idx, unique_rollout_id)
    rollout_queue = deque()
    rollout_id_counter = 0
    
    for subject_data in dataset_list:
        for rollout_idx in range(num_rollouts):
            rollout_queue.append((subject_data, rollout_idx, rollout_id_counter))
            rollout_id_counter += 1
    
    total_rollouts = len(rollout_queue)
    inference_count = 0  # Initialize inference counter
    
    if total_rollouts == 0:
        results_df, simulations_df = pd.DataFrame(), pd.DataFrame()
    else:
        batch_size = min(num_subjects_per_batch * num_rollouts, total_rollouts)
        
        # Initialize batch tensors and tracking structures
        seq_length = dataset_list[0]['input_tokens'].shape[0]
        current_tokens = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        active_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Tracking for each slot
        slot_to_rollout = {}  # slot_idx -> (subject_data, rollout_idx, unique_rollout_id)
        initial_input_tokens = {}  # unique_rollout_id -> initial_tokens
        
        # Results collection
        all_results = []
        all_simulations = []
        completed_rollouts = []
        
        def _fill_empty_slots():
            """Fill empty slots with rollouts from the queue."""
            for slot_idx in range(batch_size):
                if not active_mask[slot_idx] and rollout_queue:
                    # Load new rollout into this slot
                    subject_data, rollout_idx, unique_rollout_id = rollout_queue.popleft()
                    input_tokens = subject_data['input_tokens']
                    
                    # Initialize slot
                    current_tokens[slot_idx] = input_tokens
                    active_mask[slot_idx] = True
                    slot_to_rollout[slot_idx] = (subject_data, rollout_idx, unique_rollout_id)
                    initial_input_tokens[unique_rollout_id] = input_tokens.clone()
        
        def _process_completed_rollout(slot_idx, step, token, is_end):
            """Process a completed rollout and free up the slot."""
            subject_data, rollout_idx, unique_rollout_id = slot_to_rollout[slot_idx]
            
            # Get subject info
            subject_id = subject_data["subject_id"].item()
            end_token_value = int(subject_data['end_token'].item())
            if end_token_value == -1:
                real_end_token_str = "max_steps"
                real_end_token_steps = -1
            else:
                real_end_token_str = dataset.vocab[dataset.vocab["token"] == end_token_value]["str"].values[0]
                real_end_token_steps = (subject_data['end_token_idx'] - subject_data['start_token_idx']).item()
            
            # Determine outcome and steps
            if is_end:
                rollout_outcome = dataset.vocab[dataset.vocab["token"] == token]["str"].values[0]
                steps_taken = step + 1
            else:
                rollout_outcome = "max_steps"
                steps_taken = max_steps
            
            # Create simulation record
            simulation_row = {
                "subject_id": subject_id,
                "real_end_token_str": real_end_token_str,
                "real_end_token_steps": real_end_token_steps,
                "outcome": rollout_outcome,
                "input_tokens": initial_input_tokens[unique_rollout_id].cpu().tolist(),
                "predicted_tokens": token if token is not None else None,
                "steps_taken": steps_taken
            }
            all_simulations.append(simulation_row)
            completed_rollouts.append((subject_data, rollout_outcome, steps_taken))
            
            # Mark slot as inactive and clean up
            active_mask[slot_idx] = False
            del slot_to_rollout[slot_idx]
        
        # Initial slot filling
        _fill_empty_slots()
        
        # Main prediction loop
        with torch.no_grad():
            progress_bar = tqdm(total=total_rollouts, desc="Dynamic rollout processing")
            completed_count = 0
            save_frequency = max(1, total_rollouts // 20)  # Save 20 times during execution
            
            for step in range(max_steps):
                if not active_mask.any() and not rollout_queue:
                    break  # All rollouts completed
                
                if not active_mask.any():
                    continue  # No active rollouts, but queue might have items
                
                # Get predictions for active rollouts
                active_indices = torch.where(active_mask)[0]
                if len(active_indices) == 0:
                    continue
                
                active_tokens = current_tokens[active_indices]
                output = model({'ehr': {'input_token_ids': active_tokens}})
                inference_count += 1  # Track model inference calls
                
                # Sample next tokens
                logits = output[:, -1]
                probs = torch.softmax(logits, dim=-1)
                next_tokens = sample_from_distribution(probs, temperature=temperature)
                
                # Check for end tokens
                is_end_token = torch.isin(next_tokens, end_token_tensor)
                
                # Process each active rollout
                for i, slot_idx in enumerate(active_indices):
                    slot_idx = slot_idx.item()  # Convert tensor to Python int
                    token = next_tokens[i].item()
                    is_end = is_end_token[i].item()
                    
                    if is_end or step == max_steps - 1:
                        # Rollout finished
                        _process_completed_rollout(slot_idx, step, token, is_end)
                        completed_count += 1
                    else:
                        # Update token sequence for next iteration
                        current_tokens[slot_idx] = torch.cat([
                            current_tokens[slot_idx, 1:], 
                            torch.tensor([token], device=device)
                        ])
                
                # Fill empty slots with new rollouts
                _fill_empty_slots()
                
                # Update progress bar with both counters
                progress_bar.n = completed_count
                progress_bar.postfix = f"Inference Calls: {inference_count}"
                progress_bar.refresh()
                
                # Save results periodically
                if completed_count > 0 and completed_count % save_frequency == 0:
                    if all_simulations:
                        simulations_df = pd.DataFrame(all_simulations)
                        simulations_path = os.path.join(save_dir, "simulations.csv")
                        simulations_df.to_csv(simulations_path, index=False)
            
            progress_bar.close()
        
        # Create final aggregated results
        subject_rollouts = {}
        for subject_data, outcome, steps in completed_rollouts:
            subject_id = subject_data["subject_id"].item()
            if subject_id not in subject_rollouts:
                subject_rollouts[subject_id] = {
                    'data': subject_data,
                    'outcomes': [],
                    'steps': []
                }
            subject_rollouts[subject_id]['outcomes'].append(outcome)
            subject_rollouts[subject_id]['steps'].append(steps)
        
        # Create aggregated results
        for subject_id, info in subject_rollouts.items():
            subject_data = info['data']
            outcomes = info['outcomes']
            steps_list = info['steps']
            
            # Get subject info
            end_token_value = int(subject_data['end_token'].item())
            if end_token_value == -1:
                real_end_token_str = "max_steps"
                real_end_token_steps = -1
            else:
                real_end_token_str = dataset.vocab[dataset.vocab["token"] == end_token_value]["str"].values[0]
                real_end_token_steps = (subject_data['end_token_idx'] - subject_data['start_token_idx']).item()
            
            # Count outcomes
            outcome_counts = {}
            for outcome in outcomes:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            avg_steps_taken = sum(steps_list) / len(steps_list) if steps_list else 0
            total_rollouts_for_subject = len(outcomes)
            
            # Create result rows (one per outcome)
            for outcome, count in outcome_counts.items():
                proportion = count / total_rollouts_for_subject if total_rollouts_for_subject > 0 else 0
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
        
        # Create final dataframes
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        simulations_df = pd.DataFrame(all_simulations) if all_simulations else pd.DataFrame()
        
        # Final save
        if not results_df.empty:
            results_path = os.path.join(save_dir, "rollout_results.csv")
            results_df.to_csv(results_path, index=False)
        
        if not simulations_df.empty:
            simulations_path = os.path.join(save_dir, "simulations.csv")
            simulations_df.to_csv(simulations_path, index=False)
    
    # Final summary
    results_path = os.path.join(save_dir, "rollout_results.csv")
    simulations_path = os.path.join(save_dir, "simulations.csv")
    print(f"Saved {len(results_df)} outcome records to {results_path}")
    print(f"Saved {len(simulations_df)} simulation records to {simulations_path}")
    if len(results_df) > 0:
        # Calculate inference efficiency
        efficiency = len(simulations_df) / inference_count if inference_count > 0 else 0
        print(f"Summary: {len(results_df['subject_id'].unique())} subjects, {len(results_df)} total outcome records, {len(simulations_df)} total simulations")
        print(f"Model efficiency: {inference_count} total inferences, {efficiency:.2f} rollouts per inference")
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