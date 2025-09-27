from src.data.evaluation_datasets import RolloutEvaluationDataset
from src.evaluation.benchmarks.rollout import rollout_benchmark
from src.models.utils import load_model
import torch
import os
import yaml
import json
from datetime import datetime
import argparse

def icu_mortality_benchmark(experiment_dir: str, dataset_dir: str, num_rollouts: int = 5, 
                          num_subjects_per_batch: int = 64, temperature: float = 1.0, 
                          max_steps: int = 512, device: str = "cuda", hours_offset: int = 24):
    """
    Runs ICU mortality prediction benchmark using ICU admission + specified hours of data.
    
    Args:
        experiment_dir (str): Directory containing model and config
        dataset_dir (str): Directory containing evaluation data
        num_rollouts (int): Number of rollouts per subject
        num_subjects_per_batch (int): Batch size for processing
        temperature (float): Sampling temperature
        max_steps (int): Maximum prediction steps per individual rollout
        device (str): Device for inference
        hours_offset (int): Hours of data to include after ICU admission
    """
    
    # Load experiment config
    vocab_path = os.path.join(experiment_dir, "vocab.csv")
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        experiment_config = yaml.safe_load(f)
    
    # Load model
    model = load_model(experiment_config["model"])
    
    # ICU mortality evaluation setup
    start_token_str = "ICU_ADMISSION//Medical Intensive Care Unit (MICU)"
    end_token_strs = ["MEDS_DEATH", "ICU_DISCHARGE//Medical Intensive Care Unit (MICU)"]
    seconds_offset = hours_offset * 60 * 60  # Convert hours to seconds
    
    # Create dataset with time-based evaluation
    dataset = RolloutEvaluationDataset(
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        sequence_length=experiment_config["model"]["context_length"],
        start_token_str=start_token_str,
        end_token_strs=end_token_strs,
        seconds_offset=seconds_offset,
        include_patients_without_end_token=False,
        logger=None
    )
    
    print(f"ICU Mortality Benchmark - {hours_offset}h prediction")
    print(f"Dataset: {len(dataset)} subjects")
    print(f"Start token: {start_token_str}")
    print(f"End tokens: {end_token_strs}")
    print(f"Time window: {hours_offset} hours after ICU admission")

    if len(dataset) == 0:
        raise ValueError("No matching datapoints found in dataset, check start and end tokens")
    
    # Setup save directory
    save_dir = os.path.join(experiment_dir, f"evaluations/icu_mortality_{hours_offset}h")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create benchmark config
    benchmark_config = {
        "benchmark_type": "icu_mortality",
        "experiment_dir": experiment_dir,
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_rollouts": num_rollouts,
            "num_subjects_per_batch": num_subjects_per_batch,
            "temperature": temperature,
            "max_steps": max_steps,
            "device": device,
            "hours_offset": hours_offset,
            "seconds_offset": seconds_offset
        },
        "dataset": {
            "start_token": start_token_str,
            "end_tokens": end_token_strs,
            "sequence_length": experiment_config["model"]["context_length"],
            "total_subjects": len(dataset)
        }
    }
    
    # Save config
    config_path = os.path.join(save_dir, "benchmark_config.json")
    with open(config_path, 'w') as f:
        json.dump(benchmark_config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Run benchmark
    results_df = rollout_benchmark(
        model=model,
        dataset=dataset,
        max_steps=max_steps,
        num_rollouts=num_rollouts,
        num_subjects_per_batch=num_subjects_per_batch,
        temperature=temperature,
        device=torch.device(device),
        save_dir=save_dir
    )
    
    # Print summary statistics
    print(f"\n=== ICU Mortality Benchmark Results ({hours_offset}h) ===")
    total_subjects = len(results_df['subject_id'].unique())
    mortality_predictions = len(results_df[results_df['outcome'] == 'MEDS_DEATH'])
    discharge_predictions = len(results_df[results_df['outcome'].str.contains('DISCHARGE', na=False)])
    
    print(f"Total subjects: {total_subjects}")
    print(f"Mortality predictions: {mortality_predictions}")
    print(f"Discharge predictions: {discharge_predictions}")
    print(f"Results saved to: {save_dir}")
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICU Mortality Prediction Benchmark")
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--num_rollouts", type=int, default=5, help="Number of rollouts per subject")
    parser.add_argument("--num_subjects_per_batch", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max_steps", type=int, default=512, help="Maximum prediction steps per individual rollout")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--hours_offset", type=int, default=24, help="Hours of data after ICU admission")
    
    args = parser.parse_args()

    # print args
    print(args)
    
    results = icu_mortality_benchmark(
        experiment_dir=args.experiment_dir,
        dataset_dir=args.dataset_dir,
        num_rollouts=args.num_rollouts,
        num_subjects_per_batch=args.num_subjects_per_batch,
        temperature=args.temperature,
        max_steps=args.max_steps,
        device=args.device,
        hours_offset=args.hours_offset
    )