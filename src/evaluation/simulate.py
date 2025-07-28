import torch
import os
import yaml
from src.experiments.utils import load_model
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
        device: torch.device = torch.device("cpu")
    ):
    """
    Simulates a subject's data using a trained model.

    Args:
        tokens (torch.Tensor): The tokens for the subject.
        stop_tokens_indices (list): The indices of the stop tokens.
        experiment_dir (str): The directory containing the experiment.
        simulations_per_step (int): The number of simulations to run at each token step.
        device (torch.device): The device to run the model on.
    """
    
    # load config
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # create save directory
    os.makedirs(save_dir, exist_ok=True)

    context_length = config['data']['sequence_length']

    # load model
    model = load_model(config['model'])
    model.load_state_dict(torch.load(os.path.join(experiment_dir, "model.pth")))
    model.to(device)
    model.eval()

    vocab = pd.read_csv("/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/quantile_bin_preprocessing_full/vocab.csv")
    print(vocab.head())

    # simulate
    simulation_results = []
    for start_token_index in tqdm(range(50, len(tokens)), desc="Start Token Index", leave=True):
        input_tokens = tokens[:start_token_index]

        for simulation_index in tqdm(range(simulations_per_step), desc="Simulations", leave=False):

            simulation_tokens = input_tokens.clone()

            stop = {
                "reason": "max_steps",
                "value": None,
                "step": max_steps,
            }

            for step in tqdm(range(max_steps), desc="Step", leave=False):

                # TODO check if this should be -context_length or -(context_length - 1)
                x = simulation_tokens[-context_length:].to(device)
                x = x.unsqueeze(0) # add batch dimension
                pred = model(x)
                pred = pred.squeeze(0) # remove batch dimension
                next_token_distribution = torch.softmax(pred[-1, :], dim=-1)
                next_token = sample_from_distribution(next_token_distribution)
                simulation_tokens = torch.cat([simulation_tokens, torch.tensor([next_token])], dim=0)

                print(" ".join(vocab.iloc[simulation_tokens.tolist()]['str']))
                input()

                if next_token in stop_tokens_indices:
                    stop = {
                        "reason": "stop_token",
                        "value": next_token,
                        "step": step
                    }
                    break

            predicted_tokens = simulation_tokens[start_token_index:].tolist()

            result = {
                    "start_token_index": start_token_index,
                    "simulation_index": simulation_index,
                    "stop_reason": stop["reason"],
                    "stop_value": stop["value"],
                    "stop_step": stop["step"],
                    "predicted_tokens": predicted_tokens,
                }
            

            simulation_results.append(result)

    results_df = pd.DataFrame(simulation_results)
    results_df.to_csv(os.path.join(save_dir, f"simulation_results.csv"), index=False)    

if __name__ == "__main__":

    import argparse
    from src.data.dataset import NightingaleEvaluationDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    # calculate stop tokens
    stop_tokens_str = [
        "MEDS_DEATH",
        "TRANSFER_TO//discharge//UNKNOWN",
        "HOSPITAL_DISCHARGE//HOME",
        "HOSPITAL_DISCHARGE//UNK",
        "HOSPITAL_DISCHARGE//SKILLED",
        "TRANSFER_TO//admit//Discharge",
        "TRANSFER_TO//ED//Emergency",
    ]

    # load experiment directory
    experiment_dir = f"src/experiments/results/{args.experiment_name}"
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory {experiment_dir} does not exist")
        exit()

    # Create eval dataset
    data_dir = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/quantile_bin_preprocessing_full/tuning"
    vocab_path = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/quantile_bin_preprocessing_full/vocab.csv"
    eval_dataset = NightingaleEvaluationDataset(data_dir, vocab_path)

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
        simulations_per_step=10,
        device=torch.device("cuda:0")
    )