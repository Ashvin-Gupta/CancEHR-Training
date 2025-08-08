import os
import pandas as pd
import torch
import yaml
from src.models.utils import load_model

def run_evaluation(experiment_name: str, dataset: torch.utils.data.Dataset, device: torch.device) -> pd.DataFrame:
    """
    Run evaluation for a given experiment.

    Args:
        experiment_name (str): The name of the experiment.
        dataset (torch.utils.data.Dataset): The dataset to evaluate on.
        device (torch.device): The device to evaluate on.

    Returns:
        results_df (pd.DataFrame): A pandas DataFrame containing the evaluation results.
    """
    experiment_path = os.path.join("src/experiments/results", experiment_name)
    config = yaml.load(open(os.path.join(experiment_path, "config.yaml")), Loader=yaml.FullLoader)

    context_length = config["data"]["sequence_length"]

    # load model
    model = load_model(config["model"])
    model.load_state_dict(
        torch.load(os.path.join(experiment_path, "model.pth"), map_location=device)
    )
    model.to(device)
    model.eval()

    # initialise dict for storing results
    results = {
        "subject_id": [],
        "index": [],
        "input_tokens": [],
        "target_token": [],
        "prediction": [],
        "loss": [],
    }

    stride = context_length
    overlap = context_length - 1

    with torch.no_grad():
        for datapoint in dataset:
            tokens = datapoint["tokens"].to(device)  # (T,)
            subject_id = datapoint["subject_id"]
            T = tokens.size(0)

            offset = 0
            while offset < T - 1:  # we need at least one target
                # window is [offset : offset + stride]
                end = min(offset + stride, T - 1)  # leave room for a target
                input_window = tokens[offset:end]  # (<=stride,)
                target_window = tokens[offset + 1 : end + 1]

                logits = model(input_window.unsqueeze(0))  # (1, L, vocab)
                losses = torch.nn.functional.cross_entropy(
                    logits.squeeze(0),  # (L, vocab)
                    target_window,
                    reduction="none",
                )
                preds = logits.argmax(dim=-1).squeeze(0)  # (L,)

                # collect rows
                for j in range(len(target_window)):
                    idx = offset + j
                    start = max(0, idx - context_length + 1)
                    results["subject_id"].append(subject_id.item())
                    results["index"].append(idx)
                    results["input_tokens"].append(tokens[start : idx + 1].tolist())
                    results["target_token"].append(target_window[j].item())
                    results["prediction"].append(preds[j].item())
                    results["loss"].append(losses[j].item())

                offset += overlap
            break

    # turn the cache into a DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(experiment_path, "evaluation_results.csv"), index=False)

    return results_df

if __name__ == "__main__":
    import argparse
    from src.data.dataset import NightingaleEvaluationDataset

    parser = argparse.ArgumentParser(description="Run evaluation on a trained model")
    parser.add_argument("--experiment_name", type=str, required=True,
                      help="Name of the experiment to evaluate")
    parser.add_argument("--dataset_dir", type=str, required=True,
                      help="Path to the evaluation dataset directory")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run evaluation on (cuda/cpu)")
    args = parser.parse_args()

    # create dataset
    evaluation_dataset = NightingaleEvaluationDataset(args.dataset_dir)

    # run evaluation
    run_evaluation(args.experiment_name, evaluation_dataset, torch.device(args.device))
