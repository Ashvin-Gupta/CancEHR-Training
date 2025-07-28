import torch
import os
import yaml
import pandas as pd

def run_evaluation(experiment_name: str, dataset: torch.utils.data.Dataset, device: torch.device):
    
    experiment_path = os.path.join("src/experiments/results", experiment_name)
    config = yaml.load(open(os.path.join(experiment_path, "config.yaml")), Loader=yaml.FullLoader)

    context_length = config["data"]["sequence_length"]

    # load model
    model = load_model(config["model"])
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pth"), map_location=device))
    model.to(device)
    model.eval()

    # initialise dict for storing results
    results = {
        "subject_id": [],
        "index":      [],
        "input_tokens": [],
        "target_token": [],
        "prediction": [],
        "loss":       [],
    }

    stride = context_length
    overlap = context_length - 1

    with torch.no_grad():
        for datapoint in dataset:
            tokens = datapoint["tokens"].to(device)          # (T,)
            subject_id = datapoint["subject_id"]
            T = tokens.size(0)

            offset = 0
            while offset < T - 1:                            # we need at least one target
                # window is [offset : offset + stride]
                end = min(offset + stride, T - 1)            # leave room for a target
                input_window  = tokens[offset : end]         # (<=stride,)
                target_window = tokens[offset + 1 : end + 1]

                logits = model(input_window.unsqueeze(0))    # (1, L, vocab)
                losses = torch.nn.functional.cross_entropy(
                    logits.squeeze(0),                       # (L, vocab)
                    target_window,
                    reduction="none"
                )
                preds  = logits.argmax(dim=-1).squeeze(0)    # (L,)

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

def load_model(model_config: dict):
    """
    Loads the model from the config file.

    Args:
        model_config (dict): The model configuration.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    if model_config["type"] == "lstm":
        from src.models.lstm import LSTM
        model = LSTM(model_config["vocab_size"], model_config["embedding_dim"], model_config["hidden_dim"], model_config["n_layers"], model_config["dropout"])
    elif model_config["type"] == "transformer":
        from src.models.transformer_decoder import TransformerDecoder
        model = TransformerDecoder(model_config["vocab_size"], model_config["embedding_dim"], model_config["hidden_dim"], model_config["n_layers"], model_config["dropout"], model_config["n_heads"], model_config["max_len"])
    else:
        raise ValueError(f"Model type {model_config['type']} not supported")
    
    return model

if __name__ == "__main__":

    import argparse
    from src.data.dataset import NightingaleEvaluationDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    # create dataloader
    evaluation_dataset_dir = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/quantile_bin_preprocessing_full/tuning"
    vocab_path = "/home/joshua/data/mimic_meds/mimic_iv_meds/tokenized_data/quantile_bin_preprocessing_full/vocab.csv"
    evaluation_dataset = NightingaleEvaluationDataset(evaluation_dataset_dir, vocab_path)

    for datapoint in evaluation_dataset:
        print(len(datapoint['tokens']))
        print(datapoint['subject_id'])
        end_tokens = datapoint['tokens'][-30:]
        token_strings = evaluation_dataset.tokens_to_strings(end_tokens)
        print(list(zip(end_tokens, token_strings)))
        input()

    # run_evaluation(args.experiment_name, evaluation_dataset, torch.device("cpu"))