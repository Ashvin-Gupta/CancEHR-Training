import csv
from collections import Counter, defaultdict
from pathlib import Path
import torch
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.models.utils import load_model

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent

app = FastAPI(title="Nightingale Visualization Server")

# Mount static files using absolute path
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Templates using absolute path
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def get_experiment_results() -> list[dict]:
    """
    Simple function to read experiment results from the results directory.

    Returns:
        experiments (list): A list of dictionaries containing the experiment results.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    experiments = []

    if not results_dir.exists():
        return experiments

    for experiment_dir in results_dir.iterdir():
        if experiment_dir.is_dir():
            config_path = experiment_dir / "config.yaml"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    # Get model size if available
                    model_path = experiment_dir / "model.pth"
                    model_size_mb = None
                    if model_path.exists():
                        size_bytes = model_path.stat().st_size
                        model_size_mb = round(size_bytes / (1024 * 1024), 1)

                    # Get best validation loss from loss.log
                    best_val_loss = None
                    loss_log_path = experiment_dir / "loss.log"
                    if loss_log_path.exists():
                        try:
                            val_losses = []
                            with open(loss_log_path) as f:
                                reader = csv.DictReader(f)
                                for row in reader:
                                    val_losses.append(float(row["val_loss"]))
                            if val_losses:
                                best_val_loss = min(val_losses)
                        except Exception as e:
                            print(f"Error reading loss log for {experiment_dir.name}: {e}")

                    # Extract dataset name from config paths
                    dataset_name = None
                    try:
                        # Check data configuration for train/val paths
                        data_config = config.get("data", {})

                        # Look for train or validation dataset paths (using correct field names)
                        train_path = data_config.get("train_dataset_dir", "")
                        val_path = data_config.get("val_dataset_dir", "")

                        # Try to extract dataset name from either path
                        for path in [train_path, val_path]:
                            if path and isinstance(path, str):
                                # Look for pattern before '/train' or '/tuning'
                                import re

                                match = re.search(r"/([^/]+)/(train|tuning)/?$", path)
                                if match:
                                    dataset_name = match.group(1)
                                    break
                                # Alternative: look for the last meaningful directory name
                                elif "/tokenized_data/" in path:
                                    parts = path.split("/tokenized_data/")
                                    if len(parts) > 1:
                                        remaining = parts[1].split("/")
                                        if len(remaining) > 1:
                                            dataset_name = remaining[0]
                                            break

                        # Debug logging
                        print(
                            f"Experiment {experiment_dir.name}: train_path={train_path}, val_path={val_path}, dataset_name={dataset_name}"
                        )

                    except Exception as e:
                        print(f"Error extracting dataset name for {experiment_dir.name}: {e}")

                    experiments.append(
                        {
                            "name": experiment_dir.name,  # Use folder name
                            "model_type": config.get("model", {}).get("type", "unknown"),
                            "sequence_length": config.get("data", {}).get("sequence_length"),
                            "model_size_mb": model_size_mb,
                            "best_val_loss": best_val_loss,
                            "dataset_name": dataset_name,
                            "has_loss_log": (experiment_dir / "loss.log").exists(),
                            "has_simulations": (experiment_dir / "simulations").exists(),
                        }
                    )
                except Exception as e:
                    print(f"Error reading {experiment_dir.name}: {e}")

    return experiments


def get_loss_data() -> list[dict]:
    """
    Parse loss logs from all experiments.

    Returns:
        loss_data (list): A list of dictionaries containing the loss data for each experiment.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    loss_data = []

    if not results_dir.exists():
        return loss_data

    for experiment_dir in results_dir.iterdir():
        if experiment_dir.is_dir():
            loss_log_path = experiment_dir / "loss.log"
            if loss_log_path.exists():
                try:
                    epochs = []
                    train_losses = []
                    val_losses = []

                    with open(loss_log_path) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            epochs.append(int(row["epoch"]))
                            train_losses.append(float(row["train_loss"]))
                            val_losses.append(float(row["val_loss"]))

                    # Get model type from config
                    config_path = experiment_dir / "config.yaml"
                    model_type = "unknown"
                    if config_path.exists():
                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                            model_type = config.get("model", {}).get("type", "unknown")

                    loss_data.append(
                        {
                            "name": experiment_dir.name,
                            "model_type": model_type,
                            "epochs": epochs,
                            "train_losses": train_losses,
                            "val_losses": val_losses,
                        }
                    )
                except Exception as e:
                    print(f"Error reading loss log for {experiment_dir.name}: {e}")

    return loss_data


def get_simulation_experiments() -> list[dict]:
    """
    Get experiments that have simulation data.

    Returns:
        simulation_experiments (list): A list of dictionaries containing the simulation experiments.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    simulation_experiments = []

    if not results_dir.exists():
        return simulation_experiments

    for experiment_dir in results_dir.iterdir():
        if experiment_dir.is_dir():
            simulations_dir = experiment_dir / "simulations"
            if simulations_dir.exists():
                # Find simulation files
                simulation_files = []
                for sim_subdir in simulations_dir.iterdir():
                    if sim_subdir.is_dir():
                        csv_file = sim_subdir / "simulation_results.csv"
                        if csv_file.exists():
                            simulation_files.append({"id": sim_subdir.name, "path": str(csv_file)})

                if simulation_files:
                    # Get model type from config
                    config_path = experiment_dir / "config.yaml"
                    model_type = "unknown"
                    if config_path.exists():
                        try:
                            with open(config_path) as f:
                                config = yaml.safe_load(f)
                                model_type = config.get("model", {}).get("type", "unknown")
                        except:
                            pass

                    simulation_experiments.append(
                        {
                            "name": experiment_dir.name,
                            "model_type": model_type,
                            "simulation_files": simulation_files,
                        }
                    )

    return simulation_experiments


def analyze_simulation_data(csv_path: str, experiment_name: str) -> dict:
    """
    Analyze simulation CSV data and return visualization data.

    Args:
        csv_path (str): The path to the simulation CSV file.
        experiment_name (str): The name of the experiment.
    """
    try:
        # Read vocab.csv to map token IDs to strings
        vocab_map = {}
        results_dir = BASE_DIR.parent.parent / "experiments" / "results"
        vocab_path = results_dir / experiment_name / "vocab.csv"

        if vocab_path.exists():
            with open(vocab_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    token_id = int(row["token"])
                    token_str = row["str"]
                    vocab_map[token_id] = token_str

        # Read the CSV data
        data = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(
                    {
                        "start_token_index": int(row["start_token_index"]),
                        "simulation_index": int(row["simulation_index"]),
                        "stop_reason": row["stop_reason"],
                        "stop_value": float(row["stop_value"]) if row["stop_value"] else None,
                        "stop_step": int(row["stop_step"]) if row["stop_step"] else None,
                    }
                )

        # Analysis 1: Percentage of simulations ending in max_steps vs stop_token by start_token_index
        stop_reason_by_start = defaultdict(lambda: Counter())
        for row in data:
            stop_reason_by_start[row["start_token_index"]][row["stop_reason"]] += 1

        stop_reason_percentages = {}
        start_indices = sorted(stop_reason_by_start.keys())

        for start_idx in start_indices:
            total = sum(stop_reason_by_start[start_idx].values())
            stop_reason_percentages[start_idx] = {
                "max_steps_pct": (stop_reason_by_start[start_idx]["max_steps"] / total) * 100
                if total > 0
                else 0,
                "stop_token_pct": (stop_reason_by_start[start_idx]["stop_token"] / total) * 100
                if total > 0
                else 0,
            }

        # Analysis 2: Percentage distribution of different stop_values for stop_token simulations by start_token_index
        stop_values_by_start = defaultdict(lambda: Counter())
        all_stop_values = set()

        for row in data:
            if row["stop_reason"] == "stop_token" and row["stop_value"] is not None:
                stop_value = int(row["stop_value"])  # Convert to int for cleaner categories
                stop_values_by_start[row["start_token_index"]][stop_value] += 1
                all_stop_values.add(stop_value)

        # Calculate percentage distribution for each stop value at each start index
        stop_value_percentages = {}
        all_stop_values = sorted(list(all_stop_values))

        for start_idx in sorted(stop_values_by_start.keys()):
            total_stop_tokens = sum(stop_values_by_start[start_idx].values())
            stop_value_percentages[start_idx] = {}

            for stop_value in all_stop_values:
                count = stop_values_by_start[start_idx][stop_value]
                percentage = (count / total_stop_tokens) * 100 if total_stop_tokens > 0 else 0
                stop_value_percentages[start_idx][stop_value] = percentage

        # Create readable labels for stop values
        stop_value_labels = {}
        for stop_value in all_stop_values:
            if stop_value in vocab_map:
                # Clean up the token string for display
                token_str = vocab_map[stop_value].strip().replace("\n", "\\n").replace("\t", "\\t")
                if len(token_str) > 35:  # Increased from 20 to 35 characters
                    token_str = token_str[:32] + "..."
                stop_value_labels[stop_value] = f"{token_str} ({stop_value})"
            else:
                stop_value_labels[stop_value] = f"Token {stop_value}"

        return {
            "start_indices": start_indices,
            "stop_reason_percentages": stop_reason_percentages,
            "stop_value_percentages": stop_value_percentages,
            "all_stop_values": all_stop_values,
            "stop_value_labels": stop_value_labels,
            "total_simulations": len(data),
        }

    except Exception as e:
        print(f"Error analyzing simulation data: {e}")
        return None


def get_experiments_with_models() -> list[dict]:
    """
    Get list of experiments that have trained models.

    Returns:
        experiments (list): A list of dictionaries containing the experiment results.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    experiments = []

    if not results_dir.exists():
        return experiments

    for experiment_dir in results_dir.iterdir():
        if experiment_dir.is_dir():
            config_path = experiment_dir / "config.yaml"
            model_path = experiment_dir / "model.pth"
            vocab_path = experiment_dir / "vocab.csv"

            if config_path.exists() and model_path.exists() and vocab_path.exists():
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    experiments.append(
                        {
                            "name": experiment_dir.name,
                            "model_type": config.get("model", {}).get("type", "unknown"),
                            "vocab_size": config.get("model", {}).get("vocab_size", 0),
                            "sequence_length": config.get("data", {}).get("sequence_length", 0),
                        }
                    )
                except Exception as e:
                    print(f"Error reading experiment {experiment_dir.name}: {e}")

    return experiments


class InferenceRequest(BaseModel):
    experiment_name: str
    input_tokens: list[int]
    top_k: int = 10


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Serve the homepage template with experiment data.

    Args:
        request (Request): The request object.
    """
    experiments = get_experiment_results()
    return templates.TemplateResponse(
        "index.html", {"request": request, "experiments": experiments}
    )


@app.get("/losses", response_class=HTMLResponse)
async def losses(request: Request):
    """
    Serve the loss visualization page.

    Args:
        request (Request): The request object.
    """
    loss_data = get_loss_data()
    return templates.TemplateResponse("losses.html", {"request": request, "loss_data": loss_data})


@app.get("/simulations", response_class=HTMLResponse)
async def simulations(request: Request):
    """
    Serve the simulations visualization page.

    Args:
        request (Request): The request object.
    """
    simulation_experiments = get_simulation_experiments()
    return templates.TemplateResponse(
        "simulations.html", {"request": request, "simulation_experiments": simulation_experiments}
    )


@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request):
    """
    Serve the playground page.

    Args:
        request (Request): The request object.
    """
    experiments = get_experiments_with_models()
    return templates.TemplateResponse(
        "playground.html", {"request": request, "experiments": experiments}
    )


@app.get("/api/experiment-vocab")
async def get_experiment_vocab(experiment: str):
    """
    API endpoint to get vocabulary for an experiment.

    Args:
        experiment (str): The name of the experiment.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    vocab_path = results_dir / experiment / "vocab.csv"

    if not vocab_path.exists():
        return {"error": "Vocabulary file not found"}

    try:
        vocab_map = {}
        with open(vocab_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                token_id = int(row["token"])
                token_str = row["str"]
                vocab_map[token_id] = token_str

        return {"vocab": vocab_map}
    except Exception as e:
        return {"error": f"Failed to read vocabulary: {str(e)}"}


@app.post("/api/inference")
async def inference(request: InferenceRequest):
    """
    API endpoint for model inference.

    Args:
        request (InferenceRequest): The inference request.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    experiment_dir = results_dir / request.experiment_name

    config_path = experiment_dir / "config.yaml"
    model_path = experiment_dir / "model.pth"
    vocab_path = experiment_dir / "vocab.csv"

    if not all(p.exists() for p in [config_path, model_path, vocab_path]):
        return {"error": "Required experiment files not found"}

    try:
        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load vocabulary
        vocab_map = {}
        with open(vocab_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                token_id = int(row["token"])
                token_str = row["str"]
                vocab_map[token_id] = token_str

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config["model"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Prepare input
        input_tensor = torch.tensor(request.input_tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            logits = model(input_tensor)  # Shape: (1, seq_len, vocab_size)
            # Get logits for the last token
            last_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)

            # Get top-k predictions
            top_k_values, top_k_indices = torch.topk(last_token_logits, k=request.top_k, dim=-1)
            probabilities = torch.softmax(last_token_logits, dim=-1)
            top_k_probs = probabilities[top_k_indices]

            # Format results
            predictions = []
            for i in range(request.top_k):
                token_id = top_k_indices[i].item()
                token_str = vocab_map.get(token_id, f"<UNK:{token_id}>")
                predictions.append(
                    {
                        "token_id": token_id,
                        "token_str": token_str,
                        "probability": top_k_probs[i].item(),
                        "logit": top_k_values[i].item(),
                    }
                )

        return {"predictions": predictions, "input_length": len(request.input_tokens)}

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}


@app.get("/api/simulation-data")
async def get_simulation_data(experiment: str, simulation_id: str):
    """
    API endpoint to get simulation analysis data.

    Args:
        experiment (str): The name of the experiment.
        simulation_id (str): The ID of the simulation.
    """
    results_dir = BASE_DIR.parent.parent / "experiments" / "results"
    csv_path = results_dir / experiment / "simulations" / simulation_id / "simulation_results.csv"

    if not csv_path.exists():
        return {"error": "Simulation file not found"}

    analysis = analyze_simulation_data(csv_path, experiment)
    if analysis is None:
        return {"error": "Failed to analyze simulation data"}

    return analysis


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        status (dict): A dictionary containing the status of the server.
    """
    return {"status": "healthy"}


@app.get("/favicon.ico")
async def favicon():
    """
    Serve favicon from static directory.

    Returns:
        favicon (FileResponse): The favicon file.
    """
    from fastapi.responses import FileResponse

    favicon_path = BASE_DIR / "static" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    else:
        return {"error": "Favicon not found"}


if __name__ == "__main__":
    # Determine the correct module path based on how this script is being run
    import sys

    module_name = __name__
    if hasattr(sys.modules[__name__], "__package__") and sys.modules[__name__].__package__:
        # Running as a module (e.g., python -m src.evaluation.visualisation_server.main)
        module_path = f"{sys.modules[__name__].__package__}.main:app"
    else:
        # Running directly (e.g., python main.py)
        module_path = "main:app"

    uvicorn.run(module_path, host="0.0.0.0", port=8001, reload=True)
