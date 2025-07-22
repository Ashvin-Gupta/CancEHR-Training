import yaml
import os
import shutil   
from src.data.dataloader import get_dataloader
from src.models import LSTM
import torch
from src.training.train import train

def run_experiment(config_path: str, experiment_name: str):
    """
    Runs an experiment with a given config and experiment name.

    Args:
        config_path (str): The path to the config file.
        experiment_name (str): The name of the experiment.

    Experiment results are saved to src/experiments/results/{experiment_name}.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate config before running experiment
    validate_config(config)

    # Define experiment directory
    experiment_dir = os.path.join("src/experiments/results", experiment_name)

    # check if experiments directory exists, if so ask user if they want to overwrite
    if os.path.exists(experiment_dir):
        overwrite = input(f"Experiment directory {experiment_dir} already exists. Overwrite? (y/n): ")
        if overwrite != "y":
            print("Exiting...")
            return
        else:
            shutil.rmtree(experiment_dir)

    # Create experiment directory
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config to experiment directory
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Create dataloaders
    train_dataloader = get_dataloader(
        config['data']['train_dataset_dir'], 
        config['data']['batch_size'], 
        config['data']['shuffle'],
        config['data']['sequence_length'],
        mode="train"
    )
    val_dataloader = get_dataloader(
        config['data']['val_dataset_dir'], 
        config['data']['batch_size'], 
        config['data']['shuffle'],
        config['data']['sequence_length'],
        mode="eval"
    )

    # Create model
    if config['model']['type'] == "lstm":
        model = LSTM(config['model']['vocab_size'], config['model']['embedding_dim'], config['model']['hidden_dim'], config['model']['n_layers'], config['model']['dropout'])
    else:
        raise ValueError(f"Model type {config['model']['type']} not supported")

    # Create loss function
    if config['loss_function']['type'] == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Loss function type {config['loss_function']['type']} not supported")

    # Create optimiser
    if config['optimiser']['type'] == "adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config['optimiser']['lr'])
    else:
        raise ValueError(f"Optimiser type {config['optimiser']['type']} not supported")
    
    # Run training
    train(
        model=model, 
        experiment_dir=experiment_dir, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        optimiser=optimiser, 
        loss_function=loss_function, 
        device=config['training']['device'],
        epochs=config['training']['epochs'])


def validate_config(config: dict):
    """Validate that all required fields exist in the config."""
    required_fields = {
        'name': str,
        'model': dict,
        'optimiser': dict,
        'loss_function': dict,
        'training': dict,
        'data': dict
    }
    
    # Check top-level required fields
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: '{field}'")
        if not isinstance(config[field], expected_type):
            raise ValueError(f"Field '{field}' must be of type {expected_type.__name__}")
    
    # Validate model section
    model_required = ['type', 'vocab_size', 'embedding_dim', 'hidden_dim', 'n_layers', 'dropout']
    for field in model_required:
        if field not in config['model']:
            raise ValueError(f"Missing required field in model: '{field}'")
    
    # Validate optimiser section
    optimiser_required = ['type', 'lr']
    for field in optimiser_required:
        if field not in config['optimiser']:
            raise ValueError(f"Missing required field in optimiser: '{field}'")
    
    # Validate loss_function section
    if 'type' not in config['loss_function']:
        raise ValueError("Missing required field in loss_function: 'type'")
    
    # Validate training section
    training_required = ['epochs', 'device']
    for field in training_required:
        if field not in config['training']:
            raise ValueError(f"Missing required field in training: '{field}'")
    
    # Validate data section
    data_required = ['train_dataset_dir', 'val_dataset_dir', 'sequence_length', 'batch_size', 'shuffle']
    for field in data_required:
        if field not in config['data']:
            raise ValueError(f"Missing required field in data: '{field}'")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    config_path = os.path.join("src/experiments/configs", f"{args.config_name}.yaml")
    experiment_name = args.experiment_name

    run_experiment(config_path, experiment_name)