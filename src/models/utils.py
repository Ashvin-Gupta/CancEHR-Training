import torch
from src.models.lstm import LSTM
from src.models.transformer_decoder import TransformerDecoder


def load_model(model_config: dict) -> torch.nn.Module:
    """
    Loads a model from a config file.

    Args:
        model_config (dict): The model configuration.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    # LSTM
    if model_config["type"] == "lstm":
        model = LSTM(
            vocab_size=model_config["vocab_size"],
            embedding_dim=model_config["embedding_dim"],
            hidden_dim=model_config["hidden_dim"],
            n_layers=model_config["n_layers"],
            dropout=model_config["dropout"],
        )

    # Transformer Decoder
    elif model_config["type"] == "transformer":
        model = TransformerDecoder(
            vocab_size=model_config["vocab_size"],
            model_dim=model_config["model_dim"],
            n_layers=model_config["n_layers"],
            dropout=model_config["dropout"],
            n_heads=model_config["n_heads"],
            context_length=model_config["context_length"],
        )
    else:
        raise ValueError(f"Model type {model_config['type']} not supported")

    return model

def sample_from_distribution(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Samples indices from a softmax distribution with optional temperature scaling.
    
    Supports both 1D (single distribution) and 2D (batch of distributions) inputs.
    Temperature = 0 results in deterministic (greedy) sampling.

    Args:
        probs (torch.Tensor): A tensor of shape [vocab_size] or [batch_size, vocab_size] 
                             representing probabilities (already softmaxed).
        temperature (float): Temperature to scale the distribution. Must be >= 0.
                           temperature = 0: deterministic (greedy) sampling
                           temperature = 1: natural randomness
                           temperature > 1: more random

    Returns:
        torch.Tensor: The sampled indices. Shape [1] for 1D input, [batch_size] for 2D input.
    """
    if temperature < 0:
        raise ValueError("Temperature must be >= 0")

    # Handle deterministic case (temperature = 0)
    if temperature == 0:
        return torch.argmax(probs, dim=-1)

    # Convert probabilities back to logits for temperature scaling
    # Add small epsilon to avoid log(0)
    logits = torch.log(probs + 1e-9)
    
    # Apply temperature scaling
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    scaled_probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
    
    # Sample from the scaled distribution
    return torch.multinomial(scaled_probs, num_samples=1).squeeze(-1)
