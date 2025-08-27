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
    return torch.multinomial(scaled_probs, num_samples=1)
