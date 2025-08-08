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
