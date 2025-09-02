import torch
from src.models.core_models.lstm import LSTM
from src.models.core_models.transformer_decoder import TransformerDecoder
from src.models.note_models.lstm_note import LSTMNote


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
        model = LSTM(model_config)

    # Transformer Decoder
    elif model_config["type"] == "transformer":
        model = TransformerDecoder(model_config)

    # LSTM with notes
    elif model_config["type"] == "lstm_note":
        model = LSTMNote(model_config)

    else:
        raise ValueError(f"Model type {model_config['type']} not supported")

    return model
