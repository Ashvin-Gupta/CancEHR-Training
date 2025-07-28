from src.models.lstm import LSTM
from src.models.transformer_decoder import TransformerDecoder

def load_model(model_config: dict):
    """
    Loads a model from a config file.

    Args:
        model_config (dict): The model configuration.

    Returns:
        model (torch.nn.Module): The loaded model.
    """

    # LSTM
    if model_config['type'] == "lstm":
        model = LSTM(model_config['vocab_size'], model_config['embedding_dim'], model_config['hidden_dim'], model_config['n_layers'], model_config['dropout'])
    
    # Transformer
    elif model_config['type'] == "transformer":
        model = TransformerDecoder(
            vocab_size=model_config['vocab_size'],
            embedding_dim=model_config['embedding_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_layers=model_config['n_layers'],
            dropout=model_config['dropout'],
            n_heads=model_config['n_heads'],
            max_len=model_config['max_len']
        )
    else:
        raise ValueError(f"Model type {model_config['type']} not supported")
    
    return model