import torch
import math

class TransformerDecoder(torch.nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.5, n_heads: int = 8, max_len: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, dropout, max_len)
        
        # Transformer decoder layers
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, n_layers)
        
        # Output projection
        self.fc = torch.nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights using Xavier uniform initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weight matrices using Xavier uniform initialization.
        Leaves biases to be initialized by PyTorch.
        """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass of the transformer decoder model.

        Args:
            x (torch.Tensor): The input token sequence of shape (batch_size, sequence_length).

        Returns:
            y (torch.Tensor): The output logits of shape (batch_size, sequence_length, vocab_size). The logits are the
                unnormalized probabilities of the next token in the sequence.
        """
        # embed token sequence
        embedded = self.embedding(x)
        
        # add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # create causal mask for decoder (to prevent attending to future tokens)
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # pass through transformer decoder
        # For decoder-only models, we use the same sequence as both target and memory
        output = self.transformer_decoder(embedded, embedded, tgt_mask=causal_mask)
        
        # pass through linear layer
        y = self.fc(output)
        
        return y

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder attention.

        Args:
            seq_len (int): The length of the sequence.
            device (torch.device): The device to store the mask on.

        Returns:
            mask (torch.Tensor): The causal mask of shape (seq_len, seq_len).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class PositionalEncoding(torch.nn.Module):
    """Positional encoding for transformer models.

    Args:
        d_model (int): The dimension of the model.
        dropout (float): The dropout rate.
        max_len (int): The maximum length of the sequence.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            x (torch.Tensor): Embeddings with positional encoding added of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


if __name__ == "__main__":
    # define params
    batch_size = 3
    sequence_length = 10
    vocab_size = 100
    num_heads = 8
    embedding_dim = num_heads * 64
    hidden_dim = embedding_dim * 4
    n_layers = 2
    dropout = 0.5
    max_len = 512

    # random input
    x = torch.randint(0, vocab_size, (batch_size, sequence_length))
    print(f"Random input: {x.shape}")

    # init model and forward pass
    model = TransformerDecoder(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, n_heads=num_heads, max_len=max_len)
    y = model(x)
    print(f"Output: {y.shape}")
