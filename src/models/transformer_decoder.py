import torch
import math
from src.models.multihead_attention import MultiHeadAttention

class TransformerDecoder(torch.nn.Module):
    """
    Implementation of a GPT-style transformer decoder (https://arxiv.org/abs/1706.03762)

    Args:
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embedding.
        hidden_dim (int): The dimension of the hidden layer.
        n_layers (int): The number of layers in the transformer.
        dropout (float): The dropout rate.
        n_heads (int): The number of attention heads.
    """
    def __init__(self, vocab_size: int, model_dim: int, n_layers: int = 2, dropout: float = 0.5, n_heads: int = 8, context_length: int = 512):
        super().__init__()
        self.model_dim = model_dim
        self.n_layers = n_layers
        self.context_length = context_length
        
        # Embedding matrix
        self.embedding = torch.nn.Embedding(vocab_size, model_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(model_dim, dropout, context_length)
        
        # Create the transformer decoder layers
        # input layer
        self.layers = torch.nn.ModuleList([TransformerDecoderBlock(d_input=model_dim, d_hidden=model_dim, d_output=model_dim, n_heads=n_heads, dropout=dropout)])

        # hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(TransformerDecoderBlock(d_input=model_dim, d_hidden=model_dim, d_output=model_dim, n_heads=n_heads, dropout=dropout))
        
        # output projection
        self.linear = torch.nn.Linear(model_dim, vocab_size)
        
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

        # embed token sequence with positional encoding
        embedded = self.embedding(x)
        embedded = self.pos_encoding(embedded)
        
        # pass through transformer decoder layers sequentially
        output = embedded
        for layer in self.layers:
            output = layer(output)
        
        # pass through linear layer
        y = self.linear(output)
                
        return y
    
class TransformerDecoderBlock(torch.nn.Module):
    """
    Implementation of a single transformer decoder block (https://arxiv.org/abs/1706.03762)

    Args:
        d_model (int): The dimension of the model.
        n_heads (int): The number of attention heads.
        dim_feedforward (int): The dimension of the feedforward layer.
    """
    def __init__(self, d_input: int, d_hidden: int, d_output: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = MultiHeadAttention(d_input, d_hidden, n_heads, dropout=dropout)

        # layer norm 1
        self.norm1 = torch.nn.LayerNorm(d_hidden)

        # feedforward
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(d_hidden, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, d_output)
        )

        # layer norm 2
        self.norm2 = torch.nn.LayerNorm(d_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer decoder block.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_input).

        Returns:
            y (torch.Tensor): The output tensor of shape (batch_size, seq_len, d_output).
        """

        # get shape
        batch_size, seq_len, features = x.shape

        # self-attention
        x1 = self.multihead_attn(x)

        # layer norm 1
        x2 = self.norm1(x1 + x)

        # feedforward
        x3 = self.feedforward(x2)
        y = self.norm2(x3 + x2)

        return y

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
    batch_size = 1
    sequence_length = 10
    vocab_size = 100
    num_heads = 1
    model_dim = num_heads * 64
    n_layers = 2
    dropout = 0.5

    # random input
    rand = torch.randint(0, vocab_size, (batch_size, sequence_length + 1))
    x = rand[:, :-1]
    y = rand[:, 1:]

    print(x)
    print(y)
    
    print(f"Random input: {x.shape}")

    # init model and forward pass
    model = TransformerDecoder(vocab_size=vocab_size, model_dim=model_dim, n_layers=n_layers, dropout=dropout, n_heads=num_heads, context_length=sequence_length)
    pred = model(x)

    print(f"Pred shape: {pred.shape}")
    print(f"Target shape: {y.shape}")

    # print loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(pred.view(-1, vocab_size), y.view(-1))
    print(f"Loss: {loss}")
