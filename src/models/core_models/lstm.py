import torch
from src.models.base import BaseNightingaleModel
from src.models.registry import register_model

@register_model("lstm")
class LSTM(BaseNightingaleModel):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            n_layers: int = 2,
            dropout: float = 0.5,
        ):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def required_keys(self) -> set[str]:
        return {"ehr.input_token_ids"}

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the LSTM model.

        Args:
            x (dict): Input dictionary, with relevant keys:
                - ehr.input_token_ids (torch.Tensor): The input token sequence of shape (batch_size, sequence_length).

        Returns:
            y (torch.Tensor): The output logits of shape (batch_size, sequence_length, vocab_size). The logits are the
                unnormalized probabilities of the next token in the sequence.
        """

        input_token_ids = x["ehr"]["input_token_ids"]

        # embed token sequence
        embedded = self.embedding(input_token_ids)

        # pass through LSTM
        output, (hidden, cell) = self.lstm(embedded)

        # pass through linear layer
        y = self.fc(output)

        return y


if __name__ == "__main__":
    # define params
    batch_size = 3
    sequence_length = 10
    vocab_size = 100
    embedding_dim = 100
    hidden_dim = 100

    # random input
    x = {
        "ehr": {
            "input_token_ids": torch.randint(0, vocab_size, (batch_size, sequence_length))
        }
    }
    print(f"Random input: {x.shape}")

    # init model and forward pass
    model = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    y = model(x)
    print(f"Output: {y.shape}")