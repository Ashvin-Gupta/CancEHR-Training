import torch

class LSTM(torch.nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        # embed token sequence
        embedded = self.embedding(x)

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
    x = torch.randint(0, vocab_size, (batch_size, sequence_length))
    print(f"Random input: {x.shape}")

    # init model and forward pass
    model = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    y = model(x)
    print(f"Output: {y.shape}")