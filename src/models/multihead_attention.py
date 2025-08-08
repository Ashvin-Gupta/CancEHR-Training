import torch


class MultiHeadAttention(torch.nn.Module):
    """
    Implementation of multi-head attention (https://arxiv.org/abs/1706.03762)

    Args:
        d_input (int): The dimension of the input features.
        d_output (int): The dimension of the output features.
        n_heads (int): The number of attention heads.
        dropout (float): The dropout rate.
    """

    def __init__(self, d_input: int, d_output: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(d_input, d_output, dropout) for _ in range(n_heads)]
        )
        self.linear = torch.nn.Linear(d_output * n_heads, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shape
        batch_size, seq_len, features = x.shape

        # calculate output for each head
        outputs = [head(x) for head in self.heads]  # (batch_size, seq_len, d_output)

        # concatenate outputs
        output = torch.cat(outputs, dim=-1)  # (batch_size, seq_len, d_output * n_heads)

        # pass through linear layer
        output = self.linear(output)

        return output


class AttentionHead(torch.nn.Module):
    """
    Implementation of a single attention head (https://arxiv.org/abs/1706.03762)

    Args:
        d_input (int): The dimension of the input features.
        d_output (int): The dimension of the output features.
        dropout (float): The dropout rate.
    """

    def __init__(self, d_input: int, d_output: int, dropout: float) -> None:
        super().__init__()
        self.q = torch.nn.Linear(d_input, d_output)
        self.k = torch.nn.Linear(d_input, d_output)
        self.v = torch.nn.Linear(d_input, d_output)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shaoe
        batch_size, seq_len, features = x.shape

        # calculate queries and keys
        q = self.q(
            x
        )  # (batch_size, seq_len, d_input) @ (d_input, d_output) = (batch_size, seq_len, d_output)
        k = self.k(
            x
        )  # (batch_size, seq_len, d_input) @ (d_input, d_output) = (batch_size, seq_len, d_output)

        # calculate raw attention
        k_T = k.transpose(
            -2, -1
        )  # (batch_size, seq_len, d_output) -> (batch_size, d_output, seq_len)
        raw_attn = torch.matmul(
            q, k_T
        )  # (batch_size, seq_len, d_output) @ (batch_size, d_output, seq_len) = (batch_size, seq_len, seq_len)

        # scale raw attention to improve stability
        raw_attn *= k.shape[-1] ** -0.5  # divide by the square root of the number of features

        # create causal mask on the same device as the input
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask.masked_fill(mask == 0, float("-inf"))

        # apply mask
        masked_raw_attn = raw_attn.masked_fill(mask == float("-inf"), float("-inf"))

        # softmax
        attn = torch.softmax(masked_raw_attn, dim=-1)

        # dropout
        attn = self.dropout(attn)

        # calculate weighted output
        v = self.v(
            x
        )  # (batch_size, seq_len, d_input) @ (d_input, d_output) = (batch_size, seq_len, d_output)
        output = torch.matmul(
            attn, v
        )  # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_output) = (batch_size, seq_len, d_output)

        return output


if __name__ == "__main__":
    # create a multi-head attention
    multi_head_attention = MultiHeadAttention(d_input=5, d_output=10, n_heads=2, dropout=0.1)

    # create a random tensor
    x = torch.randn(1, 3, 5)

    print(x)

    # forward pass
    output = multi_head_attention(x)

    print(output.shape)
