import torch
from src.models.base import BaseNightingaleModel
from src.models.registry import register_model
from transformers import AutoModel

@register_model("lstm_note")
class LSTMNote(BaseNightingaleModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)

        # embedding layer
        self.embedding = torch.nn.Embedding(model_config["vocab_size"], model_config["embedding_dim"])

        # LSTM layers
        self.lstm = torch.nn.LSTM(
            model_config["embedding_dim"], model_config["hidden_dim"], model_config["n_layers"], dropout=model_config["dropout"], batch_first=True
        )

        # ClinicalBert
        self.clinical_bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        # freeze ClinicalBERT weights
        for param in self.clinical_bert.parameters():
            param.requires_grad = False
        
        # fusion layer to combine token embeddings with clinical notes
        self.embedding_fusion_layer = torch.nn.Linear(
            model_config["embedding_dim"] + self.clinical_bert.config.hidden_size, 
            model_config["embedding_dim"]
        )

        # output layer
        self.fc = torch.nn.Linear(model_config["hidden_dim"], model_config["vocab_size"])

    def required_config_keys(self) -> set[str]:
        """
        Returns the required keys for the model configuration.
        """
        return {"vocab_size", "embedding_dim", "hidden_dim", "n_layers", "dropout"}

    def required_input_keys(self) -> set[str]:
        """
        Returns the required keys for the model input.
        """
        return {"ehr.input_token_ids", "clinical_notes.token_ids", "clinical_notes.notes_mask", "clinical_notes.tokens_mask"}

    def forward(self, x: dict) -> torch.Tensor:
        """
        Forward pass of the LSTM model with clinical notes fusion.

        Args:
            x (dict): Input dictionary, with relevant keys:
                - ehr.input_token_ids (torch.Tensor): The input token sequence of shape (batch_size, sequence_length).
                - clinical_notes.token_ids (torch.Tensor): The clinical notes token sequence of shape (batch_size, max_note_count, max_tokens_per_note).
                - clinical_notes.notes_mask (torch.Tensor): The clinical notes mask of shape (batch_size, max_note_count).
                - clinical_notes.tokens_mask (torch.Tensor): The clinical notes tokens mask of shape (batch_size, max_note_count, max_tokens_per_note).

        Returns:
            y (torch.Tensor): The output logits of shape (batch_size, sequence_length, vocab_size). The logits are the
                unnormalized probabilities of the next token in the sequence.
        """

        # validate input
        self.validate_input(x)

        ehr_token_ids = x["ehr"]["input_token_ids"] # (batch_size, sequence_length)
        clinical_notes_token_ids = x["clinical_notes"]["token_ids"] # (batch_size, max_note_count, max_tokens_per_note)
        clinical_notes_mask = x["clinical_notes"]["notes_mask"] # (batch_size, max_note_count)
        clinical_notes_tokens_mask = x["clinical_notes"]["tokens_mask"] # (batch_size, max_note_count, max_tokens_per_note)

        # process clinical notes first
        # unstack clinical notes
        unstacked_note_mask = clinical_notes_mask.view(-1) # (batch_size * max_note_count)
        unstacked_note_tokens = clinical_notes_token_ids.view(-1, clinical_notes_token_ids.shape[-1]) # (batch_size * max_note_count, max_tokens_per_note)
        unstacked_note_tokens_mask = clinical_notes_tokens_mask.view(-1, clinical_notes_tokens_mask.shape[-1]) # (batch_size * max_note_count, max_tokens_per_note)

        # only select notes that are not masked
        valid_note_indices = unstacked_note_mask.nonzero(as_tuple=True)[0]  # indices of valid notes
        if len(valid_note_indices) > 0:
            valid_note_tokens = unstacked_note_tokens[valid_note_indices]  # (num_valid_notes, max_tokens_per_note)
            valid_note_tokens_mask = unstacked_note_tokens_mask[valid_note_indices]  # (num_valid_notes, max_tokens_per_note)
            
            # pass through ClinicalBERT
            clinical_bert_output = self.clinical_bert(valid_note_tokens, attention_mask=valid_note_tokens_mask)
            note_embeddings = clinical_bert_output.last_hidden_state[:, 0, :]  # Use [CLS] token, shape: (num_valid_notes, bert_hidden_dim)
            
            # calculate mean embedding of notes for each batch
            batch_size, max_note_count = clinical_notes_mask.shape
            bert_hidden_dim = note_embeddings.shape[-1]
            
            # create tensor to hold mean embeddings for each batch
            batch_note_embeddings = torch.zeros(batch_size, bert_hidden_dim, device=note_embeddings.device)
            
            # map valid note indices back to batch indices
            batch_indices = valid_note_indices // max_note_count  # which batch each valid note belongs to
            
            # calculate mean embedding for each batch
            for batch_idx in range(batch_size):
                batch_note_mask = (batch_indices == batch_idx)
                if batch_note_mask.any():
                    batch_note_embeddings[batch_idx] = note_embeddings[batch_note_mask].mean(dim=0)
                # if no valid notes for this batch, embedding remains zeros
        else:
            # if no valid notes, create zero embeddings
            batch_size = clinical_notes_mask.shape[0]
            bert_hidden_dim = self.clinical_bert.config.hidden_size
            batch_note_embeddings = torch.zeros(batch_size, bert_hidden_dim, device=clinical_notes_mask.device)

        # embed token sequence
        embedded = self.embedding(ehr_token_ids)  # (batch_size, sequence_length, embedding_dim)
        
        # expand clinical notes embeddings to match sequence length for fusion
        sequence_length = embedded.shape[1]
        expanded_note_embeddings = batch_note_embeddings.unsqueeze(1).expand(-1, sequence_length, -1)  # (batch_size, sequence_length, bert_hidden_dim)
        
        # concatenate token embeddings with clinical notes embeddings
        fused_embeddings_input = torch.cat([embedded, expanded_note_embeddings], dim=-1)  # (batch_size, sequence_length, embedding_dim + bert_hidden_dim)
        
        # pass through embedding fusion layer
        fused_embeddings = self.embedding_fusion_layer(fused_embeddings_input)  # (batch_size, sequence_length, embedding_dim)

        # pass through LSTM with fused embeddings
        lstm_output, (hidden, cell) = self.lstm(fused_embeddings)  # (batch_size, sequence_length, hidden_dim)
        
        # pass through linear layer
        y = self.fc(lstm_output)

        return y


if __name__ == "__main__":
    # define params
    batch_size = 3
    sequence_length = 10
    max_note_count = 3
    max_tokens_per_note = 256
    vocab_size = 100
    embedding_dim = 100
    hidden_dim = 100

    # random input
    x = {
        "ehr": {
            "input_token_ids": torch.randint(0, vocab_size, (batch_size, sequence_length))
        },
        "clinical_notes": {
            "token_ids": torch.randint(0, vocab_size, (batch_size, max_note_count, max_tokens_per_note)),
            "notes_mask": torch.randint(0, 2, (batch_size, max_note_count)),
            "tokens_mask": torch.randint(0, 2, (batch_size, max_note_count, max_tokens_per_note)),
        }
    }

    print(f"ehr_token_ids shape: {x['ehr']['input_token_ids'].shape}")
    print(f"clinical_notes_token_ids shape: {x['clinical_notes']['token_ids'].shape}")
    print(f"clinical_notes_notes_mask shape: {x['clinical_notes']['notes_mask'].shape}")
    print(f"clinical_notes_tokens_mask shape: {x['clinical_notes']['tokens_mask'].shape}")

    # init model and forward pass
    model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_layers": 2,
        "dropout": 0.5,
    }
    model = LSTMNote(model_config)
    y = model(x)
    print(f"Output shape: {y.shape}")