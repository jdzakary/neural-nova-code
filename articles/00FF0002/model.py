import torch
from torch import nn


class SharedActorCritic(nn.Module):

    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512
    ):
        super().__init__()
        # Embedding layer: 3 channels (player 1, player 2, empty/active) → d_model
        self.embedding = nn.Linear(3, d_model)

        # Positional encoding for 9x9 grid
        self.pos_encoding = self._generate_positional_encoding(81, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Value head: Aggregate transformer output → single value
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Actor head: Per-position logits for 81 actions
        self.actor_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    @staticmethod
    def _generate_positional_encoding(seq_len, d_model):
        # Simple 2D positional encoding for 9x9 grid
        pe = torch.zeros(seq_len, d_model)
        for pos in range(seq_len):
            x, y = divmod(pos, 9)  # Convert 0-80 to (row, col)
            for i in range(0, d_model, 2):
                pe[pos, i] = torch.sin(torch.tensor(x / (10000 ** (i / d_model))))
                if i + 1 < d_model:
                    pe[pos, i + 1] = torch.cos(torch.tensor(y / (10000 ** ((i + 1) / d_model))))
        return pe.unsqueeze(0)  # Shape: [1, 81, d_model]


    def forward(self, observations: torch.Tensor):
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)

        # Input x: [batch_size, 3, 9, 9]
        batch_size = observations.size(0)

        # Reshape to sequence: [batch_size, 81, 3]
        x = observations.view(batch_size, 3, 81).transpose(1, 2)  # [batch_size, 81, 3]

        # Embed: [batch_size, 81, d_model]
        x = self.embedding(x)

        # Add positional encoding
        pos_encoding = self.pos_encoding.to(x.device)
        x = x + pos_encoding.expand(batch_size, -1, -1)

        # Transformer: [batch_size, 81, d_model]
        transformer_out = self.transformer(x)

        # Value: Mean pool across sequence → [batch_size, d_model] → [batch_size, 1]
        value_features = transformer_out.mean(dim=1)
        value = self.value_head(value_features)

        # Actor: Per-position logits → [batch_size, 81]
        logits = self.actor_head(transformer_out)
        logits = logits.view(batch_size, 81)

        return logits, value
