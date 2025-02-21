import torch
from torch import nn


class SharedActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(1, -1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(1024*4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(1024*4, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 81),
        )
        self.feature_norm = nn.BatchNorm1d(1024*4)



    def forward(self, observations: torch.Tensor):
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        features = self.backbone(observations)
        # features = self.feature_norm(features)
        logits: torch.Tensor = self.actor_head(features)
        values = self.value_head(features)
        return logits, values
