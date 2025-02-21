import torch
from torch import nn


class SharedActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, stride=3),
            nn.LeakyReLU(),
            nn.Flatten(1, -1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 81),
        )



    def forward(self, observations: torch.Tensor):
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        features = self.backbone(observations)
        logits: torch.Tensor = self.actor_head(features)
        values = self.value_head(features)
        return logits, values
