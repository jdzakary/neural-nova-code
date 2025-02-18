import torch
from torch import nn


class MyPolicy(nn.Module):

    def __init__(self):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=3),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=3),
            nn.Flatten(0, -1),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 81),
        )


    def forward(self, observations: torch.Tensor, action_mask: torch.Tensor):
        logits: torch.Tensor = self.actor_net(observations)
        masked = torch.masked_fill(logits, action_mask == 0, -2e37)
        return masked
