import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=3)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=512*9*2, out_features=2400),
            nn.ReLU(),
            nn.Linear(in_features=2400, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_out_0 = self.flatten(self.cnn(observations[:, 0:1, :, :]))
        cnn_out_1 = self.flatten(self.cnn(observations[:, 1:2, :, :]))
        cnn_total = torch.cat((cnn_out_0, cnn_out_1), 1)
        return self.linear(cnn_total)


class SharedActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.value_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=81),
        )

    def forward(self, observations: torch.Tensor):
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        backbone_out = self.backbone(observations)
        values = self.value_head(backbone_out)
        logits = self.actor_head(backbone_out)
        return logits, values


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.actor_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=81),
        )

    def forward(self, obs_x: torch.Tensor, obs_o: torch.Tensor):
        if len(obs_x.shape) == 5:
            obs_x = obs_x.squeeze(1)
            obs_o = obs_o.squeeze(1)
        backbone_x = self.backbone(obs_x)
        backbone_o = self.backbone(obs_o)
        logits_x = self.actor_head(backbone_x)
        logits_o = self.actor_head(backbone_o)
        return logits_x, logits_o


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.value_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )

    def forward(self, observations: torch.Tensor):
        if len(observations.shape) == 3:
            observations = observations.unsqueeze(0)
        backbone_out = self.backbone(observations)
        values = self.value_head(backbone_out)
        return values
