from typing import Dict, Any

import torch
from torch import nn

from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule


class CustomTicTacToe(PPOTorchRLModule):
    def __init__(self, config: RLModuleConfig) -> None:
        super().__init__(config)

    def setup(self):
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config["fcnet_hiddens"][0]
        output_dim = self.action_space.n

        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.input_dim = input_dim
