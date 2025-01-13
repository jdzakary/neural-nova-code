from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from ray.rllib.utils.torch_utils import FLOAT_MIN

from shared.ray.model_export import create_torch_model

PROJECT_PATH = Path(__file__).parent


class ModelExport(nn.Module):
    """
    Class for exporting the TicTacToe model.

    Reconstructs the forward method used by the RlModule,
    allowing the weights to be loaded in appropriately.
    """
    def __init__(self, sample: bool = False):
        """
        :param sample:
            Should the model sample the action probabilities
            instead of always taking the arg_max?
        """
        super().__init__()
        self.actor_encoder = nn.Sequential(
            nn.Linear(in_features=9, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.ReLU(),
        )
        self.pi = nn.Sequential(
            nn.Linear(in_features=256, out_features=10, bias=True),
        )
        self.softmax = nn.Softmax(dim=0)
        self.__sample = sample

    def forward(self, x, y):
        x = self.actor_encoder(x)
        logit = self.pi(x)
        logit[y == 0] = FLOAT_MIN
        weights: torch.Tensor = self.softmax(logit)
        if self.__sample:
            weights = weights.reshape((1, 10))
            idx = torch.multinomial(weights, 1, False)[0][0]
        else:
            idx = torch.argmax(weights)
        return idx


def construct_state_path(
    checkpoint_dir: str,
    policy_name: Literal['pO', 'pX', 'default_policy']
) -> Path:
    """
    Construct the absolute path to the checkpoint folder where
    the model state is saved.

    :param checkpoint_dir:
        The checkpoint dir relative to the results dir. Should be in
        the format of "experiment/trail/checkpoint"
    :param policy_name:
        Name of the policy. Either "pO" or "pX" for multi-agent.
        "default_policy" for single agent.
    :return:
        Path to the state_folder
    """
    return PROJECT_PATH/'results'/checkpoint_dir/'learner_group/learner/rl_module'/policy_name


def onnx_export(model_torch: ModelExport, save_name: str) -> None:
    """
    Export the Model to ONNX format

    :param model_torch:
        Pytorch model with weights loaded from RlLib checkpoint
    :param save_name:
        Path to save location. Should end with ".onnx" by convention
    """
    model_torch.eval()
    fake_obs = torch.randn(9)
    fake_mask = torch.randn(10)
    torch.onnx.export(
        model=model_torch,
        args=(fake_obs, fake_mask),
        f=save_name,
        input_names=['obs', 'mask'],
        output_names=['action'],
    )


def main():
    folder = construct_state_path(
        checkpoint_dir='Random-First-Move/d0b80b71/checkpoint_000011',
        policy_name='pX'
    )
    replacement_map = {
        'encoder.actor_encoder.net.mlp': 'actor_encoder',
        'pi.net.mlp': 'pi',
    }
    model = create_torch_model(ModelExport(), folder, replacement_map)
    onnx_export(model, 'exports/model-X.onnx')


if __name__ == '__main__':
    main()
