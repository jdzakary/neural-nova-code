from os import PathLike
from typing import TypeVar

import numpy as np
import torch
from ray.rllib.core.rl_module import RLModule
from torch import nn as nn

T = TypeVar('T', bound=nn.Module)


def create_torch_model(
    model_torch: T,
    state_folder: str | PathLike,
    replacement_map: dict[str, str],
) -> T:
    """
    Constructs a pytorch model using the weights of an RlModule checkpoint.

    :param model_torch:
        An instance of the pytorch model that will accept weights
        from the RlModule checkpoint. Must have appropriately named
        layers or else loading the state dict will fail
    :param state_folder:
        Path to the RlModule checkpoint
    :param replacement_map:
        A dictionary that maps layer names from RlLib to how they
        are names in your mock class
    :return:
        The pytorch model instance with weights loaded
    """
    model_rllib = RLModule.from_checkpoint(state_folder)
    state_dict = model_rllib.get_state(inference_only=True)

    new_names = {}
    for name in state_dict:
        for key, value in replacement_map.items():
            if key in name:
                new_names[name] = f'{value}{name[len(key):]}'

    for name, new_name in new_names.items():
        state_dict[new_name] = state_dict.pop(name)

    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    model_torch.load_state_dict(state_dict)
    return model_torch


def show_model_interior(state_folder: str | PathLike) -> None:
    """
    Prints details about an RlLib Module's forward method.
    This is useful when determining how to construct a custom
    Python Class for exporting the RLModule's weights to ONNX.

    :param state_folder:
        Absolute path to the policy folder. Should contain the
        pytorch state file "module_state.pt". This folder is
        usually located at "checkpoint/learner_group/learner/rl_module/policy"
    """
    module = RLModule.from_checkpoint(state_folder)
    state_dict = module.get_state(inference_only=True)
    print('Model State Dict:')
    for name in state_dict:
        data: np.ndarray = state_dict[name]
        print(f'{name}\t{data.shape}\t{type(data)}')
    # noinspection PyUnresolvedReferences
    print(module.forward)
