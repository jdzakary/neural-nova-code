import numpy as np
import torch
from torch import nn


def model_rand_1(
    obs: np.ndarray,
    mask: np.ndarray,
    history: np.ndarray
) -> np.ndarray:
    return np.argmax(mask, axis=1).flatten()


class BackboneV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, stride=3)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=512*9*2, out_features=4800),
            nn.ReLU(),
            nn.Linear(in_features=4800, out_features=1024),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_out_0 = self.flatten(self.cnn(observations[:, 0:1, :, :]))
        cnn_out_1 = self.flatten(self.cnn(observations[:, 1:2, :, :]))
        cnn_total = torch.cat((cnn_out_0, cnn_out_1), 1)
        return self.linear(cnn_total)


class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BackboneV2()
        self.actor_head = nn.Sequential(
            nn.Linear(in_features=1024, out_features=81),
        )

    def forward(
        self,
        boards: torch.Tensor,
    ) -> torch.Tensor:
        backbone = self.backbone(boards)
        return self.actor_head(backbone)


def load_saved_model(
    path: str,
    model_class: type[nn.Module]
) -> nn.Module:
    state_dict = torch.load(path)
    model = model_class()
    new_names = {x: x.replace('module.module.module.', '') for x in state_dict.keys()}
    for name, new_name in new_names.items():
        state_dict[new_name] = state_dict.pop(name)
    model.load_state_dict(state_dict)
    model.to(0)
    model.eval()
    return model


class ModelWrapper:
    def __init__(self, model: nn.Module, max_batch: int = 400):
        self.__model = model
        self.__max_batch = max_batch

    def run(
        self,
        boards: np.ndarray,
        mask: np.ndarray,
        history: np.ndarray,
    ) -> np.ndarray:
        batch_size = boards.shape[0]

        boards = self.create_old_boards(boards, history)
        boards = boards * -1
        final = np.zeros((batch_size,), dtype=np.int8)
        boards = boards.astype(np.float32)
        mask = mask.astype(np.float32)

        with torch.no_grad():
            e = 0
            for i in range(0, batch_size // self.__max_batch):
                s = i*self.__max_batch
                e = (i+1)*self.__max_batch
                t_boards = torch.from_numpy(boards[s:e]).to(0)
                t_mask = torch.from_numpy(mask[s:e]).to(0)
                logits = self.__model(t_boards)
                logits[t_mask == 0] = -torch.inf
                result =  logits.argmax(1).flatten()
                final[s:e] = result.to('cpu').numpy()

            if e < batch_size:
                t_boards = torch.from_numpy(boards[e:]).to(0)
                t_mask = torch.from_numpy(mask[e:]).to(0)
                logits = self.__model(t_boards)
                logits[t_mask == 0] = -torch.inf
                result = logits.argmax(1).flatten()
                final[e:] = result.to('cpu').numpy()

        return final

    @staticmethod
    def create_old_boards(boards: np.ndarray, history: np.ndarray) -> np.ndarray:
        move = np.argmax(history[0, :] == -1) - 1
        idx = history[:, move].flatten()
        idx_a, idx_b = np.unravel_index(idx, (9, 9))
        old = boards.copy()
        old[np.arange(boards.shape[0]), idx_a, idx_b] = 0
        boards = np.expand_dims(boards, 1)
        old = np.expand_dims(old, 1)
        return np.concatenate((boards, old), 1)
