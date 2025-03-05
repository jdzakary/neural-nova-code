import numpy as np
import numba as nb
import torch
from torch import nn


def transform_boards(boards: np.ndarray) -> np.ndarray:
    """
    Transforms Board from (B, 3, 3, 3, 3) to (B, 1, 9, 9)
    :param boards:
    :return:
    """
    x = np.zeros((boards.shape[0], 9, 9), dtype=np.float32)
    x[:, 0:3, 0:3] = boards[:, 0, 0]
    x[:, 3:6, 0:3] = boards[:, 1, 0]
    x[:, 6:9, 0:3] = boards[:, 2, 0]
    x[:, 0:3, 3:6] = boards[:, 0, 1]
    x[:, 3:6, 3:6] = boards[:, 1, 1]
    x[:, 6:9, 3:6] = boards[:, 2, 1]
    x[:, 0:3, 6:9] = boards[:, 0, 2]
    x[:, 3:6, 6:9] = boards[:, 1, 2]
    x[:, 6:9, 6:9] = boards[:, 2, 2]
    return np.expand_dims(x, 1)


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def create_old_boards(boards: np.ndarray, history: np.ndarray) -> np.ndarray:
    result = np.zeros(boards.shape,dtype=np.float32)
    for i in nb.prange(boards.shape[0]):
        a = history[i, -1][0]
        b = history[i, -1][1]
        c = history[i, -1][2]
        d = history[i, -1][3]
        result[i] = boards[i]
        result[i, a, b, c, d] = 0
    return result


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def create_masks(
    boards: np.ndarray,
    status: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    result = np.zeros((boards.shape[0], 3, 3, 3, 3), dtype=np.float32)
    for i in nb.prange(boards.shape[0]):
        a = history[i, -1][2]
        b = history[i, -1][3]
        x = np.zeros((3, 3, 3, 3))
        if status[i, a, b] == 0 or status[i, a, b] == 3:
            x[a, b, :, :] = boards[i, a, b, :, :] == 0
        else:
            for c in range(3):
                for d in range(3):
                    if status[i, c, d] == 0 or status[i, c, d] == 3:
                        x[c, d, :, :] = boards[i, c, d, :, :] == 0
        result[i] = x
    return result


@nb.jit(nopython=True, nogil=True, cache=True, parallel=True)
def model_v1(
    boards: np.ndarray,
    status: np.ndarray,
    history: np.ndarray,
) -> np.ndarray:
    result = np.zeros((boards.shape[0], 4), dtype=np.int8)
    for i in nb.prange(boards.shape[0]):
        a = history[i, -1][2]
        b = history[i, -1][3]
        x = np.zeros((3, 3, 3, 3), dtype=np.bool_)
        if status[i, a, b] == 0 or status[i, a, b] == 3:
            x[a, b, :, :] = boards[i, a, b, :, :] == 0
        else:
            for c in range(3):
                for d in range(3):
                    if status[i, c, d] == 0 or status[i, c, d] == 3:
                        x[c, d, :, :] = boards[i, c, d, :, :] == 0
        idx = np.argwhere(x)[0]
        result[i] = idx
    return result


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
        mask: torch.Tensor,
    ) -> torch.Tensor:
        backbone = self.backbone(boards)
        logits: torch.Tensor = self.actor_head(backbone)
        logits[mask == 0] = -torch.inf
        idx = logits.argmax(1)
        return torch.column_stack(torch.unravel_index(idx, (3, 3, 3, 3)))


class ModelV2Wrapper:
    def __init__(self):
        state_dict = torch.load('../results/state/exp7/batch_423_actor_o.pt')
        new_names = {x: x.replace('module.module.module.', '') for x in state_dict.keys()}
        for name, new_name in new_names.items():
            state_dict[new_name] = state_dict.pop(name)
        self.__model_v2 = ModelV2()
        self.__model_v2.load_state_dict(state_dict)
        self.__model_v2.to(0)
        self.__model_v2.eval()

    def run(
        self,
        boards: np.ndarray,
        status: np.ndarray,
        history: np.ndarray,
    ) -> np.ndarray:
        batch_size = boards.shape[0]

        old = create_old_boards(boards, history)
        mask = create_masks(boards, status, history)
        mask = transform_boards(mask).reshape(batch_size, 81)
        boards = transform_boards(boards)
        old = transform_boards(old)

        # Need To frame stack here....
        boards = np.concatenate((boards, old), 1)
        boards = boards * -1

        final = np.zeros((batch_size, 4), dtype=np.int8)

        e = 0
        for i in range(0, batch_size // 400):
            s = i*400
            e = (i+1)*400
            t_boards = torch.from_numpy(boards[s:e]).to(0)
            t_mask = torch.from_numpy(mask[s:e]).to(0)
            result = self.__model_v2(t_boards, t_mask)
            final[s:e, :] = result.to('cpu').numpy()

        if e < batch_size:
            t_boards = torch.from_numpy(boards[e:]).to(0)
            t_mask = torch.from_numpy(mask[e:]).to(0)
            result = self.__model_v2(t_boards, t_mask)
            final[e:, :] = result.to('cpu').numpy()

        return final
