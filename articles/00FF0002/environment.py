from typing import Any, SupportsFloat

import gym
import numpy as np
import onnxruntime as ort
from gym import spaces
from gym.core import ObsType, ActType

from game import Game


class MegaTicTacToe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, options: dict, render_mode=None):
        self.observation_space = spaces.Dict({
            'observations': spaces.Box(low=-1, high=1, shape=(3, 9, 9), dtype=np.float64),
            'action_mask': spaces.Box(low=0.0, high=1.0, shape=(81,), dtype=np.bool_)
        })
        self.action_space = spaces.Discrete(81)
        self.render_mode = render_mode
        self.__tie_penalty = options.get('tie_penalty', 0.25)
        self.__player = options.get('player', 1)
        if self.__player not in [1, -1]:
            raise ValueError('Player Must be 1 or -1 (X or O)')
        self.__session = ort.InferenceSession(
            path_or_bytes='../00FF0000/exports/model-X.onnx',
            providers=['CPUExecutionProvider']
        )


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.__game = Game()
        self.__obs = np.zeros((3, 9, 9), dtype=np.float64)
        if self.__player == -1:
            self.__enemy_move_2()
        return {
            'observations': np.expand_dims(self.__obs, 0),
            'action_mask': self.__game.constraint.flatten().astype(np.bool_),
        }, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.__game.move(*np.unravel_index(action, (9, 9)))
        self.__update_obs()
        if not self.__game.game_over:
            self.__enemy_move_2()
        reward = 0
        info = {}
        if self.__game.game_over:
            if self.__game.winner == self.__player:
                reward = 1
                info['outcome'] = 'win'
            elif self.__game.winner == 0:
                reward = self.__tie_penalty
                info['outcome'] = 'tie'
            else:
                reward = -1
                info['outcome'] = 'lose'
        return (
            {
                'observations': np.expand_dims(self.__obs, 0),
                'action_mask': self.__game.constraint.flatten().astype(np.bool_),
            },
            reward,
            self.__game.game_over,
            False,
            info
        )

    def __enemy_move_1(self) -> None:
        mask = self.__game.constraint.flatten()
        idx = np.argwhere(mask).flatten()
        # move = np.random.choice(idx)
        move = idx[0]
        self.__game.move(*np.unravel_index(move, (9, 9)))
        self.__update_obs()

    def __enemy_move_2(self) -> None:
        if self.__game.big_next:
            mask = np.zeros((10,), dtype=np.float32)
            mask[0:9] = (self.__game.big_remaining == 1).flatten()
            output = self.__session.run(None, {
                'obs': self.__game.big_board.astype(np.float32).flatten(),
                'mask': mask,
            })
            big_idx = int(output[0])
            r = big_idx // 3
            c = big_idx % 3
        else:
            r, c = self.__game.target_square

        mask = np.zeros((10,), dtype=np.float32)
        mask[0:9] = self.__game.constraint[r*3:r*3+3, c*3:c*3+3].flatten()
        output = self.__session.run(None, {
            'obs': self.__game.board[r*3:r*3+3, c*3:c*3+3].astype(np.float32).flatten(),
            'mask': mask,
        })
        idx = int(output[0])
        sr = idx // 3
        sc = idx % 3
        self.__game.move(sr+3*r, sc+3*c)
        self.__update_obs()

    def __enemy_move_3(self):
        result = self.__session.run(None, {
            'observations': -1 * self.__obs.astype(np.float32)
        })
        logits = result[0].flatten()
        logits[self.__game.constraint.flatten() == 0] = -np.inf
        idx = np.argmax(logits).flatten()[0]
        self.__game.move(*np.unravel_index(idx, (9, 9)))
        self.__update_obs()




    def __update_obs(self) -> None:
        self.__obs[1:, :, :] = self.__obs[0:2, :, :]
        self.__obs[0, :, :] = self.__game.board
