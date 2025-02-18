from typing import Any, SupportsFloat

import gym
import numpy as np
from gym import spaces
from gym.core import ObsType, ActType

from game import Game


class MegaTicTacToe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, options: dict, render_mode=None):
        self.observation_space = spaces.Dict({
            'observations': spaces.Box(low=-1, high=1, shape=(1, 9, 9), dtype=np.float64),
            'action_mask': spaces.Box(low=0.0, high=1.0, shape=(81,), dtype=np.float64)
        })
        self.action_space = spaces.Discrete(81)
        self.render_mode = render_mode
        self.__tie_penalty = options.get('tie_penalty', 0.25)
        self.__player = options.get('player', 1)
        if self.__player not in [1, -1]:
            raise ValueError('Player Must be 1 or -1 (X or O)')

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.__game = Game()
        if self.__player == -1:
            self.__enemy_move()
        return {
            'observations': np.expand_dims(self.__game.board, 0),
            'action_mask': self.__game.constraint.flatten(),
        }, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.__game.move(*np.unravel_index(action, (9, 9)))
        if not self.__game.game_over:
            self.__enemy_move()
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
                'observations': np.expand_dims(self.__game.board, 0),
                'action_mask': self.__game.constraint.flatten(),
            },
            reward,
            self.__game.game_over,
            False,
            info
        )

    def __enemy_move(self) -> None:
        mask = self.__game.constraint.flatten()
        idx = np.argwhere(mask).flatten()
        move = np.random.choice(idx)
        self.__game.move(*np.unravel_index(move, (9, 9)))
