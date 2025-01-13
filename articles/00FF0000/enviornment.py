import functools
from typing import Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ActionType

from game import Game


class MultiAgentTicTacToe(ParallelEnv):
    metadata = {'render.modes': ['human'], "name": "TicTacToe"}

    def __init__(self, options: dict, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = ['X', 'O']
        self.__tie_penalty = options.get('tie_penalty', 0.25)
        self.__random_first = options.get('random_first', False)

    # noinspection PyTypeChecker
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gym.spaces:
        return spaces.Dict({
            'observations': spaces.Box(low=-1, high=1, shape=(9,), dtype=int),
            'action_mask': spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=int)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gym.spaces:
        return spaces.Discrete(10)

    # noinspection PyAttributeOutsideInit
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.__game = Game()
        if self.__random_first:
            self.__turn = 'O'
            idx = np.random.choice(np.arange(9))
            self.__game.move(*np.unravel_index(idx, (3, 3)))
        else:
            self.__turn = 'X'
        observations = {
            agent: {
                'observations': self.__game.board.flatten(),
                'action_mask': self.__create_mask(agent == self.__turn)
            } for agent in self.agents
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        rewards = {agent: 0.0 for agent in self.agents}
        if actions[self.__not_turn()] != 9:
            rewards[self.__not_turn()] = -0.1

        action = actions[self.__turn]
        if action is not None:
            self.__game.move(*np.unravel_index(action, (3, 3), order='C'))
        self.num_moves += 1

        # Update Observations
        observations = {
            agent: {
                'observations': self.__game.board.flatten(),
                'action_mask': self.__create_mask(agent != self.__turn)
            } for agent in self.agents
        }
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Check Reward conditions
        if self.__game.game_over:
            terminations['X'] = True
            terminations['O'] = True
            terminations['__all__'] = True
            if self.__game.winner_symbol == 'X':
                rewards['X'] = 1
                rewards['O'] = -1
                infos['X']['outcome'] = 'win'
                infos['O']['outcome'] = 'lose'
            elif self.__game.winner_symbol == 'O':
                rewards['X'] = -1
                rewards['O'] = 1
                infos['X']['outcome'] = 'lose'
                infos['O']['outcome'] = 'win'
            else:
                rewards['X'] = self.__tie_penalty
                rewards['O'] = self.__tie_penalty
                infos['X']['outcome'] = 'tie'
                infos['O']['outcome'] = 'tie'

        self.__switch_turn()
        return observations, rewards, terminations, truncations, infos

    def __create_mask(self, to_move_next: bool) -> np.ndarray:
        mask = np.ones((10,), dtype=np.int8)
        m1 = np.ones((3, 3), dtype=np.int8)
        occupied = self.__game.board != 0
        m1[occupied] = 0
        if to_move_next:
            mask[9] = 0
            mask[0:9] = m1.flatten()
        else:
            mask[0:9] = 0
        return mask

    def __switch_turn(self) -> None:
        self.__turn = 'X' if self.__turn == 'O' else 'O'

    def __not_turn(self) -> Literal['X', 'O']:
        return 'X' if self.__turn == 'O' else 'O'
