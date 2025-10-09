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
        self.__random_first = options.get('random_first', False)
        self.__x_tie_penalty = options.get('x_tie_penalty', 0.25) # Now a share of win reward
        self.__x_win_reward = options.get('x_win_reward', 1)
        self.__x_lose_reward = options.get('x_lose_reward', -1)
        self.__o_tie_penalty = options.get('o_tie_penalty', 0.25) # Now a share of win reward
        self.__o_win_reward = options.get('o_win_reward', 1)
        self.__o_lose_reward = options.get('o_lose_reward', -1)

    # noinspection PyTypeChecker
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gym.spaces:
        return spaces.Dict({
            'observations': spaces.Box(low=-1., high=1., shape=(9,)),
            'action_mask': spaces.Box(low=0.0, high=1.0, shape=(10,))
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gym.spaces:
        return spaces.Discrete(10)

    # noinspection PyAttributeOutsideInit
    def reset(
        self,
        seed = None,
        options = None,
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
        try:
            rewards = {agent: 0.0 for agent in self.agents}
            if actions[self.__not_turn()] != 9:
                rewards[self.__not_turn()] = -0.1

            action = actions[self.__turn]
            if action is not None and action != 9: # env_checker might sample a 9
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
                    rewards['X'] = self.__x_win_reward
                    rewards['O'] = self.__o_lose_reward
                    infos['X']['outcome'] = 'win'
                    infos['O']['outcome'] = 'lose'
                elif self.__game.winner_symbol == 'O':
                    rewards['X'] = self.__x_lose_reward
                    rewards['O'] = self.__o_win_reward
                    infos['X']['outcome'] = 'lose'
                    infos['O']['outcome'] = 'win'
                else:
                    rewards['X'] = self.__x_tie_penalty
                    rewards['O'] = self.__o_tie_penalty
                    infos['X']['outcome'] = 'tie'
                    infos['O']['outcome'] = 'tie'
            self.__switch_turn()
            return observations, rewards, terminations, truncations, infos
        except Exception() as e:
            for i in range(50):
                print("!")
            print(e)
            raise Exception()

    def __create_mask(self, to_move_next: bool) -> np.ndarray:
        mask = np.ones((10,))
        m1 = np.ones((3, 3))
        occupied = self.__game.board != 0.
        m1[occupied] = 0
        if to_move_next:
            mask[9] = 0.
            mask[0:9] = m1.flatten()
        else:
            mask[0:9] = 0.
        return mask

    def __switch_turn(self) -> None:
        self.__turn = 'X' if self.__turn == 'O' else 'O'

    def __not_turn(self) -> Literal['X', 'O']:
        return 'X' if self.__turn == 'O' else 'O'
