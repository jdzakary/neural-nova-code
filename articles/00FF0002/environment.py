import functools
from typing import Any, SupportsFloat

import gym
import gymnasium
import numpy as np
import onnxruntime as ort
from gym import spaces
from gym.core import ObsType, ActType
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AgentID, ActionType

from game import Game


class MegaTicTacToe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, options: dict, render_mode=None):
        # Misic
        self.render_mode = render_mode

        # Define Observations and Action Spaces
        self.observation_space = spaces.Dict({
            'observations': spaces.Box(low=-1, high=1, shape=(3, 9, 9), dtype=np.float64),
            'action_mask': spaces.Box(low=0.0, high=1.0, shape=(81,), dtype=np.bool_)
        })
        self.action_space = spaces.Discrete(81)

        # Env Flexible Config
        self.__tie_reward = options.get('tie_reward', 0.25)
        self.__reward = options.get('reward', 1.0)
        self.__player = options.get('player', 1)
        self.__opponent_mode = options.get('opponent_mode', 1)
        model_file = options.get('model_file', '')

        if self.__player not in [1, -1]:
            raise ValueError('Player Must be 1 or -1 (X or O)')

        if self.__opponent_mode == 1:
            self.__enemy_move = self.__enemy_move_1
        elif self.__opponent_mode == 2:
            self.__enemy_move = self.__enemy_move_2
            self.__session = ort.InferenceSession(
                path_or_bytes='../00FF0000/exports/model-X.onnx',
                providers=['CPUExecutionProvider']
            )
        elif self.__opponent_mode == 3:
            self.__enemy_move = self.__enemy_move_3
            self.__session = ort.InferenceSession(
                path_or_bytes=model_file,
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
            self.__enemy_move()
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
            self.__enemy_move()
        reward = 0
        info = {}
        if self.__game.game_over:
            if self.__game.winner == self.__player:
                reward = self.__reward
                info['outcome'] = 'win'
            elif self.__game.winner == 0:
                reward = self.__tie_reward
                info['outcome'] = 'tie'
            else:
                reward = -1 * self.__reward
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


class MultiAgent(AECEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, options: dict, render_mode=None):
        super().__init__()
        # Misic
        self.render_mode = render_mode
        self.possible_agents = ['X', 'O']

        # Env Flexible Config
        self.__tie_reward = options.get('tie_reward', 0.25)
        self.__reward = options.get('reward', 1.0)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Dict({
            'observations': gymnasium.spaces.Box(low=-1, high=1, shape=(2, 9, 9), dtype=np.float64),
            'action_mask': gymnasium.spaces.Box(low=0.0, high=1.0, shape=(81,), dtype=np.bool_)
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(81)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        self.__game = Game()
        self.__obs = np.zeros((2, 9, 9), dtype=np.float64)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: ActionType) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.__game.move(*np.unravel_index(action, (9, 9)))
        self.__update_obs()

        if self.__game.game_over:
            if self.__game.winner == 1:
                self.rewards['X'] = self.__reward
                self.rewards['O'] = - self.__reward
            elif self.__game.winner == -1:
                self.rewards['O'] = self.__reward
                self.rewards['X'] = - self.__reward
            else:
                self.rewards['X'] = self.__tie_reward
                self.rewards['O'] = self.__tie_reward
            self.terminations['X'] = True
            self.terminations['O'] = True

    def __update_obs(self) -> None:
        self.__obs[1, :, :] = self.__obs[0, :, :]
        self.__obs[0, :, :] = self.__game.board

    def close(self):
        pass

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = np.copy(self.__obs)
        mask = self.__game.constraint.flatten().astype(np.bool_)
        if agent == 'O':
            obs = -1 * obs
        return {
            'observations': obs,
            'action_mask': mask
        }
