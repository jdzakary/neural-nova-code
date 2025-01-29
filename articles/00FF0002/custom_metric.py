from typing import Optional, Dict

import gymnasium as gym
from ray.rllib import BaseEnv, Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import PolicyID


class Outcomes(DefaultCallbacks):
    """
    Custom ray metric that enables us to track
    wins, loses, and ties during training.

    This is important because the tie penalty makes it difficult/impractical
    to deduce the ratio of WinX:WinO:Tie from the average episode reward.
    By reporting our own custom metric, Ray will automatically average this
    over the training episodes.
    """
    def on_episode_end(
        self,
        *,
        episode: MultiAgentEpisode,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        info_x = episode.agent_episodes['X'].get_infos(-1)
        info_o = episode.agent_episodes['O'].get_infos(-1)
        metrics_logger.log_value('Tie', info_x['outcome'] == 'tie')
        metrics_logger.log_value('WinX', info_x['outcome'] == 'win')
        metrics_logger.log_value('WinO', info_o['outcome'] == 'win')


class OutcomesSingle(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: SingleAgentEpisode,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        info = episode.get_infos(-1)
        metrics_logger.log_value('Win', info['outcome'] == 'win')
        metrics_logger.log_value('Lose', info['outcome'] == 'lose')
        metrics_logger.log_value('Tie', info['outcome'] == 'tie')
        metrics_logger.log_value('LoseInverse', info['outcome'] != 'lose')
