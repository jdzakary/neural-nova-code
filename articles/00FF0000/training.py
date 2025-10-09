from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import ray

from ray import tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

from custom_metric import Outcomes
from environment import MultiAgentTicTacToe
from shared.ray.progress_report import CustomReporter
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
#from shared.ray.action_masking import ActionMaskingTorchRLModule

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

PROJECT_PATH = Path(__file__).parent


def env_creator(env_config: dict):
    """
    Converts the PettingZoo Env Into an RlLib Environment
    """
    return ParallelPettingZooEnv(MultiAgentTicTacToe(env_config))


def trail_dirname_creator(trial: Trial) -> str:
    """
    Used by RLLib to create trial directory name
    This prevents the annoying default behavior of super long directory names
    """
    return f'{trial.trial_id}'


def main(args):
    """
    Train agent X and agent O against one another. X will be forced
    to play a random first move, which helps O experience the entire
    observations space.

    Performs a hyperparameter search.
    Multiple trials execute in parallel and training progress is
    compared to prune under-performing trials.
    :return:
    """

    experiment_name = f'Settings: {int(100*args.x_win_reward)} {int(100*args.o_win_reward)} {int(100*args.x_lose_reward)} {int(100*args.o_lose_reward)} {int(100*args.stop_tie_threshold)}'
    max_iter_individual = 300
    max_time_total = 60 * 90
    grace_period_iter = 15

    '''
    Initialize the ray cluster. In this case, our "cluster" is just
    a single machine. If we had a multi-node cluster, we could
    use more advanced options discussed on Ray's website.
    '''
    ray.init()

    # Register our Environment and Create the Config Object
    register_env('tic-tac-toe', env_creator)
    config = (
        PPOConfig()
        #.api_stack(
        #    enable_env_runner_and_connector_v2=True,
        #    enable_rl_module_and_learner=True
        #)
        .reporting(
            metrics_num_episodes_for_smoothing=1000
        )
        .callbacks(
            callbacks_class=Outcomes
        )
        .environment(
            env='tic-tac-toe',
            env_config={
                'x_tie_penalty': tune.uniform(args.x_lose_reward, args.x_win_reward),
                'o_tie_penalty': tune.uniform(args.o_lose_reward, args.o_win_reward),
                'random_first': True,
                'x_win_reward': args.x_win_reward,
                'x_lose_reward': args.x_lose_reward,
                'o_win_reward': args.o_win_reward,
                'o_lose_reward': args.o_lose_reward,
            }
        )
        .multi_agent(
            policies={'pX', 'pO'},
            policy_mapping_fn=(lambda aid, *args, **kwargs: f'p{aid}')
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    'pX': RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config=DefaultModelConfig(fcnet_activation='relu'),
                    ),
                    'pO': RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config=DefaultModelConfig(fcnet_activation='relu'),
                    )
                }
            )
        )
        .training(
            lr=tune.loguniform(1e-5, 1e-2),
            gamma=tune.uniform(0.80, 0.99),
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0.2,
        )
        .env_runners(
            num_env_runners=2,
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=1,
        )
    )

    # Create Custom Progress Reporter
    reporter = CustomReporter(
        metric_columns={
            'time_total_s': 'Seconds',
            'env_runners/Tie': 'Tie',
            'env_runners/WinX': 'WinX',
            'env_runners/WinO': 'WinO',
            'training_iteration': 'Iters',
        },
        max_report_frequency=10,
        metric='Tie', # The metric to sort tables by
        mode='max',
        time_col='Seconds',
        rounding={
            'Seconds': 0,
            'Tie': 3,
            'WinX': 3,
            'WinO': 3,
        }
    )

    # Create Checkpoint Config
    config_checkpoint = train.CheckpointConfig(
        checkpoint_at_end=True,
        num_to_keep=10,
        checkpoint_frequency=20,
        checkpoint_score_order='max',
        checkpoint_score_attribute='env_runners/Tie',
    )

    # Create Tuner Config
    config_tuner = tune.TuneConfig(
        metric='env_runners/Tie', # Metric to optimize. Pick settings that yield lots of ties.
        mode='max',
        trial_dirname_creator=trail_dirname_creator,
        search_alg=ConcurrencyLimiter(
            searcher=HyperOptSearch(),
            max_concurrent=5,
        ),
        scheduler=ASHAScheduler(
            time_attr="training_iteration",
            grace_period=grace_period_iter, # Don't stop trials before this point
            max_t=max_iter_individual,      # Stop trials after this point, no matter what
        ),
        num_samples=-1,
        time_budget_s=max_time_total,       # Timeout for all trials
    )

    # Create Tuner Object
    os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            name=experiment_name,
            stop={
                'env_runners/Tie': args.stop_tie_threshold, # No non-ties within the window
            },
            storage_path=str(PROJECT_PATH / 'results'),
            checkpoint_config=config_checkpoint,
            progress_reporter=reporter,
            verbose=1,
        ),
        tune_config=config_tuner
    )

    # Start Training
    tuner.fit()


if __name__ == "__main__":
    # python training.py --x-lose-reward -.1 --o-win-reward .1 --stop-tie-threshold 1.0
    # Could try asymmetric tie penalties if this doesn't work.
    parser = argparse.ArgumentParser()
    parser.add_argument("--x-win-reward", type=float, default=1)
    parser.add_argument("--x-lose-reward", type=float, default=-1)
    parser.add_argument("--o-win-reward", type=float, default=1)
    parser.add_argument("--o-lose-reward", type=float, default=-1)
    parser.add_argument("--stop-tie-threshold", type=float, default=0.99)
    args = parser.parse_args()
    main(args)
