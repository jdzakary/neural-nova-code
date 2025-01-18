from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import ray

from ray import tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

from custom_metric import Outcomes
from enviornment import MegaTicTacToe
from shared.ray.progress_report import CustomReporter
from shared.ray.action_masking import ActionMaskingTorchRLModule

if TYPE_CHECKING:
    from ray.tune.experiment import Trial

PROJECT_PATH = Path(__file__).parent


def env_creator(env_config: dict):
    """
    Converts the PettingZoo Env Into an RlLib Environment
    """
    return ParallelPettingZooEnv(MegaTicTacToe(env_config))


def trail_dirname_creator(trial: Trial) -> str:
    """
    Used by RLLib to create trial directory name
    This prevents the annoying default behavior of super long directory names
    """
    return f'{trial.trial_id}'


def main():
    """
    Train agent X and agent O against one another. X will be forced
    to play a random first move, which helps O experience the entire
    observations space.

    Performs a hyperparameter search.
    Multiple trials execute in parallel and training progress is
    compared to prune under-performing trials.
    :return:
    """

    experiment_name = 'Experiment-1'
    max_time_total = 60 * 60 * 3
    grace_period_iter = 40

    '''
    Initialize the ray cluster. In this case, our "cluster" is just
    a single machine. If we had a multi-node cluster, we could
    use more advanced options discussed on Ray's website.
    '''
    ray.init()
    model_config = {
        'fcnet_activation': 'relu',
        'dim': 9,
        'conv_filters': [
            [64, [3, 3], 3],
            [256, [3, 3], 3],
        ]
    }

    # Register our Environment and Create the Config Object
    register_env('tic-tac-toe', env_creator)
    config = (
        PPOConfig()
        .api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True
        )
        .reporting(
            metrics_num_episodes_for_smoothing=400
        )
        .callbacks(
            callbacks_class=Outcomes
        )
        .environment(
            env='tic-tac-toe',
            env_config={
                'tie_penalty': -0.25,
                'random_first': True,
            }
        )
        .multi_agent(
            policies={'pX', 'pO'},
            policy_mapping_fn=(lambda aid, *args, **kwargs: f'p{aid}'),
            policies_to_train=['pO']
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    'pX': RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config_dict=model_config
                    ),
                    'pO': RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        model_config_dict=model_config
                    )
                }
            )
        )
        .training(
            lr=tune.loguniform(0.00005, 0.001),
            gamma=tune.uniform(0.8, 0.99),
            use_kl_loss=False,
            entropy_coeff=tune.uniform(0, 0.2),
            # train_batch_size_per_learner=10_000,
            # mini_batch_size_per_learner=1000,
            num_sgd_iter=5,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0.25,
        )
        .env_runners(
            num_env_runners=3,
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
            'perf/gpu_util_percent0': 'GPU',
        },
        max_report_frequency=10,
        metric='WinO',
        mode='max',
        time_col='Seconds',
        rounding={
            'Seconds': 0,
            'Tie': 3,
            'WinX': 3,
            'WinO': 3,
            'GPU': 3,
        }
    )

    # Create Checkpoint Config
    config_checkpoint = train.CheckpointConfig(
        checkpoint_at_end=True,
        num_to_keep=10,
        checkpoint_frequency=20,
        checkpoint_score_order='max',
        checkpoint_score_attribute='env_runners/WinO',
    )

    # Create Tuner Config
    config_tuner = tune.TuneConfig(
        metric='env_runners/WinO',
        mode='max',
        trial_dirname_creator=trail_dirname_creator,
        # max_concurrent_trials=1,
        search_alg=ConcurrencyLimiter(
            searcher=HyperOptSearch(),
            max_concurrent=4,
        ),
        scheduler=ASHAScheduler(
            time_attr="training_iterations",
            grace_period=grace_period_iter,
        ),
        num_samples=80,
        time_budget_s=max_time_total,
    )

    # Create Tuner Object
    os.environ['RAY_AIR_NEW_OUTPUT'] = '0'
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=train.RunConfig(
            name=experiment_name,
            stop={
                'env_runners/WinO': 0.95,
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
    main()
