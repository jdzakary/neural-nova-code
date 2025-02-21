import multiprocessing
import warnings
from collections import OrderedDict

import gym
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torchrl.modules import MaskedOneHotCategorical
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger
from tqdm import tqdm

from model import SharedActorCritic

warnings.filterwarnings(action='ignore')


def create_env() -> TransformedEnv:
    device = 'cpu'

    gym.register(
        id='MegaTicTacToe-v0',
        entry_point='environment:MegaTicTacToe',
    )

    base_env = GymEnv(
        env_name='MegaTicTacToe-v0',
        options={
            'tie_penalty': -1.25,
            'player': -1,
        },
        device=device
    )
    return TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        )
    )


def main():
    #--- Config ---#
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    lr = 0.0008
    max_grad_norm = 1.0
    frames_per_batch = 5_000
    total_frames = 5_000_000
    num_envs = 4
    num_epochs = 3
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-3
    exp_name = 'exp1'

    #--- Policy ---#
    shared_actor_critic = SharedActorCritic()
    shared_actor_critic.to(device=device)
    policy_module = TensorDictModule(
        module=shared_actor_critic,
        in_keys=['observations'],
        out_keys=['logits', 'state_value'],
    )

    dist = ProbabilisticTensorDictModule(
        in_keys={'logits': 'logits', 'mask': 'action_mask'},
        out_keys=['action'],
        distribution_class=MaskedOneHotCategorical,
        return_log_prob=True,
        log_prob_key='action_log_prob',
    )

    actor = ProbabilisticTensorDictSequential(
        OrderedDict({'module': policy_module, 'dist': dist})
    )

    collector = MultiSyncDataCollector(
        create_env_fn=[create_env for _ in range(num_envs)],
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device='cpu',
        update_at_each_batch=True,
        cat_results=0
    )

    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=policy_module,
        average_gae=True,
        skip_existing=True
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=policy_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=0.8,
        loss_critic_type="l2",
        separate_losses=True,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=total_frames // frames_per_batch,
        eta_min=lr / 2
    )


    logger = CSVLogger(exp_name, 'results/logs')
    pbar = tqdm(total=total_frames // frames_per_batch)

    # Collect Data
    for i, tensordict_data in enumerate(collector):
        tensordict_data: TensorDict
        gpu_dict = tensordict_data.to(device=device)
        next_obs = gpu_dict['next']['observations']
        logits, state_value = shared_actor_critic(next_obs)
        gpu_dict['next'].update({'logits': logits, 'state_value': state_value.unsqueeze(-1)})

        # Train on batch "num_epochs" times
        for _ in range(num_epochs):
            data = advantage_module(gpu_dict)
            loss_vals = loss_module(data)
            loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

        # After training, log diagnostics
        episodic = tensordict_data['next', 'reward'][tensordict_data['next', 'reward'] != 0]
        logger.log_scalar('reward_mean', episodic.mean().item(), i)
        logger.log_scalar('step_count_max', tensordict_data["step_count"].max().item(), i)
        logger.log_scalar('lr', optim.param_groups[0]['lr'], i)
        logger.log_scalar('loss_objective', loss_vals["loss_objective"].item(), i)
        logger.log_scalar('loss_critic', loss_vals["loss_critic"].item(), i)
        logger.log_scalar('loss_entropy', loss_vals["loss_entropy"].item(), i)
        pbar.update()
        scheduler.step()

        if (i+1) % 50 == 0:
            torch.save(shared_actor_critic.state_dict(), f'results/state/{exp_name}/batch_{i}.pt')

        # Every once in a while, run evaluation
        # if i % 10 == 0:
        #     with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        #         eval_rollout = env.rollout(1000, policy_actor)
        #         logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
        #         logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
        #         logs["eval step_count"].append(eval_rollout["step_count"].max().item())
        #         eval_str = (
        #             f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
        #             f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
        #             f"eval step-count: {logs['eval step_count'][-1]}"
        #         )
        #         del eval_rollout



if __name__ == "__main__":
    main()
