import multiprocessing
import warnings
from collections import defaultdict, OrderedDict

import gym
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torchrl.modules import MaskedOneHotCategorical
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
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
            'tie_penalty': -0.25,
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

    lr = 0.003
    max_grad_norm = 1.0
    frames_per_batch = 5_000
    total_frames = 100_000
    num_envs = 4
    num_epochs = 5
    clip_epsilon = 0.2
    gamma = 0.97
    lmbda = 0.95
    entropy_eps = 1e-4

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
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
        separate_losses=True,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optim,
        T_max=total_frames // frames_per_batch,
        eta_min=lr / 10
    )


    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

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
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

        # After training, log diagnostics
        logs["batch"].append(i)
        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
        scheduler.step()

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
