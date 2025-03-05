import multiprocessing
import random
import warnings
from collections import OrderedDict
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDictBase, TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule, \
    TensorDictSequential
from torchrl.modules import MaskedOneHotCategorical
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose, PettingZooWrapper, Transform
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger
from tqdm import tqdm

from model.cnn import Actor, Critic
from environment import MultiAgent

warnings.filterwarnings(action='ignore')


def create_env() -> TransformedEnv:
    device = 'cpu'
    base_env = PettingZooWrapper(
        env=MultiAgent(
            {
                'tie_reward': 0.25,
                'reward': 1.0,
            }
        ),
        use_mask=True,
        group_map=None,
        device=device
    )
    env = TransformedEnv(
        base_env,
        Compose(
            DoubleToFloat(),
            StepCounter(),
        )
    )
    return env


def create_agent(
    label: Literal['X', 'O'],
    device: torch.device,
    clip_epsilon: float,
    entropy_eps: float,
    gamma: float,
    lmbda: float
):
    actor_net = Actor()
    actor_net.to(device=device)
    policy_module = TensorDictModule(
        module=actor_net,
        in_keys=(label, "observation", "obs"),
        out_keys=(label, "logits"),
    )
    critic_net = Critic()
    critic_net.to(device=device)
    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=(label, "observation", "obs"),
        out_keys=(label, "state_value"),
    )
    dist = ProbabilisticTensorDictModule(
        in_keys={
            'logits': (label, "logits"),
            'mask': (label, "observation", "mask")
        },
        out_keys=(label, 'action'),
        distribution_class=MaskedOneHotCategorical,
        return_log_prob=True,
        log_prob_key=(label, 'action_log_prob'),
    )

    actor = ProbabilisticTensorDictSequential(
        OrderedDict({
            'module': policy_module,
            'dist': dist,
        })
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=0.8,
        loss_critic_type="smooth_l1",
    )
    loss_module.set_keys(
        reward=(label, 'reward'),
        action=(label, 'action'),
        value=(label, "state_value"),
    )

    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=critic_module,
        average_gae=True,
    )
    advantage_module.set_keys(
        reward=(label, 'reward'),
        value=(label, "state_value"),
        done=(label, 'done'),
        terminated=(label, 'terminated')
    )

    return actor, loss_module, advantage_module


def main():
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    lr = 0.001
    max_grad_norm = 1.0
    frames_per_batch = 4_000
    sub_batch = 400
    total_frames = 4_000_000
    num_envs = 4
    epochs = 3
    clip_epsilon = 0.2
    gamma = 1
    lmbda = 0.95
    entropy_eps = 0.0005
    exp_name = 'exp7'

    actor_x, loss_x, adv_x = create_agent('X', device, clip_epsilon, entropy_eps, gamma, lmbda)
    actor_o, loss_o, adv_o = create_agent('O', device, clip_epsilon, entropy_eps, gamma, lmbda)
    combined_policy = TensorDictSequential([actor_x, actor_o])

    collector = MultiSyncDataCollector(
        create_env_fn=[create_env for _ in range(num_envs)],
        policy=combined_policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device='cpu',
        update_at_each_batch=True,
        cat_results=0
    )

    optimisers = {
        'X': torch.optim.Adam(
                loss_x.parameters(), lr=lr
            ),
        'O': torch.optim.Adam(
                loss_o.parameters(), lr=lr
            ),
    }
    losses = {
        'X': loss_x,
        'O': loss_o,
    }
    advs = {
        'X': adv_x,
        'O': adv_o,
    }

    logger = CSVLogger(exp_name, 'results/logs')
    pbar = tqdm(total=total_frames // frames_per_batch)
    ema = 0
    smooth_factor = 0.1
    best = -np.inf
    spacer = 0
    logger.log_hparams({
        'ema_smooth_factor': smooth_factor,
        'learning_rate': lr,
        'max_grad_norm': max_grad_norm,
        'frames_per_batch': frames_per_batch,
        'sub_batch': sub_batch,
        'total_frames': total_frames,
        'num_envs': num_envs,
        'epochs': epochs,
        'clip_epsilon': clip_epsilon,
        'gamma': gamma,
        'lambda': lmbda,
        'entropy_eps': entropy_eps,
    })

    # Collect Data
    try:
        for i, tensordict_data in enumerate(collector):
            gpu_dict: TensorDict = tensordict_data.to(device=device)

            # TorchRL multi-agent wrapper sets batch size to [B, 1]... we need [B]
            gpu_dict['collector', 'traj_ids'] = gpu_dict['collector', 'traj_ids'].unsqueeze(1)
            b = gpu_dict.batch_size[0]
            gpu_dict.batch_size = [b, 1]
            gpu_dict = gpu_dict.squeeze(1)

            # Train each agent
            for group in ['X', 'O']:
                loss_module = losses[group]
                adv = advs[group]
                optim = optimisers[group]
                for _ in range(epochs):
                    adv(gpu_dict)
                    sampled_idx = []
                    for _ in range(frames_per_batch // sub_batch):
                        subdata_idx = random.sample(
                            population=[x for x in range(frames_per_batch) if x not in sampled_idx],
                            k=sub_batch
                        )
                        sampled_idx.extend(subdata_idx)
                        subdata = gpu_dict[subdata_idx]
                        loss_vals = loss_module(subdata)
                        loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                        optim.step()
                        optim.zero_grad()

            # After training, log diagnostics
            episodic_x = tensordict_data['next', 'X', 'reward'][tensordict_data['next', 'X', 'reward'] != 0]
            win_x = (episodic_x == 1).sum() / len(episodic_x)
            win_o = (episodic_x == -1).sum() / len(episodic_x)
            tie = (episodic_x == 0.25).sum() / len(episodic_x)
            if i == 0:
                ema = tie.item()
            else:
                ema = (tie.item() * smooth_factor) + (ema * (1 - smooth_factor))
            spacer += 1
            logger.log_scalar('reward_x', episodic_x.mean().item(), i)
            logger.log_scalar('win_x', win_x, i)
            logger.log_scalar('win_o', win_o, i)
            logger.log_scalar('tie', tie, i)
            logger.log_scalar('tie_smooth', ema, i)
            pbar.update()
            if spacer > 40 and ema > best:
                spacer = 0
                best = ema
                torch.save(actor_x.state_dict(), f'results/state/{exp_name}/batch_{i}_actor_x.pt')
                torch.save(actor_o.state_dict(), f'results/state/{exp_name}/batch_{i}_actor_o.pt')
    except KeyboardInterrupt:
        print('Training interrupted.')
    finally:
        # noinspection PyUnboundLocalVariable
        torch.save(actor_x.state_dict(), f'results/state/{exp_name}/batch_{i}_actor_x.pt')
        torch.save(actor_o.state_dict(), f'results/state/{exp_name}/batch_{i}_actor_o.pt')


if __name__ == "__main__":
    main()
