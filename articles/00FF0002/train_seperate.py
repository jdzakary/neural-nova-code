import multiprocessing
import warnings
from collections import OrderedDict

import gym
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torch.cuda.amp import GradScaler, autocast
from torchrl.modules import MaskedOneHotCategorical
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger
from tqdm import tqdm

from model import Actor, Critic

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

    lr = 0.0001
    max_grad_norm = 1.0
    frames_per_batch = 500
    total_frames = 10_000_000
    num_envs = 4
    clip_epsilon = 0.1
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 0.1
    exp_name = 'exp3'

    #--- Policy ---#
    actor_net = Actor(
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
    )
    actor_net.to(device=device)
    policy_module = TensorDictModule(
        module=actor_net,
        in_keys=['observations'],
        out_keys=['logits'],
    )
    critic_net = Critic()
    critic_net.to(device=device)
    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=['observations'],
        out_keys=['state_value'],
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
        value_network=critic_module,
        average_gae=True
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
    optim = torch.optim.Adam(loss_module.parameters(), lr)

    logger = CSVLogger(exp_name, 'results/logs')
    pbar = tqdm(total=total_frames // frames_per_batch)
    ema = 0
    smooth_factor = 0.1
    best = -np.inf
    spacer = 0

    # Collect Data
    try:
        for i, tensordict_data in enumerate(collector):
            tensordict_data: TensorDict
            gpu_dict = tensordict_data.to(device=device)

            # Train on batch
            data = advantage_module(gpu_dict)
            loss_vals = loss_module(data)
            loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

            # After training, log diagnostics
            episodic = tensordict_data['next', 'reward'][tensordict_data['next', 'reward'] != 0]
            ema = (episodic.mean().item() * smooth_factor) + (ema * (1 - smooth_factor))
            spacer += 1
            logger.log_scalar('reward_mean', episodic.mean().item(), i)
            logger.log_scalar('reward_smooth', ema, i)
            logger.log_scalar('step_count_max', tensordict_data["step_count"].max().item(), i)
            logger.log_scalar('lr', optim.param_groups[0]['lr'], i)
            # noinspection PyUnboundLocalVariable
            logger.log_scalar('loss_objective', loss_vals["loss_objective"].item(), i)
            logger.log_scalar('loss_critic', loss_vals["loss_critic"].item(), i)
            logger.log_scalar('loss_entropy', loss_vals["loss_entropy"].item(), i)
            pbar.update()
            if spacer > 40 and ema > best:
                spacer = 0
                best = ema
                torch.save(actor.state_dict(), f'results/state/{exp_name}/batch_{i}_actor.pt')
    except KeyboardInterrupt:
        print('Training interrupted.')
    finally:
        # noinspection PyUnboundLocalVariable
        torch.save(actor.state_dict(), f'results/state/{exp_name}/batch_{i}_actor.pt')


if __name__ == "__main__":
    main()
