import multiprocessing
import random
import warnings
from collections import OrderedDict

import gym
import numpy as np
import torch
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
from torchrl.modules import MaskedOneHotCategorical
from torchrl.collectors import MultiSyncDataCollector
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import CSVLogger
from tqdm import tqdm

from model.cnn import Actor, Critic

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
            'player': -1,
            'reward': 1.0,
            'tie_reward': 0.25,
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

    lr = 0.0003
    max_grad_norm = 1.0
    frames_per_batch = 5_000
    sub_batch = 100
    total_frames = 4_000_000
    num_envs = 3
    epochs = 4
    clip_epsilon = 0.15
    gamma = 0.985
    lmbda = 0.95
    entropy_eps = 0.15
    exp_name = 'exp2'

    #--- Policy ---#
    actor_net = Actor()
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
            gpu_dict = tensordict_data.to(device=device)
            for _ in range(epochs):
                # Modifies GPU_DICT in place!!!
                advantage_module(gpu_dict)
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
            episodic = tensordict_data['next', 'reward'][tensordict_data['next', 'reward'] != 0]
            if i == 0:
                ema = episodic.mean().item()
            else:
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
