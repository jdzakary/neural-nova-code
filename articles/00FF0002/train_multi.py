import multiprocessing
import random
import warnings
from collections import OrderedDict

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule
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


class MultiAgentStackTransform(Transform):
    def __init__(self, in_keys=None, out_keys=None):
        # Define default input and output keys
        if in_keys is None:
            in_keys = {
                "observation": [("O", "observation", "observations"), ("X", "observation", "observations")],
                "action_mask": [("O", "action_mask"), ("X", "action_mask")]
            }
        if out_keys is None:
            out_keys = {
                "observation": ("agents", "observation"),
                "action_mask": ("agents", "action_mask")
            }

        # Initialize parent class with in_keys and out_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.in_keys_dict = in_keys
        self.out_keys_dict = out_keys

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Process the tensordict during forward step."""
        # Stack observations
        obs_o = tensordict.get(self.in_keys_dict["observation"][0]).squeeze(1)  # [5, 2, 9, 9]
        obs_x = tensordict.get(self.in_keys_dict["observation"][1]).squeeze(1)  # [5, 2, 9, 9]
        stacked_obs = torch.stack([obs_o, obs_x], dim=1)  # [5, 2, 2, 9, 9]
        tensordict.set(self.out_keys_dict["observation"], stacked_obs)

        # Stack action masks
        mask_o = tensordict.get(self.in_keys_dict["action_mask"][0]).squeeze(1)  # [5, 81]
        mask_x = tensordict.get(self.in_keys_dict["action_mask"][1]).squeeze(1)  # [5, 81]
        stacked_masks = torch.stack([mask_o, mask_x], dim=1)  # [5, 2, 81]
        tensordict.set(self.out_keys_dict["action_mask"], stacked_masks)

        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        """Process the tensordict during reset."""
        # Apply the same transformation as in _call
        return self._call(tensordict_reset)

    def transform_output_spec(self, output_spec):
        """Update the environment spec to reflect the transformed keys."""
        # This is optional and depends on whether you need to update the spec
        # For simplicity, we assume the spec is handled by the base env
        return output_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Optional: Inverse transform (not needed here, but included for completeness)."""
        # If you need to reverse the stacking (e.g., for action mapping), implement it here
        return tensordict


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
            # MultiAgentStackTransform(),
            DoubleToFloat(),
            StepCounter(),
        )
    )
    return env


def main():
    #--- Config ---#
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )

    lr = 0.0005
    max_grad_norm = 1.0
    frames_per_batch = 5_000
    sub_batch = 100
    total_frames = 1_000_000
    num_envs = 3
    epochs = 5
    clip_epsilon = 0.15
    gamma = 0.985
    lmbda = 0.95
    entropy_eps = 0.01
    exp_name = 'exp6'

    #--- Policy ---#
    actor_net = Actor()
    actor_net.to(device=device)
    policy_module = TensorDictModule(
        module=actor_net,
        in_keys=[("X", "observation", "observations"), ("O", "observation", "observations")],
        out_keys=[("X", "logits"), ("O", "logits")],
    )
    critic_net = Critic()
    critic_net.to(device=device)
    critic_module = TensorDictModule(
        module=critic_net,
        in_keys=[("X", "observation", "observations"), ("O", "observation", "observations")],
        out_keys=[("X", "state_value"), ("O", "state_value")],
    )

    dist = ProbabilisticTensorDictModule(
        in_keys=[
            ("X", "logits"), ("O", "logits"),
            ("X", "action_mask"), ("O", "action_mask")
        ],
        out_keys=[('X', 'action'), ('O', 'action')],
        distribution_class=MaskedOneHotCategorical,
        return_log_prob=True,
        log_prob_key=[('X', 'action_log_prob'), ('O', 'action_log_prob')],
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
        average_gae=False,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=0.8,
        loss_critic_type="smooth_l1",
        normalize_advantage=False,
    )
    loss_module.set_keys(
        # reward=env.reward_key,
        # action=env.action_key,
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
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
            if spacer > 20 and ema > best:
                spacer = 0
                best = ema
                torch.save(actor_net.state_dict(), f'results/state/{exp_name}/batch_{i}_actor.pt')
                torch.save(critic_net.state_dict(), f'results/state/{exp_name}/batch_{i}_critic.pt')
    except KeyboardInterrupt:
        print('Training interrupted.')
    finally:
        # noinspection PyUnboundLocalVariable
        torch.save(actor_net.state_dict(), f'results/state/{exp_name}/batch_{i}_actor.pt')
        torch.save(critic_net.state_dict(), f'results/state/{exp_name}/batch_{i}_critic.pt')


if __name__ == "__main__":
    main()
