import multiprocessing
import warnings
from collections import defaultdict, OrderedDict

import gym
import torch
from torch import nn
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
from torch.distributions import OneHotCategorical, Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import GymEnv, TransformedEnv, DoubleToFloat, StepCounter, Compose
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from model import MyPolicy

warnings.filterwarnings(action='ignore')

#--- Config ---#
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

lr = 0.003
max_grad_norm = 1.0
frames_per_batch = 1000
total_frames = 100_000
sub_batch_size = 100
num_epochs = 5
clip_epsilon = 0.2
gamma = 0.97
lmbda = 0.95
entropy_eps = 1e-4

#--- Environment ---#
gym.register(
    id='MegaTicTacToe-v0',
    entry_point='environment:MegaTicTacToe',
)
base_env = GymEnv(
    env_name='MegaTicTacToe-v0',
    options={
        'tie_penalty': -0.25,
        'player': -1,
    }
)
env = TransformedEnv(
    base_env,
    Compose(
        DoubleToFloat(),
        StepCounter(),
    )
)

#--- Policy ---#
policy_module = TensorDictModule(
    module=MyPolicy(),
    in_keys=['observations', 'action_mask'],
    out_keys=['logits'],
)

policy_actor = ProbabilisticActor(
    module=policy_module,
    in_keys=['logits'],
    spec=env.action_spec,
    distribution_class=OneHotCategorical,
    return_log_prob=True,
    log_prob_key='logits',
)
sequential = ProbabilisticTensorDictSequential(
    OrderedDict({'A': policy_actor})
)

#--- Value Estimator ---#
value_net = nn.Sequential(
    nn.Flatten(1, -1),
    nn.Linear(81, 256),
    nn.Tanh(),
    nn.Linear(256, 1),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observations"],
)

collector = SyncDataCollector(
    env,
    policy_actor,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=0
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=sequential,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optim,
    T_max=total_frames // frames_per_batch,
    eta_min=0.0
)


logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    # Train on batch "num_epochs" times
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
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
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()
