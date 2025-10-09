import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from shared.ray.result_extraction import extract_df, identify_best


def create_readable_csv(experiment_name):
    """
    Create CSV files of key training metrics.
    These files can be used for plotting and comparing
    trials with each other.
    """
    col_map = {
        'num_env_steps_sampled_lifetime': 'EnvSteps',
        'num_episodes_lifetime':'Episodes',
        'training_iteration': 'Iters',
        'time_this_iter_s': 'TimeThisIter',
        'time_total_s': 'TimeTotal',
        'env_runners/WinX': 'WinX',
        'env_runners/WinO': 'WinO',
        'env_runners/Tie': 'Tie',
        'env_runners/episode_len_mean': 'EpisodeLengthMean',
        'env_runners/episode_return_mean': 'EpisodeReturnMean',
        'env_runners/agent_episode_returns_mean/O': 'ReturnO',
        'env_runners/agent_episode_returns_mean/X': 'ReturnX',
    }
    to_keep = list(col_map.keys())
    for trial in os.listdir(f'results/{experiment_name}'):
        if os.path.isdir(f'results/{experiment_name}/{trial}'):
            df = extract_df(f'results/{experiment_name}/{trial}', to_keep)
            if (df is not None):
                df.rename(columns=col_map, inplace=True)
                df = df.round(4)
                df.to_csv(f'analysis/{trial}.csv', index=False)


def find_best(experiment_name):
    """
    Find the trial that gives the highest amount of ties.
    Print dataframe to console.
    """
    best = identify_best(
        experiment=f'results/{experiment_name}',
        target_metric='env_runners/Tie'
    )
    hyper = hyperparameter_extraction(experiment_name)
    info = pd.merge(best, hyper, left_index=True, right_index=True)
    info.to_csv('analysis/trial_info.csv')


def hyperparameter_extraction(experiment_name) -> pd.DataFrame:
    info = []
    for trial in os.listdir(f'results/{experiment_name}'):
        trial_dir = f'results/{experiment_name}/{trial}'
        if os.path.isdir(trial_dir) and len(list(os.listdir(trial_dir))) > 0:
            with open(f'results/{experiment_name}/{trial}/params.json', 'r') as file:
                data = json.load(file)
            if ('tie_penalty' in data['env_config']):
                info.append({
                    'trial': trial,
                    'lr': data['lr'],
                    'tie_penalty': data['env_config']['tie_penalty'],
                    'gamma': data['gamma'],
                })
            else:
                info.append({
                    'trial': trial,
                    'lr': data['lr'],
                    'x_tie_penalty': data['env_config']['x_tie_penalty'],
                    'o_tie_penalty': data['env_config']['o_tie_penalty'],
                    'gamma': data['gamma'],
                })
    info = pd.DataFrame(info)
    info.set_index('trial', inplace=True)
    return info


def create_plots(trial) -> None:
    df = pd.read_csv(f'analysis/{trial}.csv').to_dict(orient="list")

    fig: Figure = plt.figure(figsize=(8, 6))
    grid = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])

    ax1.plot(df['Iters'], df['WinX'], label='WinX')
    ax1.plot(df['Iters'], df['WinO'], label='WinO')
    ax1.plot(df['Iters'], df['Tie'], label='Tie')
    ax1.set_ylabel('Outcome Percentage', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_xlim(0)
    ax1.set_ylim(0)
    labels = ax1.get_xticks()
    ax1.set_xticklabels([f'{x:.0f}k' for x in labels])
    ax1.legend(edgecolor='black')

    ax2.plot(df['Iters'], df['EpisodeReturnMean'], label='SUM', color='#116925')
    ax2.plot(df['Iters'], df['ReturnO'], label='Agent O', color='#9e0c09')
    ax2.plot(df['Iters'], df['ReturnX'], label='Agent X', color='#424b54')
    labels = ax2.get_xticks()
    ax2.set_xticklabels([f'{x:.0f}k' for x in labels])
    ax2.set_ylabel('Average Return', fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.legend(edgecolor='black')

    ax3.plot(df['Iters'], df['EpisodeLengthMean'], color='black')
    labels = ax3.get_xticks()
    ax3.set_xticklabels([f'{x:.0f}k' for x in labels])
    ax3.set_ylabel('Average Turns', fontsize=14)
    ax3.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(f'analysis/{trial}.png', dpi=100)
    plt.show()


def main(args):
    create_readable_csv(args.experiment_name)
    find_best(args.experiment_name)
    if (args.create_plots is not None):
        create_plots(args.create_plots)


if __name__ == '__main__':
    # python analysis.py --experiment-name "Random-First-Move" --create-plots "2da66817"
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--create-plots", type=str, default=None)
    args = parser.parse_args()
    main(args)
