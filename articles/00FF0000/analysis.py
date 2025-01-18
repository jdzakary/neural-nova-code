import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from shared.ray.result_extraction import extract_df, identify_best


def create_readable_csv():
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
    for trial in os.listdir('results/Random-First-Move'):
        if os.path.isdir(f'results/Random-First-Move/{trial}'):
            df = extract_df(f'results/Random-First-Move/{trial}', to_keep)
            df.rename(columns=col_map, inplace=True)
            df = df.round(4)
            df.to_csv(f'analysis/{trial}.csv', index=False)


def find_best():
    """
    Find the trial that gives the highest amount of ties.
    Print dataframe to console.
    """
    best = identify_best(
        experiment='results/Random-First-Move',
        target_metric='env_runners/Tie'
    )
    hyper = hyperparameter_extraction()
    info = pd.merge(best, hyper, left_index=True, right_index=True)
    print(info)
    info.to_csv('analysis/trial_info.csv')


def hyperparameter_extraction() -> pd.DataFrame:
    info = []
    for trial in os.listdir('results/Random-First-Move'):
        if os.path.isdir(f'results/Random-First-Move/{trial}'):
            with open(f'results/Random-First-Move/{trial}/params.json', 'r') as file:
                data = json.load(file)
            info.append({
                'trial': trial,
                'lr': data['lr'],
                'tie_penalty': data['env_config']['tie_penalty'],
                'gamma': data['gamma'],
            })
    info = pd.DataFrame(info)
    info.set_index('trial', inplace=True)
    return info


def create_plots() -> None:
    df = pd.read_csv('analysis/d0b80b71.csv')
    df['Episodes'] /= 1000

    fig: Figure = plt.figure(figsize=(8, 6))
    grid = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])

    ax1.plot(df['Episodes'], df['WinX'], label='WinX')
    ax1.plot(df['Episodes'], df['WinO'], label='WinO')
    ax1.plot(df['Episodes'], df['Tie'], label='Tie')
    ax1.set_ylabel('Outcome Percentage', fontsize=14)
    ax1.tick_params(labelsize=12)
    ax1.set_xticklabels([f'{x:.0f}k' for x in [0, 0, 25, 50, 75, 100, 125]])
    ax1.legend(edgecolor='black')

    ax2.plot(df['Episodes'], df['EpisodeReturnMean'], label='SUM', color='#116925')
    ax2.plot(df['Episodes'], df['ReturnO'], label='Agent O', color='#9e0c09')
    ax2.plot(df['Episodes'], df['ReturnX'], label='Agent X', color='#424b54')
    labels = ax2.get_xticks()
    ax2.set_xticklabels([f'{x:.0f}k' for x in labels])
    ax2.set_ylabel('Average Return', fontsize=14)
    ax2.tick_params(labelsize=12)
    ax2.legend(edgecolor='black')

    ax3.plot(df['Episodes'], df['EpisodeLengthMean'], color='black')
    labels = ax3.get_xticks()
    ax3.set_xticklabels([f'{x:.0f}k' for x in labels])
    ax3.set_ylabel('Average Length', fontsize=14)
    ax3.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig('analysis/d0b80b71.png', dpi=100)
    plt.show()


def main():
    create_readable_csv()
    find_best()
    create_plots()


if __name__ == '__main__':
    main()
