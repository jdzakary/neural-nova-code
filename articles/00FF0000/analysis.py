import json
import os
import pandas as pd

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


def main():
    create_readable_csv()
    find_best()


if __name__ == '__main__':
    main()
