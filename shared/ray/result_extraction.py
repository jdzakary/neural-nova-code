import os

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


def extract_df(trial: str, keep: list[str] = None) -> pd.DataFrame:
    """
    Extract a dataframe from the progress CSV file.

    :param trial:
        Full path to the trail directory.
    :param keep:
        Optional list of columns to keep. If provided,
        will prune DF to only contain these columns.
        If not provided, will keep all columns.
    """
    df = pd.read_csv(f'{trial}/progress.csv')
    df.dropna(axis=0, inplace=True)
    if keep is not None:
        drop = [c for c in df.columns if c not in keep]
        df.drop(labels=drop, axis=1, inplace=True)
    return df


def extract_all_df(experiment: str, threshold: int = 0) -> pd.DataFrame:
    """
    Extract progress dataframes for every trial in a Ray experiment,
    and combine into a single dataframe.

    :param experiment:
        Full path to the experiment directory.
    :param threshold:
        Minimum number of iterations a trial must have
        undergone during training. If a trial was trained
        for less than `threshold`, do not include it
    """
    trials = [x for x in os.listdir(experiment) if os.path.isdir(f'{experiment}/{x}')]
    df = pd.DataFrame([])
    for trial in trials:
        new = extract_df(f'{experiment}/{trial}')
        new['trial'] = trial
        if len(new) > threshold:
            df = pd.concat((df, new.iloc[threshold:, :]), ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    return df


def get_max_score_iter(group, target: str):
    """
    Get the training iteration that resulted in
    the maximum score for an individual trial.

    Useful for determining if a trail kept improving up
    until termination or if it peaked earlier in training.
    """
    return group.loc[group[target].idxmax()]['training_iteration']


def identify_best(
    experiment: str,
    target_metric: str,
    threshold: int = 100
) -> pd.DataFrame:
    """
    Identify the top trials in a Ray experiment based on
    a metric that should be maximized.

    :param experiment:
        Full path to the experiment directory.
    :param target_metric:
        Name of the target metric. Needs to follow
        the ray naming convention.
    :param threshold:
        Minimum number of training iterations a trail
        must have undergone in order to be considered.
    :return:
    """
    df = extract_all_df(experiment, threshold)
    g1: DataFrameGroupBy = df.groupby(by='trial')
    best_iter = g1.apply(lambda g: get_max_score_iter(g, target_metric), include_groups=False)
    best_iter.name = 'Best_Iter'
    df1 = g1.aggregate({target_metric: 'max', 'training_iteration': 'max'})
    df1.rename(columns={target_metric: 'Metric', 'training_iteration': 'Iters'}, inplace=True)
    df1.sort_values(by='Metric', ascending=False, inplace=True)
    df1 = df1.join(other=best_iter)
    return df1