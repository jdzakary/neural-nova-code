"""
This is a custom progress reporter for Ray Tune.
It uses curses to create a live "dashboard" that updates
values as training progresses.

I made this because I was annoyed by long-running
tune jobs that would output thousands of lines to console.
It was difficult to easily glance at stdout to see the current
training progress.
Since this uses curses, you will need to "emulate terminal output" if
using an IDE to execute your code.

In the future, I plan to add robust fault tolerance.
Currently, curses will crash the whole training pipeline if
the reporter attempts to print data that is too big for the console window.
This is a standard curses error and could be handled.
For now, ensure your console window is big enough to handle all the
data printed to it.
"""

from __future__ import annotations

import curses
import time
from typing import TYPE_CHECKING, Optional, Any, Literal
from datetime import datetime, timedelta

import pandas as pd
from ray.tune import ProgressReporter
# noinspection PyProtectedMember
from ray.tune.experiment.trial import _Location
from ray.tune.result import NODE_IP, PID
from ray.tune.utils import unflattened_lookup


if TYPE_CHECKING:
    from ray.tune.experiment import Trial


class CustomReporter(ProgressReporter):
    """
    Custom Progress Reporter for Ray Tune
    """

    def __init__(
        self,
        metric_columns: Optional[dict[str, str]] = None,
        parameter_columns: Optional[dict[str, str]] = None,
        max_report_frequency: int = 20,
        metric: Optional[str] = None,
        mode: Optional[Literal['min', 'max']] = 'max',
        include_location: bool = False,
        rounding: Optional[dict[str, int]] = None,
        time_col: Optional[str] = None
    ):
        """
        Initialize the custom reporter.

        :param metric_columns:
            Metric columns to include in the report.
        :param parameter_columns:
            Parameter columns to include in the report.
        :param max_report_frequency:
            Minimum time between console updates.
        :param metric:
            Single metric to sort trials by
        :param mode:
            Should we sort by min or max?
        :param include_location:
            Should we include physical location of trial execution?
        :param rounding:
            Round specific columns to a specified precision
        :param time_col:
            Name of a column to be formatted as datetime.
        """
        self.__max_report_frequency = max_report_frequency
        self.__last_report_time = 0
        self.__last_update_time = 0
        self.__start_time = 0
        self.__window: curses.window
        self.__counter = 0
        self.__metric_columns = metric_columns or {}
        self.__parameter_columns = parameter_columns or {}
        self.__metric = metric
        self.__mode = mode
        self.__include_location = include_location
        self.__rounding = rounding
        self.__time_col = time_col

    def __get_trial_info(
        self,
        trial: Trial,
    ) -> dict[str, Any]:
        """
        Get the data for a single row of the stats table
        """
        result = trial.last_result
        config = trial.config
        location = self.__get_trial_location(trial, result)
        info = {
            "Name": trial.trial_id,
            "Status": trial.status,
        }
        if self.__include_location:
            info['Location'] = str(location)
        for k, v in self.__parameter_columns.items():
            info[v] = unflattened_lookup(k, config, default=None)
        for k, v in self.__metric_columns.items():
            info[v] = unflattened_lookup(k, result, default=None)
        return info

    def __create_stats_table(self, trials: list[Trial]) -> pd.DataFrame:
        """
        Create the main stats table, which is displayed in the console.
        This is limited at 20 rows to prevent crashing the program due to the
        curses bug described at the top of this file.
        """
        df = pd.DataFrame(
            data=[
                self.__get_trial_info(t) for t in trials
            ]
        )
        by = ['Status']
        ascending = [True]
        if self.__metric is not None:
            by.append(self.__metric)
            ascending.append(self.__mode == 'min')
        df.sort_values(by=by, ascending=ascending, inplace=True)
        df.reset_index(inplace=True, drop=True)
        if self.__rounding is not None:
            for key, value in self.__rounding.items():
                df[key] = df[key].round(value)
        if self.__time_col is not None:
            df[self.__time_col] = pd.to_timedelta(df[self.__time_col], unit='s')
        return df.iloc[0:20]

    @staticmethod
    def __get_trial_location(trial: Trial, result: dict) -> _Location:
        """
        Extract information telling us which machine a trial is executing on.
        Useful when training on a cluster.
        """
        # we get the location from the result, as the one in trial will be
        # reset when trial terminates
        node_ip, pid = result.get(NODE_IP, None), result.get(PID, None)
        if node_ip and pid:
            location = _Location(node_ip, pid)
        else:
            # fallback to trial location if there hasn't been a report yet
            location = trial.temporary_state.location
        return location

    def should_report(self, trials: list[Trial], done: bool = False) -> bool:
        """
        Called by Ray Tune to determine if we should update the console.
        If we return True, ray will eventually call our "report" method.
        """
        now = time.time()
        if now - self.__last_update_time > 2:
            self.__last_update_time = now
            self.__update_time(now)
        if now - self.__last_report_time > self.__max_report_frequency:
            self.__last_report_time = now
            return True
        return done

    def __update_time(self, now: float) -> None:
        """
        Update the Time Elapsed label at the top of the console
        """
        now = datetime.fromtimestamp(now)
        start = datetime.fromtimestamp(self.__start_time)
        elapsed: timedelta = now - start
        self.__window.addstr(0, 0, f'Time Elapsed: {elapsed}')
        self.__window.refresh()

    # noinspection PyUnresolvedReferences
    def report(self, trials: list[Trial], done: bool, *sys_info: dict):
        """
        Main function called by Ray Tune to refresh the window.
        """
        stats = self.__create_stats_table(trials)
        info = sys_info[0] + ' ' + sys_info[1]
        # self.__window.addstr(1, 0, info)
        self.__window.addstr(3, 0, stats.__repr__())
        self.__window.refresh()

    def setup(
        self,
        start_time: Optional[float] = None,
        total_samples: Optional[int] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        **kwargs,
    ):
        """
        Called by Ray Tune before training begins.
        Needed for initialization of our curses window.
        """
        if start_time is not None:
            self.__start_time = time.time()
        else:
            self.__start_time = start_time
        self.__window = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        curses.use_default_colors()

