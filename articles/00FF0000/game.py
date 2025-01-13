from typing import Literal

import numpy as np


class Game:
    """
    Class to track the logic of a simple tic-tac-toe game.
    The training environment calls move until the game terminates.
    Properties convey information to the environment about the game's status.
    """

    @property
    def remaining_moves(self):
        return np.sum(self.board == 0)

    @property
    def board(self) -> np.ndarray:
        return self.__board

    @property
    def turn(self) -> Literal['X', 'O']:
        return 'X' if self.__player == 1 else 'O'

    @property
    def game_over(self) -> bool:
        return self.__winner != 0 or self.remaining_moves == 0

    @property
    def winner_symbol(self) -> Literal['X', 'O'] | None:
        if self.__winner == 1:
            return 'X'
        elif self.__winner == -1:
            return 'O'
        else:
            return None

    def __init__(self):
        self.__board = np.zeros((3, 3), dtype=np.int8)
        self.__player = 1
        self.__winner = 0

    def __switch_player(self) -> None:
        self.__player *= -1

    def move(self, a: int, b: int) -> None:
        if self.game_over:
            raise ValueError('Game is already over!')
        if self.__board[a, b] != 0:
            raise ValueError('Space is occupied')

        self.__board[a, b] = self.__player
        if self.__check_winner(self.__board, self.__player):
            self.__winner = self.__player
        else:
            self.__switch_player()

    # noinspection DuplicatedCode
    @staticmethod
    def __check_winner(board: np.ndarray, player: int) -> bool:
        truth: np.ndarray = np.equal(board, player)
        # noinspection PyTypeChecker
        if any(np.sum(truth, axis=0) == 3):
            return True
        # noinspection PyTypeChecker
        if any(np.sum(truth, axis=1) == 3):
            return True
        if all([truth[0, 0], truth[1, 1], truth[2, 2]]):
            return True
        if all([truth[0, 2], truth[1, 1], truth[2, 0]]):
            return True
        return False
