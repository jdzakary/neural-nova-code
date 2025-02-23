from typing import Literal

import numpy as np


class Game:
    """
    Class to track the logic of a mega tic-tac-toe game.
    The training environment calls move until the game terminates.
    Properties convey information to the environment about the game's status.
    """

    @property
    def board(self) -> np.ndarray:
        return self.__board

    @property
    def turn(self) -> Literal['X', 'O']:
        return 'X' if self.__player == 1 else 'O'

    @property
    def game_over(self) -> bool:
        if self.__winner != 0:
            return True
        return np.all(self.__big_remaining == 0)

    @property
    def winner(self) -> int:
        return self.__winner

    @property
    def constraint(self) -> np.ndarray:
        return self.__constraint

    @property
    def big_next(self) -> bool:
        return self.__big_next

    @property
    def big_board(self) -> np.ndarray:
        return self.__big_board

    @property
    def target_square(self) -> (int, int):
        return self.__target_square

    def __init__(self):
        self.__board = np.zeros((9, 9), dtype=np.float64)
        self.__big_board = np.zeros((3, 3), dtype=np.float64)
        self.__big_remaining = np.ones((3, 3), dtype=np.float64)
        self.__player = 1
        self.__winner = 0
        self.__constraint = np.ones((9, 9), dtype=np.float64)
        self.__turns = 0

    def __switch_player(self) -> None:
        self.__player *= -1

    def move(self, a: int, b: int) -> None:
        self.__turns += 1
        if self.game_over:
            raise ValueError(f'Game is already over! {self.__turns}')
        if self.__constraint[a, b] == 0:
            raise ValueError(f'Illegal Move! ({a}, {b}, {self.__turns})')

        self.__board[a, b] = self.__player
        if self.__check_winner(a, b):
            self.__winner = self.__player
        else:
            self.__update_constraint(a, b)
            self.__switch_player()

    def __check_winner(self, a: int, b: int) -> bool:
        # Global position of the local square top-left corner
        r = (a // 3) * 3
        c = (b // 3) * 3

        # Big-Board Idx that corresponds to local square
        br = a // 3
        bc = b // 3

        # Local square data
        section = self.__board[r:r+3, c:c+3]

        # Check local Square
        if not self.__can_continue(section):
            self.__big_remaining[br, bc] = 0
            if self.__local_winner(section, self.__player):
                self.__big_board[br, bc] = self.__player

        return self.__local_winner(self.__big_board, self.__player)

    def __update_constraint(self, a: int, b: int):
        # Get the relative position within the local board
        # This maps to the big-remaining
        r = a % 3
        c = b % 3

        self.__constraint[:, :] = 0
        if self.__big_remaining[r, c] == 1:
            self.__target_square = (r, c)
            self.__constraint[r*3:r*3+3, c*3:c*3+3] = 1
            self.__big_next = False
        else:
            self.__big_next = True
            for br in range(3):
                for bc in range(3):
                    if self.__big_remaining[br, bc] == 1:
                        self.__constraint[br*3:br*3+3, bc*3:bc*3+3] = 1
        self.__constraint[self.__board != 0] = 0

    def __can_continue(self, board: np.ndarray) -> bool:
        if np.all(board != 0):
            return False
        if self.__local_winner(board, 1):
            return False
        if self.__local_winner(board, -1):
            return False
        return True

    # noinspection DuplicatedCode
    @staticmethod
    def __local_winner(board: np.ndarray, player: int) -> bool:
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
