from typing import Literal

import numpy as np
import numba as nb


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :], nb.int64[:], nb.int64[:], nb.int8[:], nb.int8[:, :, :])],
    '(b,x,x),(b),(b),(y)->(b,y,y)',
    nopython=True,
    cache=True,
    target='parallel',
)
def get_sections(
    board: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    output_helper: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(board.shape[0]):
        result[i] = board[i, r[i]:r[i]+3, c[i]:c[i]+3]


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :], nb.int64[:], nb.int64[:], nb.int8[:, :, :])],
    '(b,x,x),(b),(b)->(b,x,x)',
    nopython=True,
    cache=True,
    target='parallel',
)
def constraint_helper(
    constraint: np.ndarray,
    r: np.ndarray,
    c: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(constraint.shape[0]):
        result[i] = np.zeros((9, 9), dtype=np.int8)
        result[i, r[i]*3:r[i]*3+3, c[i]*3:c[i]*3+3] = 1


class GameVectorized:
    """
    Class to track the logic of a mega tic-tac-toe game.
    The training environment calls move until the game terminates.
    Properties convey information to the environment about the game's status.
    """

    @property
    def mask(self) -> np.ndarray:
        return self.constraint.reshape(self.constraint.shape[0], 81)

    @property
    def finished(self) -> np.ndarray:
        return (~ np.any(self.mask, axis=1)) | (self.winner != 0)

    def __init__(self, ):
        self.board = np.zeros((1, 9, 9), dtype=np.int8)
        self.big_board = np.zeros((1, 3, 3), dtype=np.int8)
        self.big_remaining = np.ones((1, 3, 3), dtype=np.int8)
        self.winner = np.zeros((1,), dtype=np.int8)
        self.constraint = np.ones((1, 9, 9), dtype=np.int8)
        self.player = 1

    def __switch_player(self) -> None:
        self.player *= -1

    def move(self, moves: np.ndarray) -> None:
        self.__check_moves(moves)
        self.__update_board(moves)
        self.__check_winner(moves)
        self.__update_constraint(moves)
        self.__switch_player()

    def __check_moves(self, moves: np.ndarray) -> None:
        if moves.size != self.constraint.shape[0]:
            raise ValueError('Batch Error: Moves != Constraint')
        selected = self.mask[np.arange(moves.size), moves]
        if selected.sum() < moves.size:
            raise ValueError('One or more moves is an invalid action')

    def __update_board(self, moves: np.ndarray) -> None:
        if moves.size != self.board.shape[0]:
            raise ValueError('Batch Error: Moves != Board')
        idx = np.unravel_index(moves, (9, 9))
        self.board[np.arange(moves.size), idx[0], idx[1]] = self.player

    def __check_winner(self, moves: np.ndarray) -> None:
        # Row, Column Idx Arrays
        a, b = np.unravel_index(moves, (9, 9))
        if not isinstance(a, np.ndarray):
            a = np.array([a])
            b = np.array([b])
        batch_idx = np.arange(moves.size)

        # Global position of the local square top-left corner
        r = (a // 3) * 3
        c = (b // 3) * 3

        # Big-Board Idx that corresponds to local square
        br = a // 3
        bc = b // 3

        # Check if destination section is playable
        helper = np.zeros((3,), dtype=np.int8)
        section = get_sections(self.board, r, c, helper)
        mask_c, mask_x, mask_o = GameVectorized.can_continue(section)
        mask_c = ~ mask_c
        self.big_remaining[batch_idx[mask_c], br[mask_c], bc[mask_c]] = 0

        # If destination section is not playable, check for winner
        if self.player == 1:
            win_mask = mask_x
        else:
            win_mask = mask_o
        self.big_board[batch_idx[win_mask], br[win_mask], bc[win_mask]] = self.player

        # Check updated big board for winner
        mask_big = GameVectorized.local_winner(self.big_board, self.player)
        self.winner[mask_big] = self.player

    def __update_constraint(self, moves: np.ndarray) -> None:
        # Row, Column Idx Arrays
        a, b = np.unravel_index(moves, (9, 9))
        if not isinstance(a, np.ndarray):
            a = np.array([a])
            b = np.array([b])
        batch_idx = np.arange(moves.size)

        # Relative position within the local board maps to big-remaining
        r = a % 3
        c = b % 3

        # Reset Constraint to 0
        self.constraint[:] = 0

        # If destination is playable, set constraint to that 3x3
        m1 = self.big_remaining[batch_idx, r, c] == 1
        self.constraint[m1] = constraint_helper(self.constraint[m1], r[m1], c[m1])

        # If destination is not playable, widen constraint
        for br in range(3):
            for bc in range(3):
                m2 = self.big_remaining[batch_idx, br, bc] == 1
                self.constraint[~m1 & m2, br*3:br*3+3, bc*3:bc*3+3] = 1

        # Ensure constraint is 0 for all filled squares
        self.constraint[self.board != 0] = 0

    @staticmethod
    def can_continue(squares: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = np.ones(squares.shape[0], dtype=np.bool_)

        # Masks for no spaces and winners
        mask = np.all(squares != 0, axis=(1, 2))
        mask_x = GameVectorized.local_winner(squares, 1)
        mask_o = GameVectorized.local_winner(squares, -1)
        result[mask | mask_x | mask_o] = np.False_

        return result, mask_x, mask_o

    @staticmethod
    def local_winner(squares: np.ndarray, player: int) -> np.ndarray:
        result = np.zeros((squares.shape[0],), dtype=np.bool_)
        truth: np.ndarray = np.equal(squares, player)

        # Check Horizontal
        mask = np.any(np.sum(truth, axis=1) == 3, axis=1)
        result[mask] = np.True_

        # Check Vertical
        mask = np.any(np.sum(truth, axis=2) == 3, axis=1)
        result[mask] = np.True_

        # Check Diagonals
        mask = np.all(truth[:, [0, 1, 2], [0, 1, 2]], axis=1)
        result[mask] = np.True_
        mask = np.all(truth[:, [0, 1, 2], [2, 1, 0]], axis=1)
        result[mask] = np.True_

        return result
