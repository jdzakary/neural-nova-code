"""
We will use this script to test our exported ONNX models.
Since tic-tac-toe is a very simple game, we can iterate through all
possible games of tic-tac-toe using recursion.
If we identify any situations in which our agent loses, we will print the
sequence of moves so we can test it ourselves.
"""
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import onnxruntime

PROJECT_PATH = Path(__file__).parent


class Model:
    """
    Class to manage our model.
    The validation method calls the move method to test
    how the model responds to a given situation.
    """
    def __init__(
        self,
        player: Literal['X', 'O'],
        model_name: str,
    ) -> None:
        self.__player = 1 if player == 'X' else -1
        file_name = str(PROJECT_PATH / f'exports/{model_name}.onnx')
        self.__model = onnxruntime.InferenceSession(
            path_or_bytes=file_name,
            providers=['CPUExecutionProvider']
        )

    def move(self, board: np.ndarray) -> int:
        obs = board.astype(np.float32)
        mask = np.zeros((10,), dtype=np.float32)
        m1 = np.ones((3, 3), dtype=np.float32)
        occupied = board.reshape((3, 3)) != 0
        m1[occupied] = 0
        mask[0:9] = m1.flatten()
        idx = self.__model.run(None, {'obs': obs, 'mask': mask})
        return int(idx[0])


def game_over(board: np.ndarray, force_win: bool = False) -> bool:
    """
    Given a (3x3) board, determine if the game is over.

    Force_win is useful during post-mortem analysis to determine
    if a player won or tied.
    """
    board = board.reshape((3, 3))
    if np.any(np.abs(board.sum(axis=0)) == 3):
        return True
    if np.any(np.abs(board.sum(axis=1)) == 3):
        return True
    if board[0, 0] == board[1, 1] == board[2, 2] != 0:
        return True
    if board[0, 2] == board[1, 1] == board[2, 0] != 0:
        return True
    if force_win:
        return False
    return np.abs(board).sum() == 9


def compute_unexplored(history: bytes) -> list[list[int]]:
    """
    Given a sequence of moves, compute all possible next moves.
    """
    options = list(range(9))
    for i in history:
        options.remove(i)
    return [
        [*history, x] for x in options
    ]


def compute_board(history: list[int] | bytes) -> np.ndarray:
    """
    Given a sequence of moves, compute the board state.
    """
    board = np.zeros((9,))
    player = 1
    for i in history:
        board[i] = player
        player *= -1
    return board


def explore(
    model: Callable[[np.ndarray], int],
    history: bytes = bytes(),
    terminations: list[bytes] = None,
    move_first: bool = False,
) -> list[bytes]:
    """
    Recursive function that iterates all possible tic-tac-toe games.
    Test an AI agent to see if it ever loses.
    """
    if terminations is None:
        terminations = []
    if not len(history) and not move_first:
        idx = model(compute_board([]))
        explore(model, bytes([idx]), terminations, move_first)
    else:
        next_moves = compute_unexplored(history)
        for move in next_moves:
            board = compute_board(move)
            if game_over(board):
                terminations.append(bytes(move))
            else:
                idx = model(board)
                m1 = [*move, idx]
                if not isinstance(idx, int):
                    raise Exception
                if game_over(compute_board(m1)):
                    terminations.append(bytes(m1))
                else:
                    explore(model, bytes(m1), terminations, move_first)
    return terminations


def compute_winner(history: bytes) -> Literal['X', 'O', 'Tie']:
    """
    Given a sequence of moves, compute the winner.
    """
    board = compute_board(history)
    if game_over(board, force_win=True):
        return 'O' if len(history) % 2 == 0 else 'X'
    return 'Tie'


def show_stats(results: list[Literal['X', 'O', 'Tie']], model: Literal['X', 'O']) -> None:
    """
    Given a list of results, compute percentages and print to screen.
    """
    print(f'Games Played: {len(results)}')
    print(f'Model Loses: {results.count("X" if model == "O" else "O")}')
    print(f"Win X: {results.count('X') / len(results):.4f}")
    print(f"Win O: {results.count('O') / len(results):.4f}")
    print(f"Tie  : {results.count('Tie') / len(results):.4f}")


def show_game_history(history: list[int] | bytes) -> None:
    """
    Print the game history to console.
    """
    board = '         '
    player = 'X'
    for i in history:
        board = board[:i] + player + board[i + 1:]
        print(f'|{board[0:3]}|\n|{board[3:6]}|\n|{board[6:9]}|\n')
        player = 'O' if player == 'X' else 'X'


def main() -> None:
    """
    Test both RL agents against every possible tic-tac-toe game.
    Compute percentages for win/tie/loss, and show the sequence
    of moves that result in any losses.
    """
    print('Model is O:')
    model = Model(player='O', model_name='model-O')
    result_o = explore(model.move, move_first=True)
    winner_o = [compute_winner(r) for r in result_o]
    show_stats(winner_o, 'O')
    for i, winner in enumerate(winner_o):
        if winner == 'X':
            print(list(result_o[i]))

    print('\nModel is X:')
    model = Model(player='X', model_name='model-X')
    result_x = explore(model.move, move_first=False)
    winner_x = [compute_winner(r) for r in result_x]
    show_stats(winner_x, 'X')


if __name__ == '__main__':
    main()
