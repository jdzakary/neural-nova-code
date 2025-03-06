from typing import Callable

import numpy as np
import time
from game_vectorized import GameVectorized
from models import model_v1, ModelWrapper


def explore(
    model: Callable,
    starting: np.ndarray,
    last_move: int,
):
    # Check input
    if starting.shape != (1, 9, 9):
        raise ValueError('Starting board must have shape (1, 9, 9)')
    if starting.dtype != np.int8:
        raise ValueError('Starting board must have dtype np.int8')

    # Initialize Arrays
    result_history = np.ones((0, 81), dtype=np.int8)
    result_board = np.ones((0, 9, 9), dtype=np.int8)
    result_winner = np.ones((0,), dtype=np.int8)
    active_history = -1 * np.ones((1, 81), dtype=np.int8)
    active_history[0, 0] = last_move

    # Setup game
    game = initialize_game(starting, last_move)
    move_count = np.sum(game.board != 0)

    if np.sum(starting != 0) % 2 == 0:
        t1 = time.perf_counter()
        moves = expand_unexplored(game)
        active_history = active_history[moves[:, 0]]
        game.move(moves[:, 1].flatten())
        board_f, winner_f, finished = remove_finished(game)
        result_board = np.concatenate((result_board, board_f), axis=0)
        result_winner = np.concatenate((result_winner, winner_f), axis=0)
        result_history = np.concatenate((result_history, active_history[finished]), axis=0)
        active_history = active_history[~finished]
        t2 = time.perf_counter()
        move_count += 1
        print(
            f'X {move_count:>2} '
            f'{active_history.shape[0]:>16,} '
            f'{result_winner.size:>12,} '
            f'{t2 - t1:>10.3f} '
            f'{game.board.nbytes / 1000_000:>8,.2f} MB'
        )

    while game.board.shape[0] > 0:
        t1 = time.perf_counter()
        moves = model(game.board, game.mask, active_history)
        game.move(moves)
        board_f, winner_f, finished = remove_finished(game)
        result_board = np.concatenate((result_board, board_f), axis=0)
        result_winner = np.concatenate((result_winner, winner_f), axis=0)
        result_history = np.concatenate((result_history, active_history[finished]), axis=0)
        active_history = active_history[~finished]
        t2 = time.perf_counter()
        move_count += 1
        print(
            f'O {move_count:>2} '
            f'{active_history.shape[0]:>16,} '
            f'{result_winner.size:>12,} '
            f'{t2 - t1:>10.3f} '
            f'{game.board.nbytes / 1000_000:>8,.2f} MB'
        )

        t1 = time.perf_counter()
        moves = expand_unexplored(game)
        active_history = active_history[moves[:, 0]]
        game.move(moves[:, 1].flatten())
        board_f, winner_f, finished = remove_finished(game)
        result_board = np.concatenate((result_board, board_f), axis=0)
        result_winner = np.concatenate((result_winner, winner_f), axis=0)
        result_history = np.concatenate((result_history, active_history[finished]), axis=0)
        active_history = active_history[~finished]
        t2 = time.perf_counter()
        move_count += 1
        print(
            f'X {move_count:>2} '
            f'{active_history.shape[0]:>16,} '
            f'{result_winner.size:>12,} '
            f'{t2 - t1:>10.3f} '
            f'{game.board.nbytes / 1000_000:>8,.2f} MB'
        )
    return result_board, result_winner, result_history



def remove_finished(game: GameVectorized):
    finished = game.finished
    board_f = game.board[finished]
    winner_f = game.winner[finished]
    game.board = game.board[~finished]
    game.big_board = game.big_board[~finished]
    game.big_remaining = game.big_remaining[~finished]
    game.winner = game.winner[~finished]
    game.constraint = game.constraint[~finished]
    return board_f, winner_f, finished


def expand_unexplored(game: GameVectorized):
    idx = np.argwhere(game.mask)
    game.board = game.board[idx[:, 0]]
    game.big_board = game.big_board[idx[:, 0]]
    game.big_remaining = game.big_remaining[idx[:, 0]]
    game.winner = game.winner[idx[:, 0]]
    game.constraint = game.constraint[idx[:, 0]]
    return idx


def initialize_game(starting: np.ndarray, last_move: int) -> GameVectorized:
    player = starting[0, *np.unravel_index(last_move, (9, 9))]
    if player == 0:
        raise ValueError('Last Move does not correspond to any player')
    starting[0, *np.unravel_index(last_move, (9, 9))] = 0
    big = np.zeros((9, 3, 3), dtype=np.int8)
    big[0] = starting[0, 0:3, 0:3]
    big[1] = starting[0, 0:3, 3:6]
    big[2] = starting[0, 0:3, 6:9]
    big[3] = starting[0, 3:6, 0:3]
    big[4] = starting[0, 3:6, 3:6]
    big[5] = starting[0, 3:6, 6:9]
    big[6] = starting[0, 6:9, 0:3]
    big[7] = starting[0, 6:9, 3:6]
    big[8] = starting[0, 6:9, 6:9]
    mask_c, mask_x, mask_o = GameVectorized.can_continue(big)
    big_remaining = np.zeros((1, 3, 3), dtype=np.int8)
    big_board = np.zeros((1, 3, 3), dtype=np.int8)
    big_remaining[0, mask_c.reshape((3, 3))] = 1
    big_board[0, mask_x.reshape((3, 3))] = 1
    big_board[0, mask_o.reshape((3, 3))] = -1

    game = GameVectorized()
    game.board = starting
    game.big_board = big_board
    game.big_remaining = big_remaining
    game.player = player
    game.move(np.array(last_move, dtype=np.int8))

    return game


def create_starting(file_name: str) -> np.ndarray:
    starting: np.ndarray = np.load(file_name)
    starting = starting.astype(np.int8)
    visualize_board(starting)
    return starting.reshape((1, 9, 9))


def visualize_board(board: np.ndarray) -> None:
    """
    Visualize a board of shape (9, 9)
    :param board:
    :return:
    """
    for i in range(9):
        if i % 3 == 0:
            print('-' * 13)
        row = []
        for j in range(9):
            row.append(board[i, j])
        for k, value in enumerate(row):
            match value:
                case 0:
                    char = ' '
                case 1:
                    char = 'X'
                case -1:
                    char = 'O'
                case _:
                    raise ValueError
            if k % 3 == 0 and k != 0:
                print('|', end='')
            print(char, end='')
        print('|')
    print('-' * 13)


def main():
    print('Starting Main!')
    starting = create_starting('start_1.npy')
    model = ModelWrapper()
    result_board, result_winner, result_history = explore(
        model=model.run,
        starting=starting,
        last_move=2,
    )
    print(result_board.shape)
    print(f'Win X: {np.mean(result_winner == 1) * 100:.3f} %')
    print(f'Win O: {np.mean(result_winner == -1) * 100:.3f} %')
    print(f'Tie  : {np.mean(result_winner == 0) * 100:.3f} %')


if __name__ == '__main__':
    main()
