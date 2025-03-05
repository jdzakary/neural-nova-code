import cProfile
import pstats
import time
from typing import Callable
import torch
import numpy as np
import cupy as cp
import numba as nb

from models import model_v1, ModelV2Wrapper

GU_FUNC_TARGET = 'parallel'


@nb.jit(nopython=True, nogil=True, cache=True)
def compute_status(square: np.ndarray) -> int:
    temp_status = np.zeros((8,))
    temp_status[0] = determine_status(square[0, :])
    temp_status[1] = determine_status(square[1, :])
    temp_status[2] = determine_status(square[2, :])
    temp_status[3] = determine_status(square[:, 0])
    temp_status[4] = determine_status(square[:, 1])
    temp_status[5] = determine_status(square[:, 2])
    temp_status[6] = determine_status(np.array([square[0, 0], square[1, 1], square[2, 2]]))
    temp_status[7] = determine_status(np.array([square[0, 2], square[1, 1], square[2, 0]]))
    if np.any(temp_status == 1):
        return 1
    if np.any(temp_status == -1):
        return -1
    if np.any(temp_status == 0):
        return 0
    if np.all(temp_status != 3):
        return 2
    return 3


@nb.jit(nopython=True, nogil=True, cache=True)
def determine_status(vector: np.ndarray) -> int:
    if np.all(vector == -1):
        return -1
    if np.all(vector == 1):
        return 1
    if np.all(vector != 0):
        return 2
    if np.any((vector == 2) | (vector == 3)):
        return 3
    if np.any(vector == -1) and np.any(vector == 1):
        return 3
    return 0


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :, :, :], nb.int8[:, :, :])],
    '(n,m,m,m,m)->(n,m,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET,
)
def apply_small(boards: np.ndarray, result: np.ndarray) -> np.ndarray:
    for i in range(boards.shape[0]):
        for a in range(3):
            for b in range(3):
                result[i, a, b] = compute_status(boards[i, a, b, :, :])


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :], nb.int8[:])],
    '(n,m,m)->(n)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET
)
def apply_big(boards: np.ndarray, result: np.ndarray) -> np.ndarray:
    for i in range(boards.shape[0]):
        result[i] = compute_status(boards[i, :, :])


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :, :, :], nb.int8[:, :, :], nb.int8[:, :, :], nb.int8[:, :, :, :, :])],
    '(n,m,m,m,m),(n,m,m),(n,x,y)->(n,m,m,m,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET
)
def compute_unexplored(
    boards: np.ndarray,
    status: np.ndarray,
    history: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(boards.shape[0]):
        a = history[i, -1][2]
        b = history[i, -1][3]
        x = np.zeros((3, 3, 3, 3), dtype=np.bool_)
        if status[i, a, b] == 0 or status[i, a, b] == 3:
            x[a, b, :, :] = boards[i, a, b, :, :] == 0
        else:
            for c in range(3):
                for d in range(3):
                    if status[i, c, d] == 0 or status[i, c, d] == 3:
                        x[c, d, :, :] = boards[i, c, d, :, :] == 0
        result[i] = x


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :, :, :], nb.int64[:, :], nb.int8[:, :, :, :, :])],
    '(n,m,m,m,m),(a,x)->(a,m,m,m,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET
)
def update_boards_explorer(
    boards: np.ndarray,
    unexplored: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(unexplored.shape[0]):
        idx = unexplored[i, 0]
        result[i] = boards[idx]
        a = unexplored[i, 1]
        b = unexplored[i, 2]
        y = unexplored[i, 3]
        x = unexplored[i, 4]
        result[i, a, b, y, x] = 1


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :], nb.int64[:, :], nb.int8[:], nb.int8[:, :, :])],
    '(n,x,y),(a,b),(c)->(a,c,y)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET
)
def update_history_explorer(
    history: np.ndarray,
    unexplored: np.ndarray,
    output_helper: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(unexplored.shape[0]):
        idx = unexplored[i, 0]
        result[i, :-1, :] = history[idx, :, :]
        result[i, -1, :] = unexplored[i, 1:]


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :, :, :], nb.int8[:, :], nb.int8[:, :, :, :, :])],
    '(n,m,m,m,m),(n,x)->(n,m,m,m,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET,
)
def update_boards_model(
    boards: np.ndarray,
    moves: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(boards.shape[0]):
        a = moves[i, 0]
        b = moves[i, 1]
        y = moves[i, 2]
        x = moves[i, 3]
        result[i] = boards[i]
        result[i, a, b, y, x] = -1


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :], nb.int8[:, :], nb.int8[:], nb.int8[:, :, :])],
    '(n,x,y),(n,y),(a)->(n,a,y)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET,
)
def update_history_model(
    history: np.ndarray,
    moves: np.ndarray,
    output_helper: int,
    result: np.ndarray,
) -> np.ndarray:
    for i in range(history.shape[0]):
        result[i, :-1, :] = history[i, :, :]
        result[i, -1, :] = moves[i, :]


def check_for_completed(
    game: np.ndarray,
    status: np.ndarray,
    boards: np.ndarray,
    history: np.ndarray,
    result: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = game == 0
    if len(result):
        result[0] = np.concatenate((result[0], game[~mask]), axis=0)
        result[1] = np.concatenate((result[1], boards[~mask]), axis=0)
        result[2].append(history[~mask])
    else:
        result.extend([game[~mask], boards[~mask], [history[~mask]]])
    return status[mask], boards[mask], history[mask]


def check_for_unique(
    status: np.ndarray,
    boards: np.ndarray,
    history: np.ndarray,
):
    return check_for_unique_v2(status, boards, history)


def check_for_unique_v1(
    status: np.ndarray,
    boards: np.ndarray,
    history: np.ndarray,
):
    unique, idx = np.unique(boards, axis=0, return_index=True)
    return status[idx], unique, history[idx]


def check_for_unique_v2(
    status: np.ndarray,
    boards: np.ndarray,
    history: np.ndarray,
):
    pack = apply_pack(boards)
    unique, idx = np.unique(pack, axis=0, return_index=True)
    unpack = apply_unpack(unique)
    idx = idx[::-1]
    unpack = unpack[::-1, :, :, :, :]
    return status[idx], unpack, history[idx]


def check_for_unique_v3(
    status: np.ndarray,
    boards: np.ndarray,
    history: np.ndarray,
):
    boards_gpu = cp.asarray(boards)
    unique, idx = cp.unique(boards_gpu, axis=0, return_index=True)
    unique = cp.asnumpy(unique)
    idx = cp.asnumpy(idx)
    return status[idx], unique, history[idx]


@nb.jit(nopython=True, nogil=True, cache=True)
def bit_pack(board: np.ndarray) -> np.ndarray:
    result = np.zeros((3,), dtype=np.uint64)
    for i, v in enumerate(board.flat):
        j = (i * 2) % 62
        current = i // 31
        if v == 1:
            result[current] += np.uint64(1) << np.uint(j)
        elif v == -1:
            result[current] += np.uint64(2) << np.uint(j)
    return result


@nb.jit(nopython=True, nogil=True, cache=True)
def bit_unpack(compress: np.ndarray) -> np.ndarray:
    result = np.zeros((81,), np.int8)
    for i in range(81):
        current = i // 31
        j = (i * 2) % 62
        target = (compress[current] >> np.uint(j)) & np.uint64(3)
        if target == 1:
            result[i] = 1
        elif target == 2:
            result[i] = -1
    return result.reshape((3, 3, 3, 3))


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.int8[:, :, :, :, :], nb.uint64[:, :])],
    '(n,m,m,m,m)->(n,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET,
)
def apply_pack(
    boards: np.ndarray,
    result: np.ndarray
) -> np.ndarray:
    for i in range(boards.shape[0]):
        result[i] = bit_pack(boards[i])


# noinspection PyTypeChecker
@nb.guvectorize(
    [(nb.uint64[:, :], nb.int8[:, :, :, :, :])],
    '(n,m)->(n,m,m,m,m)',
    nopython=True,
    cache=True,
    target=GU_FUNC_TARGET,
)
def apply_unpack(
    boards: np.ndarray,
    result: np.ndarray,
) -> np.ndarray:
    for i in range(boards.shape[0]):
        result[i] = bit_unpack(boards[i])


def explore(
    model: Callable,
    starting: np.ndarray = None,
    last_move: int = None,
    diagnostics: bool = False,
):
    result = []
    if starting is None:
        history = np.zeros((81, 1, 4), dtype=np.int8)
        boards = np.zeros((81, 3, 3, 3, 3), dtype=np.int8)
        for i in range(81):
            a = np.unravel_index(i, (3, 3, 3, 3))
            history[i] = np.array(a)
            boards[i, *a] = 1
            boards[i, *np.unravel_index(i, (3, 3, 3, 3))] = 1
        status = apply_small(boards)
    else:
        if starting.shape != (3, 3, 3, 3):
            raise ValueError('Starting board must be of the shape (3, 3, 3, 3)')
        turns = np.sum(starting != 0)
        boards = np.zeros((1, 3, 3, 3, 3), dtype=np.int8)
        history = np.zeros((1, 1, 4), dtype=np.int8)
        history[0, 0] = np.unravel_index(last_move, (3, 3, 3, 3))
        boards[0] = starting
        status = apply_small(boards)
        if turns % 2 == 0:
            print('Exploring before main loop!')
            unexplored = compute_unexplored(boards, status, history)
            unexplored = np.argwhere(unexplored)
            boards = update_boards_explorer(boards, unexplored)
            helper = np.zeros((history.shape[1] + 1), dtype=np.int8)
            history = update_history_explorer(history, unexplored, helper)
            status = apply_small(boards)
            status, boards, history = check_for_unique(status, boards, history)
            game = apply_big(status)
            status, boards, history = check_for_completed(game, status, boards, history, result)

    pr = cProfile.Profile()
    enabled = False
    while status.shape[0] > 0:
        if diagnostics and not enabled and history.shape[1] > 4:
            pr.enable()
            enabled = True
        t1 = time.perf_counter()
        moves = model(boards, status, history)
        boards = update_boards_model(boards, moves)
        helper = np.zeros(history.shape[1] + 1, dtype=np.int8)
        history = update_history_model(history, moves, helper)
        status = apply_small(boards)
        status, boards, history = check_for_unique(status, boards, history)
        game = apply_big(status)
        status, boards, history = check_for_completed(game, status, boards, history, result)
        t2 = time.perf_counter()
        print(
            f'O {history.shape[1]:>2} '
            f'{boards.shape[0]:>16,} '
            f'{result[0].size:>12,} '
            f'{t2 - t1:>10.3f} '
            f'{boards.nbytes / 1000_000:>8,.2f} MB'
        )
        if status.shape[0] == 0:
            break

        t1 = time.perf_counter()
        unexplored = compute_unexplored(boards, status, history)
        unexplored = np.argwhere(unexplored)
        boards = update_boards_explorer(boards, unexplored)
        helper = np.zeros((history.shape[1] + 1), dtype=np.int8)
        history = update_history_explorer(history, unexplored, helper)
        status = apply_small(boards)
        status, boards, history = check_for_unique(status, boards, history)
        game = apply_big(status)
        status, boards, history = check_for_completed(game, status, boards, history, result)
        t2 = time.perf_counter()
        print(
            f'X {history.shape[1]:>2} '
            f'{boards.shape[0]:>16,} '
            f'{result[0].size:>12,} '
            f'{t2 - t1:>10.3f} '
            f'{boards.nbytes / 1000_000:>8,.2f} MB'
        )

        if diagnostics:
            pr.disable()
            sort_by = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr).sort_stats(sort_by)
            ps.print_stats()

    return result


def create_starting_1() -> np.ndarray:
    starting: np.ndarray = np.load('start_board_1.npy')
    starting = starting.astype(np.int8)
    visualize_board(starting)
    return starting


def visualize_board(board: np.ndarray) -> None:
    """
    Visualize a board of shape (3, 3, 3, 3).
    :param board:
    :return:
    """
    for i in range(9):
        if i % 3 == 0:
            print('-' * 13)
        row = []
        a = i // 3
        y = i % 3
        for b in range(3):
            for x in range(3):
                row.append(board[a, b, y, x])
        for j, value in enumerate(row):
            match value:
                case 0:
                    char = ' '
                case 1:
                    char = 'X'
                case -1:
                    char = 'O'
                case _:
                    raise ValueError
            if j % 3 == 0:
                print('|', end='')
            print(char, end='')
        print('|')
    print('-' * 13)


def visualize_board_v2(board: np.ndarray) -> None:
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
    starting = create_starting_1()
    model_v2 = ModelV2Wrapper()
    result = explore(
        model=model_v2.run,
        starting=starting,
        last_move=2,
        diagnostics=False
    )
    game: np.ndarray = result[0]
    boards: np.ndarray = result[1]
    history: list[np.ndarray] = result[2]
    print(game.shape, boards.shape)
    print(f'Win X: {np.sum(game == 1) / len(game) * 100:.3f} %')
    print(f'Win O: {np.sum(game == -1) / len(game) * 100:.3f} %')
    print(f'Tie  : {np.sum((game == 2) | (game == 3)) / len(game) * 100:.3f} %')



if __name__ == '__main__':
    main()
