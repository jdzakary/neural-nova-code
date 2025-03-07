import time

import numpy as np
import torch
from torch import nn
from model.cnn import Actor
from validation.run import visualize_board
from game import Game

def load_model() -> nn.Module:
    state_dict = torch.load('results/state/exp8/batch_149_actor_o.pt')
    model = Actor()
    model.load_state_dict(state_dict)
    model.to(torch.device('cuda:0'))
    model.eval()
    return model


def play() -> None:
    print('''
    This is a very low-effort validation script. It is your responsibility
    to make sure you are doing valid moves, etc.
    ''')
    model = load_model()
    game = Game()
    with torch.no_grad():
        old_obs = torch.zeros((1, 9, 9), dtype=torch.float32).to(torch.device('cuda:0'))
        while not game.game_over:
            if game.turn == 'X':
                time.sleep(1)
                visualize_board(game.board)
                r = input('Please enter the row (0-8): ')
                c = input('Please enter the column (0-8): ')
                try:
                    game.move(int(r), int(c))
                except ValueError:
                    print('Illegal Move!')
            else:
                obs = torch.from_numpy(game.board.reshape((1, 9, 9))).float().to(torch.device('cuda:0'))
                stacked = torch.concat((obs, old_obs), dim=0)
                stacked = stacked.unsqueeze(0)
                logits = model(stacked).to('cpu').numpy().flatten()
                logits[game.constraint.flatten() == 0] = -np.inf
                idx = logits.argmax()
                old_obs = obs
                game.move(*np.unravel_index(idx, (9, 9)))
    visualize_board(game.board)
    if game.winner == 1:
        winner = 'X Wins'
    elif game.winner == -1:
        winner = 'O Wins'
    else:
        winner = 'Draw'
    print(f'Game Over! {winner}')


if __name__ == '__main__':
    play()
