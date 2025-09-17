import numpy as np
from train.model.sac import SAC
from train.experiments.train_sac import take_action
from game.model import Board
from game.constants import *


def test_take_action():
    board = Board()
    agent = SAC()

    board.reset()

    init_state = board.get_state()

    next_state, _, _, _, _ = take_action(board, agent, BLACK, None)

    assert (init_state != next_state).any(), "board state did not change"