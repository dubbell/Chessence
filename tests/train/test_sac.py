from train.model.sac import SAC
from train.experiments import train_sac
from train.experiments.train_sac import take_action
from game.model import Board
from game.constants import *
import yaml


def test_take_action():
    board = Board()
    agent = SAC()

    board.reset()

    init_state = board.get_state()

    next_state, _, _, _, _ = take_action(board, agent, BLACK, None)

    assert (init_state != next_state).any(), "board state did not change"


def test_run_config():
    try:
        with open("config/test_config.yaml", "r") as file:
            config = yaml.safe_load(file)
    except:
        raise IOError("Config file not found.")
    
    train_sac.train_sac(config)