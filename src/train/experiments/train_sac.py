from ..utils.buffer import get_buffer_loader
from ... import game
from ..model.sac import SAC
from ..model.common import StateEncoder



import mlflow


def train_sac(horizon = 1000):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    replay_buffer, replay_loader = get_buffer_loader()

    board = game.utils.
    board.

    board_encoder = StateEncoder()
    sac_agent = SAC()



    for t in range(horizon):
