from game.model import Board
from train.model.common import StateEncoder



def test_board_encoder_shape():
    board = Board()
    board.reset()

    encoder = StateEncoder()

    state = board.get_state()
    assert state.shape == (1, 16, 8, 8), f"board state incorrect shape, {state.shape} not (1, 16, 8, 8)"
    embedding = encoder(state)
    assert embedding.shape == (1, 511), f"board state embedding incorrect shape, {embedding.shape} not (1, 511)"