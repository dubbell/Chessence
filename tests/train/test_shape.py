from game.model import Board
from train.model.common import StateEncoder



def test_board_encoder_shape():
    board = Board()
    board.reset()

    encoder = StateEncoder()

    state = board.get_state()
    print(state.shape)
    print()
    print()
    print()
    # print(encoder(state).shape)


