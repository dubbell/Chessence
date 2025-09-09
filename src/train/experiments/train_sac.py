from ..utils.buffer import get_buffer_loader
from ... import game
from ...game.constants import *
from ...game.utils import other_team
from ..model.sac import SAC
from ..model.common import StateEncoder

import numpy as np
import mlflow


def train_sac(horizon = 1000):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    replay_buffer, replay_loader = get_buffer_loader()

    board = game.utils.Board()
    board.reset()

    # 16x8x8 state
    board_state = board.get_state()

    current_team = WHITE
    en_passant = None

    sac_agent = SAC()



    for step in range(horizon):
        # map of piece to list of moves
        move_map = game.move_calc.get_moves(board, current_team, en_passant)

        # checkmate
        if move_map is None:
            board.reset()
            current_team = WHITE
            en_passant = None
            board_state = board.get_state()

        # simplified representation of all available moves
        # row represents coordinate of piece that is being moved (select)
        # col represents coordinate that the piece can be moved to (target)
        move_matrix = np.zeros((64, 64), dtype=bool)
        for piece, moves in move_map.items():
            available_select = np.ravel_multi_index(piece.coord, (8, 8))
            for move in moves:
                available_target = np.ravel_multi_index(move.to_coord, (8, 8))
                move_matrix[available_select, available_target] = True

        # sample select, target from agent, and unravel to actual coordinate representation
        action_select, action_target = sac_agent.sample_action(board_state, move_matrix)
        action_select = np.unravel_index(action_select, (8, 8))
        action_target = np.unravel_index(action_target, (8, 8))

        # piece to move
        selected_piece = board.coord_map[*action_select]
        en_passant = board.move_piece(selected_piece, game.model.Move(*action_target))

        current_team = other_team(current_team)