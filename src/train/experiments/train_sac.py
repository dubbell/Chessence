from ..utils.buffer import get_buffer_loader
from ... import game
from ...game.constants import *
from ...game.utils import other_team
from ..model.sac import SAC
from ..model.common import StateEncoder

import numpy as np
import torch
import mlflow
import sys
import yaml



def train_sac(config):
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    replay_buffer, replay_loader = get_buffer_loader()

    board = game.utils.Board()
    board.reset()

    en_passant = None

    # 16x8x8 state
    board_state = board.get_state()
    # map of piece -> list of moves
    move_map = game.move_calc.get_moves(board, current_team, en_passant)

    train_agent = SAC()
    fixed_agent = SAC()

    train_team = WHITE
    current_team = WHITE

    for step in range(config.max_timesteps):
        # simplified representation of all available moves
        # row represents coordinate of piece that is being moved (select)
        # col represents coordinate that the piece can be moved to (target)
        move_matrix = np.zeros((64, 64), dtype=bool)
        for piece, moves in move_map.items():
            available_select = np.ravel_multi_index(piece.coord, (8, 8))
            for move in moves:
                available_target = np.ravel_multi_index(move.to_coord, (8, 8))
                move_matrix[available_select, available_target] = True

        # sample select and target action from agent
        if current_team == train_team:
            action = train_agent.sample_action(board_state, move_matrix)
        else:
            action = fixed_agent.sample_action(board_state, move_matrix)

        # unravel to coordinate representation
        action_select = np.unravel_index(action[0], (8, 8))
        action_target = np.unravel_index(action[1], (8, 8))

        # piece to move
        selected_piece = board.coord_map[*action_select]
        # move piece
        en_passant = board.move_piece(selected_piece, game.model.Move(*action_target))

        # next state for RL
        next_board_state = board.get_state()

        # switch team
        current_team = other_team(current_team)
        # get new moves
        move_map = game.move_calc.get_moves(board, current_team, en_passant)

        reward = 1
        replay_buffer.insert(board_state, next_board_state, reward, )

        # checkmate if no new moves
        if move_map is None:
            board.reset()
            current_team = WHITE
            en_passant = None
            board_state = board.get_state()

            # moves in reset board
            move_map = game.move_calc.get_moves(board, current_team, en_passant)
            reward = 1
        