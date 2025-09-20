from game.move_calc import get_moves
from game.model import Board, Move, Piece
from game.constants import *
from game.utils import other_team

from train.model.buffer import ReplayBuffer
from train.model.sac import SAC
from train.utils import DRAW, LOSS, CONTINUE, to_ndim

import git
import numpy as np
import mlflow
from datetime import datetime
from tqdm import tqdm
from typing import Mapping, List

import torch
torch.autograd.set_detect_anomaly(True)


def will_be_promoted(team, rank):
    """If the pawn will be promoted when moved."""
    return team == WHITE and rank == 1 or team == BLACK and rank == 6


def get_move_matrix(move_map : Mapping[Piece, List[Move]]):
    """Simplified representation of all available moves. 
       Row = coordinate of piece that is being moved (select). 
       Col = coordinate that the selected piece can be moved to (target).
       0 if move not available, 1 if move is available, 2 if available and will promote pawn"""
    
    move_matrix = np.zeros((64, 64))
    for piece, moves in move_map.items():
        available_select = np.ravel_multi_index(piece.coord, (8, 8))
        for move in moves:
            available_target = np.ravel_multi_index(move.to_coord, (8, 8))
            move_matrix[available_select, available_target] = \
                2 if piece.piece_type == PAWN and will_be_promoted(piece.team, piece.coord[0]) else 1

    return move_matrix


def take_action(board : Board, agent : SAC, team : Team, en_passant : np.array):
    """
    Let agent take action on board for the given team.
    Input: the board, the agent, and the agent's team
    Output: next state, move matrix, action, potential en passant location, result
    """

    current_state = board.get_state()
    move_map = get_moves(board, team, en_passant)
    # no moves available because checkmate
    if move_map is None:
        return current_state, None, None, None, LOSS
    # no moves available because draw
    elif not move_map:
        return current_state, None, None, None, DRAW

    move_matrix = get_move_matrix(move_map)
    action = agent.sample_actions(current_state, move_matrix, team.value, eval=True)[:3]  # ignore logp

    select = np.concatenate(np.unravel_index(action[0], (8, 8)))
    target = np.concatenate(np.unravel_index(action[1], (8, 8)))
    promote = action[2]

    selected_piece = board.coord_map[*select]
    en_passant = board.move_piece(selected_piece, Move(target), promote)
    
    next_state = board.get_state()

    return next_state, move_matrix, action, en_passant, CONTINUE



def get_run_name():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"SAC_{timestamp}"


def get_latest_commit_hash():
    repo = git.Repo(".", search_parent_directories=True)
    return repo.head.commit.hexsha


def start_training(config):
    replay_buffer = ReplayBuffer(batch_size = config["batch_size"])

    board = Board()
    board.reset()
    en_passant = None

    state = board.get_state()

    train_agent, fixed_agent = SAC(), SAC()
    white_agent, black_agent = train_agent, fixed_agent

    train_steps_remaining = 0
    game_count = 0
    current_team = WHITE

    # ensures that the black state transition is not added after first move
    # and so that the final game runs to completion
    first_step = True

    if config["enable_logging"]:
        mlflow.log_params({
            "learning_rate": train_agent.lr,
            "gamma": train_agent.lr,
            "commit": get_latest_commit_hash(),
            **config})
        
    pb = tqdm(total = config["total_games"])
    
    while game_count < config["total_games"]:
        # AGENT OPTIMIZATION
        while train_steps_remaining >= 1:
            train_agent.train_step(replay_buffer.sample_batch())
            train_steps_remaining -= 1

        # UPDATE FIXED AGENT AT SET INTERVALS
        if first_step and game_count % config["update_interval"] == 0 and game_count >= config["train_start"]:
            fixed_agent.load_state_dict(train_agent.state_dict())

        # TAKE ENVIRONMENT STEP
        next_state, move_matrix, action, en_passant, move_result = \
            take_action(board, white_agent if current_team == WHITE else black_agent, current_team, en_passant)
        
        # REPLAY BUFFER INSERTION
        if move_result == CONTINUE:
            replay_buffer.insert(state, move_matrix, *action, 0, current_team.value)
        else: 
            # if the game has ended, i.e. player had no moves available,
            # set rewards for previous moves
            replay_buffer.set_win_rewards(move_result)
            
            # reset board and switch teams
            board.reset()
            first_step = True
            state = board.get_state()
            white_agent, black_agent = black_agent, white_agent
            current_team = WHITE

            if config["enable_logging"]:
                is_white = train_agent == white_agent  # if train_agent is white
                mlflow.log_metric(
                    "white_win" if is_white else "black_win",
                    int(is_white != (current_team == WHITE)))

            game_count += 1
            pb.update()

            if game_count >= config["train_start"]:
                train_steps_remaining += config["train_steps_per_game"]

            continue

        # PREPARE FOR NEXT ITERATION
        state = next_state
        current_team = other_team(current_team)
        first_step = False
    
    pb.close()


def train_sac(config):
    if config["enable_logging"]:
        with mlflow.start_run(run_name=get_run_name(), log_system_metrics=True):
            start_training(config)
    else:
        start_training(config)
