from game.move_calc import get_moves
from game.model import Board, Move, Piece
from game.constants import *
from game.utils import other_team

from train.model.buffer import ReplayBuffer
from train.model.sac import SAC
from train.utils import to_ndim

import git
import numpy as np
import mlflow
import datetime
from tqdm import tqdm

import torch
torch.autograd.set_detect_anomaly(True)


def take_action(current_state : np.array, move_matrix : np.array, board : Board, agent : SAC, team : Team):
    """
    Let agent take action on board for the given team.
    Input: the board, the agent, and the agent's team
    Output: next state, next move matrix, action
    """
    assert not (move_matrix == 0).all(), f"no moves available, \n{board}"
    action = agent.sample_actions(current_state, move_matrix)[:3]  # ignore logp

    select = np.concatenate(np.unravel_index(action[0], (8, 8)))
    target = np.concatenate(np.unravel_index(action[1], (8, 8)))
    promote = action[2]

    selected_piece = board.coord_map[*select]
    _, capture_reward = board.move_piece(Move(selected_piece, target, promote))
    
    next_team = other_team(team)
    next_state = board.get_state(next_team)
    next_move_matrix = get_moves(board, next_team)

    return next_state, next_move_matrix, action, capture_reward



def get_run_name():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"SAC_{timestamp}"


def get_latest_commit_hash():
    repo = git.Repo(".", search_parent_directories=True)
    return repo.head.commit.hexsha


def start_training(config):
    replay_buffer = ReplayBuffer(batch_size = config["batch_size"])

    board = Board()
    board.reset()

    current_team = WHITE
    state = board.get_state(current_team)
    move_matrix = get_moves(board, current_team)

    train_agent, fixed_agent = SAC(), SAC()
    white_agent, black_agent = train_agent, fixed_agent

    train_steps_remaining = 0
    game_count = 0

    if config["enable_logging"]:
        mlflow.log_params({
            "learning_rate": train_agent.lr,
            "gamma": train_agent.lr,
            "commit": get_latest_commit_hash(),
            **config})
        
    pb = tqdm(total = config["total_games"])

    train_loss_count, train_win_count, train_draw_count = 0, 0, 0

    game_steps = 0

    wins = np.zeros(50)
    
    while game_count < config["total_games"]:
        # AGENT OPTIMIZATION
        while train_steps_remaining >= 1:
            train_agent.train_step(replay_buffer.sample_batch())
            train_steps_remaining -= 1

        # UPDATE FIXED AGENT AT SET INTERVALS
        if (game_count - config["train_start"]) % config["update_interval"] == 0 and game_count >= config["train_start"]:
            fixed_agent.load_state_dict(train_agent.state_dict())
            train_loss_count, train_win_count, train_draw_count = 0, 0, 0
        
        if game_steps == 0:
            start_time = datetime.datetime.now()
        
        # TAKE ENVIRONMENT STEP
        current_agent = white_agent if current_team == WHITE else black_agent
        next_state, next_move_matrix, action, move_reward = take_action(state, move_matrix, board, current_agent, current_team)

        game_steps += 1

        # EVALUATE BOARD STATE AND MOVE, BUFFER INSERTION, LOGGING
        # checkmate or draw
        if next_move_matrix is None or (next_move_matrix == 0).all():
            # checkmate
            if next_move_matrix is None:
                wins[game_count % 50] = 0 if current_agent == train_agent else 1
                replay_buffer.insert(state, next_state, move_matrix, torch.zeros((64, 64)), *action, 50, True)
            # draw
            else:
                wins[game_count % 50] = 0.5
                replay_buffer.insert(state, next_state, move_matrix, next_move_matrix, *action, 0, True)
            
            game_count += 1
            pb.update()

            if config["enable_logging"] and game_count >= config["train_start"]:
                step = game_count - config["train_start"]
                if next_move_matrix is None:
                    if current_agent == train_agent:
                        train_loss_count += 1
                    else:
                        train_win_count += 1
                else:
                    train_draw_count += 1

                mlflow.log_metric("game_steps", game_steps, step=step)
                mlflow.log_metric("ms_per_game_step", np.round(((datetime.datetime.now() - start_time).microseconds / 1000) / game_steps, 2), step=step)
                mlflow.log_metric("draw", train_draw_count, step=step)
                mlflow.log_metric("win", train_win_count, step=step)
                mlflow.log_metric("loss", train_loss_count, step=step)
                mlflow.log_metric("win_diff", train_win_count - train_loss_count, step=step)
                mlflow.log_metric("win_rate", np.around(wins.mean(), 3), step=step)


            # reset board and switch teams
            board.reset()

            current_team = WHITE
            white_agent, black_agent = black_agent, white_agent
            state = board.get_state(current_team)
            move_matrix = get_moves(board, current_team)
            assert not (move_matrix == 0).all(), f"{board.check_50_move_rule()}, {board.check_threefold()}"

            if game_count >= config["train_start"]:
                train_steps_remaining += config["train_steps_per_game"]
            
            game_steps = 0
            
        # game continues
        else:  
            replay_buffer.insert(state, next_state, move_matrix, next_move_matrix, *action, move_reward, False)

            # prepare for next iteration
            state = next_state
            move_matrix = next_move_matrix
            current_team = other_team(current_team)
    
    pb.close()


def train_sac(config):
    if config["enable_logging"]:
        with mlflow.start_run(run_name=get_run_name(), log_system_metrics=True):
            start_training(config)
    else:
        start_training(config)
