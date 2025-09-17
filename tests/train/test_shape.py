from game.model import Board
from train.model.common import StateEncoder, Critic, Actor
from train.model.sac import SAC
import numpy as np
from ..utils import get_dummy_board_states, get_dummy_move_matrices, get_dummy_state_embeddings
from game.constants import BLACK, WHITE
import torch


def shape_test(observed_shape, desired_shape):
    assert observed_shape == desired_shape, f"incorrect shape: {observed_shape} not {desired_shape}"


def test_board_encoder_shape():
    board = Board()
    board.reset()

    encoder = StateEncoder()

    state = board.get_state()
    assert state.shape == (1, 16, 8, 8), f"board state incorrect shape, {state.shape} not (1, 16, 8, 8)"
    embedding = encoder(state)
    assert embedding.shape == (1, 511), f"board state embedding incorrect shape, {embedding.shape} not (1, 511)"


def test_critic_shape():
    critic = Critic()

    embeddings = torch.zeros((32, 511))
    teams = torch.zeros((32, 1))
    select = torch.zeros((32, 1))
    target = torch.zeros((32, 1))
    promote = torch.zeros((32, 1)).long() - 1
    promote[[3, 5, 8]] = torch.tensor([0, 3, 2]).reshape(-1, 1).long()

    values = critic(embeddings, teams, select, target, promote)

    shape_test(values.shape, (32, 1))


def test_actor_shape():
    actor = Actor()
    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)
    teams = torch.tensor([WHITE.value, WHITE.value, BLACK.value, BLACK.value]).reshape(-1, 1).float()

    stacked = torch.hstack((dummy_embeddings, teams))
    shape_test(stacked.shape, (4, 512))

    projected = actor.body(stacked)
    shape_test(projected.shape, (4, 512))

    select, logp = actor.get_select(projected, dummy_move_matrices)
    shape_test(select.shape, (4, 1))
    assert logp.ndim == 0, "logp not 0 dim"

    target, logp = actor.get_target(projected, dummy_move_matrices, select)
    shape_test(target.shape, (4, 1))
    assert logp.ndim == 0, "logp not 0 dim"

    maybe_promotes = [[1, 2], [6], [], [34, 23, 21, 54]]
    promote, logp = actor.get_promote(projected, select, target, maybe_promotes)
    shape_test(promote.shape, (4, 1))
    assert logp.ndim == 0, "logp not 0 dim"
    
    select, target, promote, logp = actor(dummy_embeddings, teams, dummy_move_matrices, maybe_promotes)
    shape_test(select.shape, (4, 1))
    shape_test(target.shape, (4, 1))
    shape_test(promote.shape, (4, 1))
    assert logp.ndim == 0, "logp not 0 dim"


def test_actor_then_critic_shape():
    actor = Actor()
    critic = Critic()
    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)
    teams = torch.tensor([WHITE.value, WHITE.value, BLACK.value, BLACK.value]).reshape(-1, 1).float()

    stacked = torch.hstack((dummy_embeddings, teams))

    projected = actor.body(stacked)

    select, _ = actor.get_select(projected, dummy_move_matrices)

    target, _ = actor.get_target(projected, dummy_move_matrices, select)

    maybe_promotes = [[1, 2], [6], [], [34, 23, 21, 54]]
    promote, _ = actor.get_promote(projected, select, target, maybe_promotes)
    
    select, target, promote, _ = actor(dummy_embeddings, teams, dummy_move_matrices, maybe_promotes)

    values = critic(dummy_embeddings, teams, select, target, promote)

    shape_test(values.shape, (4, 1))



def test_action_sampling_shape():
    sac = SAC()

    move_matrices = np.zeros((2, 64, 64))
    select_distrs = np.zeros((2, 64))
    target_distrs = np.zeros((2, 64))

    select, target, logp = sac.get_action_samples(select_distrs, target_distrs, move_matrices)

    assert select.shape == (2,), f"incorrect select shape: {select.shape} not 2"
    assert target.shape == (2,), f"incorrect target shape: {target.shape} not 2"
    assert logp.shape == (2,), f"incorrect logp shape: {logp.shape} not 2"

