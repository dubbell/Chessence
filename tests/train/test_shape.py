from game.model import Board
from train.model.common import StateEncoder, Critic, Actor
from train.model.sac import SAC
import numpy as np


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

    embeddings = np.zeros((32, 511))
    teams = np.zeros(32)

    values = critic(embeddings, teams)

    assert values.shape == (32,), "critic output incorrect shape"


def test_actor_shape():
    actor = Actor()

    embeddings = np.zeros((32, 511))
    teams = np.zeros(32)

    select, target = actor(embeddings, teams)

    assert select.shape == (32, 64), "actor output incorrect select shape"
    assert target.shape == (32, 64), "actor output incorrect target shape"


def test_action_sampling_shape():
    sac = SAC()

    move_matrices = np.zeros((2, 64, 64))
    select_distrs = np.zeros((2, 64))
    target_distrs = np.zeros((2, 64))

    select, target, logp = sac.get_action_samples(select_distrs, target_distrs, move_matrices)

    assert select.shape == (2,), f"incorrect select shape: {select.shape} not 2"
    assert target.shape == (2,), f"incorrect target shape: {target.shape} not 2"
    assert logp.shape == (2,), f"incorrect logp shape: {logp.shape} not 2"

