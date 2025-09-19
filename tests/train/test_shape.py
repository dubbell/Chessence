from game.model import Board
from train.model.common import StateEncoder, Critic, Actor
from tests.utils import get_dummy_board_states, get_dummy_move_matrices, get_dummy_state_embeddings
from game.constants import BLACK, WHITE
import torch


def shape_test(observed_shape, desired_shape):
    assert observed_shape == desired_shape, f"incorrect shape: {observed_shape} not {desired_shape}"


def test_board_encoder_shape():
    board = Board()
    board.reset()

    encoder = StateEncoder()

    state = board.get_state()
    shape_test(state.shape, (16, 8, 8))
    embedding = encoder(state)
    shape_test(embedding.shape, (1, 511))


def test_critic_shape():
    critic = Critic()

    embeddings = torch.zeros((32, 511))
    teams = torch.zeros(32)
    select = torch.zeros(32).long()
    target = torch.zeros(32).long()
    promote = torch.zeros(32).long() - 1
    promote[[3, 5, 8]] = torch.tensor([0, 3, 2])

    values = critic(embeddings, teams, select, target, promote)

    shape_test(values.shape, (32, 1))


def test_actor_shape():
    actor = Actor()
    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)
    teams = torch.tensor([WHITE.value, WHITE.value, BLACK.value, BLACK.value]).int()

    stacked = torch.hstack((dummy_embeddings, teams.reshape((-1, 1))))
    shape_test(stacked.shape, (4, 512))

    projected = actor.body(stacked)
    shape_test(projected.shape, (4, 512))

    select, logp = actor.get_select(projected, dummy_move_matrices)
    shape_test(select.shape, (4,))
    shape_test(logp.shape, (4,))

    target, logp = actor.get_target(projected, dummy_move_matrices, select)
    shape_test(target.shape, (4,))
    shape_test(logp.shape, (4,))

    promote, logp = actor.get_promote(projected, dummy_move_matrices, select, target)
    shape_test(promote.shape, (4,))
    shape_test(logp.shape, (4,))
    
    select, target, promote, logp = actor(dummy_embeddings, teams, dummy_move_matrices)
    shape_test(select.shape, (4,))
    shape_test(target.shape, (4,))
    shape_test(promote.shape, (4,))
    shape_test(logp.shape, (4,))


def test_actor_then_critic_shape():
    actor = Actor()
    critic = Critic()
    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)
    teams = torch.tensor([WHITE.value, WHITE.value, BLACK.value, BLACK.value]).int()

    stacked = torch.hstack((dummy_embeddings, teams.reshape((-1, 1))))

    projected = actor.body(stacked)

    select, _ = actor.get_select(projected, dummy_move_matrices)

    target, _ = actor.get_target(projected, dummy_move_matrices, select)

    promote, _ = actor.get_promote(projected, dummy_move_matrices, select, target)
    
    select, target, promote, _ = actor(dummy_embeddings, teams, dummy_move_matrices)

    values = critic(dummy_embeddings, teams, select, target, promote)

    shape_test(values.shape, (4,))

