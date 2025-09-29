from game.model import Board
from train.model.common import StateEncoder, Critic, Actor
from tests.utils import get_dummy_board_states, get_dummy_move_matrices, get_dummy_state_embeddings
from game.constants import BLACK, WHITE
import torch
import pytest


def shape_test(observed_shape, desired_shape):
    assert observed_shape == desired_shape, f"incorrect shape: {observed_shape} not {desired_shape}"


@pytest.mark.parametrize("eval,team", [(eval, team) for eval in [True, False] for team in [BLACK, WHITE]])
def test_board_encoder_shape(eval, team):
    board = Board()
    board.reset()

    encoder = StateEncoder()
    if eval:
        encoder.eval()
    else:
        encoder.train()

    state = board.get_state(team)
    shape_test(state.shape, (16, 8, 8))
    embedding = encoder.forward(state)
    shape_test(embedding.shape, (1, 512))

    if eval:
        encoder.train()
    else:
        encoder.eval()

    embeddings = encoder.forward(torch.zeros((4, 16, 8, 8)))
    shape_test(embeddings.shape, (4, 512))


@pytest.mark.parametrize("eval", [False, True])
def test_critic_shape(eval):
    critic = Critic()
    if eval:
        critic.eval()

    embeddings = torch.zeros((32, 512))
    select = torch.zeros(32).long()
    target = torch.zeros(32).long()
    promote = torch.zeros(32).long() - 1
    promote[[3, 5, 8]] = torch.tensor([0, 3, 2])

    values = critic.forward(embeddings, select, target, promote)

    shape_test(values.shape, (32,))


@pytest.mark.parametrize("eval", [False, True])
def test_actor_shape(eval):
    actor = Actor()
    if eval:
        actor.eval()

    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)

    projected = actor.body(dummy_embeddings)
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
    
    select, target, promote, logp = actor(dummy_embeddings, dummy_move_matrices)
    shape_test(select.shape, (4,))
    shape_test(target.shape, (4,))
    shape_test(promote.shape, (4,))
    shape_test(logp.shape, (4,))


@pytest.mark.parametrize("eval", [False, True])
def test_actor_then_critic_shape(eval):
    actor = Actor()
    critic = Critic()
    if eval:
        actor.eval()
        critic.eval()

    dummy_move_matrices = get_dummy_move_matrices(4) + 1
    dummy_embeddings = get_dummy_state_embeddings(4)

    projected = actor.body(dummy_embeddings)

    select, _ = actor.get_select(projected, dummy_move_matrices)

    target, _ = actor.get_target(projected, dummy_move_matrices, select)

    promote, _ = actor.get_promote(projected, dummy_move_matrices, select, target)
    
    select, target, promote, _ = actor(dummy_embeddings, dummy_move_matrices)

    values = critic(dummy_embeddings, select, target, promote)

    shape_test(values.shape, (4,))

