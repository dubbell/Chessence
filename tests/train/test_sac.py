import numpy as np
from train.model.sac import SAC




def test_get_action_sample():
    sac = SAC()

    move_matrix = np.zeros((64, 64))
    move_matrix[4, [5, 7, 10]] = 1
    move_matrix[8, [2, 15, 56]] = 1
    
    select_distr = np.ones(64)
    select_distr[4] = 2

    target_distr = np.ones(64)
    target_distr[5] = 2
    target_distr[7] = 3
    target_distr[10] = 4

    target_distr[2] = 2
    target_distr[15] = 3
    target_distr[56] = 4

    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (4, 10), f"wrong selection, {action[:2]} not (4, 10)"

    target_distr[10] = 1
    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (4, 7), f"wrong selection, {action[:2]} not (4, 7)"

    target_distr[7] = 1
    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (4, 5), f"wrong selection, {action[:2]} not (4, 5)"

    move_matrix[4] = 0

    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (8, 56), f"wrong selection, {action[:2]} not (8, 56)"
    
    target_distr[56] = 1
    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (8, 15), f"wrong selection, {action[:2]} not (8, 15)"

    target_distr[15] = 1
    action = sac.get_action_samples(select_distr, target_distr, move_matrix)
    assert action[:2] == (8, 2), f"wrong selection, {action[:2]} not (8, 2)"