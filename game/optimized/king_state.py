from move import Board
from constants import *

queen_corner_ = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1]])

rook_corner_ = np.array([
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]])

bishop_corner_ = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

knight_corner_ = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0]])


queen_side_ = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 0]])

rook_side_ = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [0, 1, 0]])

bishop_side_ = np.array([
    [0, 0, 0],
    [1, 0, 1],
    [0, 0, 0]])

knight_side_ = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 1]])


queen_contact = np.zeros((3, 3, 3, 3))
rook_contact = np.zeros((3, 3, 3, 3))
bishop_contact = np.zeros((3, 3, 3, 3))
knight_contact = np.zeros((3, 3, 3, 3))

for piece_contact, piece_corner, piece_side in zip(
        [queen_contact, rook_contact, bishop_contact, knight_contact],
        [queen_corner_, rook_corner_, bishop_corner_, knight_corner_],
        [queen_side_, rook_side_, bishop_side_, knight_side_]):
    
    for i, corner in enumerate([[0, 0], [2, 0], [2, 2], [0, 2]]):
        piece_contact[*corner] = np.rot90(piece_corner, i)
    for i, side in enumerate([[0, 1], [1, 0], [2, 1], [1, 2]]):
        piece_contact[*side] = np.rot90(piece_side, i)

pawn_contact = np.zeros((2, 3, 3, 3, 3))
for team, move_dir in zip([WHITE, BLACK], [-1, 1]):
    for rank in range(3):
        for file in range(3):
            if rank == 0 and team == WHITE or rank == 2 and team == BLACK:
                continue
            if file < 2:
                pawn_contact[team, rank, file, rank + move_dir, file + 1]
            if file > 0:
                pawn_contact[team, rank, file, rank + move_dir, file - 1]

contact = [None, queen_contact, rook_contact, bishop_contact, knight_contact]




def king_state(board : Board, team : int):
    """Returns the current state of the team's king on the board. 
       State is defined as (controlled, pins) where controlled are 
       the locations in the 3x3 king square that are controlled by the 
       opponent and pins are which pieces are pinned to the king."""
    
    controlled = np.zeros((3, 3))  # squares around king controlled by opponent
    pin_coords = []  # coordinates of team's pinned pieces
    pin_dirs = []  # direction from which the pieces are pinned

    neighbours = np.zeros((3, 3))
    
    king_coord = board.coords[team][KING]

    team_diffs = board.coords[team] - king_coord
    oppo_diffs = board.coords[int(not team)] - king_coord

    # populate neighbours with team pieces
    for diff in team_diffs:
        if diff.abs().sum() == 1 and (diff == 0).any() or diff.abs().sum() == 2:
            neighbours[*(diff + 1)] = -1

    # first pieces found in each diagonal line, only considering one direction
    diag_forward = [None for _ in range(10)]
    diag_forward_team = [None for _ in range(10)]
    diag_forward_dist = [10 for _ in range(10)]  # to keep track of distances
    # same for the other direction
    diag_backward = [None for _ in range(10)]
    diag_backward_team = [None for _ in range(10)]
    diag_backward_dist = [10 for _ in range(10)]  # to keep track of distances

    # potential diagonal pinners
    diag_pinners = [None for _ in range(4)]
    diag_pinners_team = [None for _ in range(4)]
    diag_pinners_dist = [10 for _ in range(4)]  # distance to potential pinners


    # first pieces found in each lateral line, only considering one direction
    lat_forward = [None for _ in range(6)]
    lat_forward_team = [None for _ in range(6)]
    lat_forward_dist = [10 for _ in range(6)]
    # same for the other direction
    lat_backward = [None for _ in range(6)]
    lat_backward_team = [None for _ in range(6)]
    lat_backward_dist = [10 for _ in range(6)]

    # potential lateral pinners
    lat_pinners = [None for _ in range(4)]
    lat_pinners_team = [None for _ in range(4)]
    lat_pinners_dist = [10 for _ in range(4)]

    # loop through every opponent piece to populate above lists
    for piece_type, piece_team, diff in zip(board.types[int(not team)], 
                                            [WHITE for _ in range(len(team_diffs))] + [BLACK for _ in range(len(oppo_diffs))], 
                                            np.vstack((team_diffs, oppo_diffs))):
        diff_abs = diff.abs()
        diff_sum = diff.sum()
        diff_dif = diff[0] - diff[1]

        # distance to piece
        distance = diff_abs.sum()

        # if in 3x3 square around the king
        if diff_abs.sum() == 1 and (diff == 0).any() or diff_abs.sum() == 2:
            neighbours[*(diff + 1)] = piece_type
        # if on lateral line
        elif (diff_abs.sum() <= 1).any():
            # indices for laterals on which the piece lies
            lat_indices = np.arange(6)[[diff[0] == x for x in range(-1, 2)] + [diff[1] == x for x in range(-1, 2)]]

            # loop through pieces's laterals
            for lat_index in lat_indices:
                # is the piece forward or backward on the lateral in relation to the king?
                is_forward = (diff[1] > 0) if lat_index < 3 else (diff[0] > 0)
                # current closest piece list for the lateral in given direction
                lat_pieces = lat_forward if is_forward else lat_backward
                # current distances used for determining closest pieces
                lat_dists = lat_forward_dist if is_forward else lat_backward_dist
                # team of each closest piece
                lat_team = lat_forward_team if is_forward else lat_backward_team
                # if it's on the king's lateral, then determine the index of that lateral in the pinner lists
                pin_index = 0 if lat_index == 1 and is_forward else \
                            1 if lat_index == 1 and not is_forward else \
                            2 if lat_index == 4 and is_forward else \
                            3 if lat_index == 4 and not is_forward else None
                
                # if this is the closest piece found so far
                if distance < lat_dists[lat_index]:
                    # if on king's lateral, also update second closest piece and distance, since it might be pinning a piece to the king
                    if pin_index is not None:
                        lat_pinners[pin_index] = lat_pieces[lat_index]
                        lat_pinners_team[pin_index] = piece_team
                        lat_pinners_dist[pin_index] = lat_dists[lat_index]
                    # update closest piece and distance
                    lat_pieces[lat_index] = piece_type
                    lat_team[lat_index] = piece_team
                    lat_dists[lat_index] = distance
                
                # if it's only the second closest piece, but it's on the king's lateral, then update second closest
                elif pin_index is not None and distance < lat_pinners_dist[pin_index]:
                    lat_pinners[pin_index] = piece_type
                    lat_pinners_team[pin_index] = piece_team
                    lat_pinners_dist[pin_index] = distance

        else:
            # indices for diagonals on which the piece lies (empty if it doesn't lie on any diagonal)
            diag_indices = np.arange(10)[[diff_sum == x for x in range(-2, 3)] + [diff_dif == x for x in range(-2, 3)]]

            # loop through piece's diagonals
            for diag_index in diag_indices:
                # is the piece forward or backward on the diagonal in relation to the king?
                is_forward = (diff_dif > 0) if diag_index < 5 else (diff_sum > 0) 
                # current closest piece list for the diagonal in given direction
                diag_pieces = diag_forward if is_forward else diag_backward
                # current distances used for determining closest pieces
                diag_dists = diag_forward_dist if is_forward else diag_backward_dist
                # team of each closest piece
                diag_team = diag_forward_team if is_forward else diag_backward_team
                # if it's on the king's diagonal, then determine the index of that diagonal in the pinner lists
                pin_index = 0 if diag_index == 2 and is_forward else \
                            1 if diag_index == 2 and not is_forward else \
                            2 if diag_index == 7 and is_forward else \
                            3 if diag_index == 7 and not is_forward else None

                # if this is the closest piece found so far
                if distance < diag_dists[diag_index]:
                    # if on king's diagonal, also update second closest piece and distance, since it might be pinning a piece to the king
                    if pin_index is not None:
                        diag_pinners[pin_index] = diag_pieces[diag_index]
                        diag_pinners_team[pin_index] = piece_team
                        diag_pinners_dist[pin_index] = diag_dists[diag_index]
                    # update closest piece and distance
                    diag_pieces[diag_index] = piece_type
                    diag_team[diag_index] = piece_team
                    diag_dists[diag_index] = distance

                # if it's only the second closest piece, and it's on the king's diagonal, then only update second closest
                elif pin_index is not None and distance < diag_pinners_dist[pin_index]:
                    diag_pinners[pin_index] = piece_type
                    diag_pinners_team[pin_index] = piece_team
                    diag_pinners_dist[pin_index] = distance
    
    # loop through neighbours to determine control
    for rank in range(3):
        for file in range(3):
            neighbour = neighbours[rank, file]

            # empty or team piece
            if neighbour <= 0:
                continue

            is_corner = not (rank == 1 or file == 1)

            if is_corner and neighbour in [QUEEN, ROOK]:
                # filter for blocking piec  es among neighbours
                contact_block = np.ones((3, 3))
                if neighbours[1, file] != 0:
                    contact_block[0 if rank == 2 else 2, file] = 0
                if neighbours[rank, 1] != 0:
                    contact_block[rank, 0 if file == 2 else 2] = 0

                if neighbours[rank, file] == PAWN:
                    controlled = np.logical_or(controlled, np.logical_and(pawn_contact[int(not team), rank, file], contact_block))
                else:
                    controlled = np.logical_or(controlled, np.logical_and(contact[neighbour][rank, file], contact_block))
            else:
                if neighbours[rank, file] == PAWN:
                    controlled = np.logical_or(controlled, pawn_contact[int(not team), rank, file])
                else:
                    controlled = np.logical_or(controlled, contact[neighbour][rank, file])

    # list of coordinates of the 3x3 square around the king
    king_square_coords = np.array([(rank, file) 
        for rank in range(-1, 2)
        for file in range(-1, 2)])

    # order in which the squares appear in the diagonals, concatenated
    king_square_orders = np.concatenate(np.array([
        np.arange(10)[
            [rank + file == x for x in range(-2, 3)] +
            [rank - file == x for x in range(-2, 3)]]
        for rank, file in king_square_coords]).T)

    # mapping from diag_index to the ordered steps in which the diagonal intersects the king square
    diag_index_to_steps = \
        [king_square_coords[np.arange(9)[king_square_orders == diag_index]]
         for diag_index in range(10)]
    
    diag_pin_index_to_dir = \
        [np.array([-1, 1] if index % 2 == 0 else [1, 1]) for index in range(4)]

    for diag_index in range(10):
        # forward diagonals
        pin_index = 0 if diag_index == 2 else \
                    2 if diag_index == 7 else None
        
        # first check pin
        if pin_index is not None \
                and diag_forward_team[diag_index] == team \
                and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                and diag_pinners_team[pin_index] == int(not team):
            pin_dir = diag_pin_index_to_dir[pin_index]
            pin_coords.append(king_coord + int(diag_forward_dist[diag_index] / 2) * pin_dir)
            pin_dirs.append(pin_dir)
        
        if diag_forward[diag_index] in [QUEEN, BISHOP]:
            for step in diag_index_to_steps[diag_index]:
                controlled[*step] = True
                if neighbours[*step] != 0:
                    break

        # backward diagonals
        pin_index = 1 if diag_index == 2 else \
                    3 if diag_index == 7 else None
        
        # first check pin
        if pin_index is not None \
                and diag_backward_team[diag_index] == team \
                and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                and diag_pinners_team[pin_index] == int(not team):
            pin_dir = diag_pin_index_to_dir[pin_index]
            pin_coords.append(king_coord + int(diag_backward_dist[diag_index] / 2) * pin_dir)
            pin_dirs.append(pin_dir)
        
        if diag_backward[diag_index] in [QUEEN, BISHOP]:
            # take steps backwards when checking backward diagonal
            for step in diag_index_to_steps[diag_index][::-1]:
                controlled[*step] = True
                if neighbours[*step] != 0:
                    break
    

    # remaining: knight control, opponent king control, lateral control and pins

    return controlled, (pin_coords, pin_dirs)