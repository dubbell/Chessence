from model import Board
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

contact = [None, queen_contact, rook_contact, bishop_contact, knight_contact]




def king_state(board : Board, team : int):
    """Returns the current state of the team's king on the board. 
       State is defined as (controlled, pin_coords, pin_dirs) where `controlled` are 
       the locations in the 3x3 king square that are controlled by the 
       opponent, `pin_coords` are coordinates of the pieces that are pinned to the king,
       and `pin_dirs` are the directions from which these pieces are pinned."""
    
    controlled = np.zeros((3, 3), dtype=int)  # squares around king controlled by opponent
    pin_coords = []  # coordinates of team's pinned pieces
    pin_dirs = []  # direction from which the pieces are pinned

    # -1 if team piece, 0 if empty, otherwise opponent piece_type (0 piece_type impossible because king)
    neighbours = np.zeros((3, 3))
    
    king_coord = board.coords[team][KING]

    team_diffs = board.coords[team] - king_coord
    oppo_diffs = board.coords[int(not team)] - king_coord

    # first pieces found in each diagonal line, only considering one direction
    diag_forward_diff = [None for _ in range(10)]
    diag_forward_type = [None for _ in range(10)]
    diag_forward_team = [None for _ in range(10)]
    diag_forward_dist = [10 for _ in range(10)]  # to keep track of distances
    # same for the other direction
    diag_backward_diff = [None for _ in range(10)]
    diag_backward_type = [None for _ in range(10)]
    diag_backward_team = [None for _ in range(10)]
    diag_backward_dist = [10 for _ in range(10)]  # to keep track of distances

    # potential diagonal pinners
    diag_pinners = [None for _ in range(4)]
    diag_pinners_team = [None for _ in range(4)]
    diag_pinners_dist = [10 for _ in range(4)]  # distance to potential pinners


    # first pieces found in each lateral line, only considering one direction
    lat_forward_diff = [None for _ in range(10)]
    lat_forward_type = [None for _ in range(6)]
    lat_forward_team = [None for _ in range(6)]
    lat_forward_dist = [10 for _ in range(6)]
    # same for the other direction
    lat_backward_diff = [None for _ in range(10)]
    lat_backward_type = [None for _ in range(6)]
    lat_backward_team = [None for _ in range(6)]
    lat_backward_dist = [10 for _ in range(6)]

    # potential lateral pinners
    lat_pinners = [None for _ in range(4)]
    lat_pinners_team = [None for _ in range(4)]
    lat_pinners_dist = [10 for _ in range(4)]

    # loop through every piece to populate above lists
    for piece_type, piece_team, diff in zip(np.concatenate((board.types[team], board.types[int(not team)])), 
                                            [team for _ in range(len(team_diffs))] + [int(not team) for _ in range(len(oppo_diffs))], 
                                            np.vstack((team_diffs, oppo_diffs))):
        diff_abs = np.abs(diff)
        diff_sum = diff.sum()
        diff_dif = diff[0] - diff[1]

        # distance to piece
        distance = diff_abs.sum()

        # ignore if it's the king itself
        if distance == 0:
            continue
        # if in 3x3 square around the king
        if distance == 1 or distance == 2 and (diff_abs <= 1).all():
            neighbours[*(diff + 1)] = -1 if piece_team == team else piece_type
            

        # IF ON A LATERAL LINE -------------

        # indices for laterals on which the piece lies
        lat_indices = np.arange(6)[[diff[0] == x for x in range(-1, 2)] + [diff[1] == x for x in range(-1, 2)]]

        # loop through piece's laterals
        for lat_index in lat_indices:
            # is the piece forward or backward on the lateral in relation to the king?
            is_forward = (diff[1] > 0) if lat_index < 3 else (diff[0] > 0)
            # current closest piece coordinates for the lateral in given direction
            lat_coords = lat_forward_diff if is_forward else lat_backward_diff
            # current closest piece type list 
            lat_types = lat_forward_type if is_forward else lat_backward_type
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
                # if on king's lateral, also update second closest piece and distance, since it might be pinning the current piece to the king
                if pin_index is not None:
                    lat_pinners[pin_index] = lat_types[lat_index]
                    lat_pinners_team[pin_index] = piece_team
                    lat_pinners_dist[pin_index] = lat_dists[lat_index]
                # update closest piece and distance
                lat_coords[lat_index] = diff
                lat_types[lat_index] = piece_type
                lat_team[lat_index] = piece_team
                lat_dists[lat_index] = distance
            
            # if it's only the second closest piece, but it's on the king's lateral, then update second closest i.e. pinner
            elif pin_index is not None and distance < lat_pinners_dist[pin_index]:
                lat_pinners[pin_index] = piece_type
                lat_pinners_team[pin_index] = piece_team
                lat_pinners_dist[pin_index] = distance


        # IF ON A DIAGONAL LINE -------------
    
        # indices for diagonals on which the piece lies (empty if it doesn't lie on any diagonal)
        diag_indices = np.arange(10)[[diff_sum == x for x in range(-2, 3)] + [diff_dif == x for x in range(-2, 3)]]

        # loop through piece's diagonals
        for diag_index in diag_indices:
            # is the piece forward or backward on the diagonal in relation to the king?
            is_forward = (diff_dif > 0) if diag_index < 5 else (diff_sum > 0) 
            # current closest piece coords for the diagonal in the given direction
            diag_coords = diag_forward_diff if is_forward else diag_backward_diff
            # current closest piece list
            diag_pieces = diag_forward_type if is_forward else diag_backward_type
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
                # if on king's diagonal, also update second closest piece and distance, since it might be pinning the piece to the king
                if pin_index is not None:
                    diag_pinners[pin_index] = diag_pieces[diag_index]
                    diag_pinners_team[pin_index] = piece_team
                    diag_pinners_dist[pin_index] = diag_dists[diag_index]
                # update closest piece and distance
                diag_coords[diag_index] = diff
                diag_pieces[diag_index] = piece_type
                diag_team[diag_index] = piece_team
                diag_dists[diag_index] = distance

            # if it's only the second closest piece, and it's on the king's diagonal, then only update second closest i.e. pinner
            elif pin_index is not None and distance < diag_pinners_dist[pin_index]:
                diag_pinners[pin_index] = piece_type
                diag_pinners_team[pin_index] = piece_team
                diag_pinners_dist[pin_index] = distance
    
    # check control by opponent pawns
    for pawn_diff in oppo_diffs[board.type_locs[int(not team), -1]:]:
        pawn_coord = pawn_rank, pawn_file = pawn_diff + 1  # convert diff to control matrix coordinates
        pawn_control_coords = pawn_coord + [[pawn_rank, pawn_file + file_diff] for file_diff in [-1, 1]]
        for pawn_control_coord in pawn_control_coords:
            if (pawn_control_coord <= 2).all() and (pawn_control_coord >= 0).all():
                controlled[*pawn_control_coord] = 1

    # loop through neighbours to determine control
    for rank in range(3):
        for file in range(3):
            neighbour = neighbours[rank, file]

            # empty or team piece or pawn
            if neighbour <= 0 or neighbour == PAWN:
                continue

            is_corner = not (rank == 1 or file == 1)

            if is_corner and neighbour in [QUEEN, ROOK]:
                # filter for blocking pieces among neighbours
                contact_block = np.ones((3, 3))
                if neighbours[1, file] != 0:
                    contact_block[0 if rank == 2 else 2, file] = 0
                if neighbours[rank, 1] != 0:
                    contact_block[rank, 0 if file == 2 else 2] = 0

                controlled = np.logical_or(controlled, np.logical_and(contact[neighbour][rank, file], contact_block))
            else:
                controlled = np.logical_or(controlled, contact[neighbour][rank, file])

    # list of coordinates of the 3x3 square around the king
    king_square_coords = np.array([(rank, file) 
        for rank in range(-1, 2)
        for file in range(-1, 2)])
    
    # mapping of diagonal index to steps in 3x3 square
    diag_index_to_steps = [[] for _ in range(10)]

    # mapping of lateral index to steps in 3x3 square
    lat_index_to_steps = [[] for _ in range(6)]

    # populate index to steps lists
    for king_square_coord in king_square_coords:
        king_square_rank, king_square_file = king_square_coord
        diag_index_to_steps[king_square_rank + king_square_file + 2].append(king_square_coord + 1)
        diag_index_to_steps[king_square_rank - king_square_file + 7].append(king_square_coord + 1)

        lat_index_to_steps[king_square_rank + 1].append(king_square_coord + 1)
        lat_index_to_steps[king_square_file + 4].append(king_square_coord + 1)
    
    # direction of pin diagonal index
    diag_pin_index_to_dir = \
        [np.array([-1, 1] if index % 2 == 0 else [1, 1]) for index in range(4)]
    
    lat_pin_index_to_dir = \
        [np.array([0, 1] if index % 2 == 0 else [1, 0]) for index in range(4)]

    # CHECK DIAGONALS
    for diag_index in range(10):
        # FORWARD DIAGONALS
        pin_index = 0 if diag_index == 2 else \
                    2 if diag_index == 7 else None
        
        # if pin diagonal, and pinned is team, and pinner is opponent QUEEN/BISHOP
        if pin_index is not None \
                and diag_forward_team[diag_index] == team \
                and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                and diag_pinners_team[pin_index] == int(not team):
            pin_dir = diag_pin_index_to_dir[pin_index]
            pin_diff = diag_forward_diff[diag_index]
            pin_coords.append(king_coord + pin_diff)
            pin_dirs.append(pin_dir)
            if (np.abs(pin_diff) <= 1).all():  # if pinned piece is next to king, then set controlled to true (workaround)
                controlled[*(pin_diff + 1)] = 1
        
        # if first piece on diagonal is opponent QUEEN/BISHOP, then update control along diagonal
        elif diag_forward_team[diag_index] == int(not team) \
                and diag_forward_type[diag_index] in [QUEEN, BISHOP]:
            for step in diag_index_to_steps[diag_index][::-1]:
                controlled[*step] = 1
                if neighbours[*step] != 0:  # break if piece blocks
                    break

        # BACKWARD DIAGONALS
        pin_index = 1 if diag_index == 2 else \
                    3 if diag_index == 7 else None
        
        # first check pin
        if pin_index is not None \
                and diag_backward_team[diag_index] == team \
                and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                and diag_pinners_team[pin_index] == int(not team):
            pin_dir = diag_pin_index_to_dir[pin_index]
            pin_diff = diag_backward_diff[diag_index]
            pin_coords.append(king_coord + pin_diff)
            pin_dirs.append(-pin_dir)
            if (np.abs(pin_diff) <= 1).all():  # if pinned piece is next to king, then set controlled to true (workaround)
                controlled[*(pin_diff + 1)] = 1

        
        # if first piece on diagonal is opponent QUEEN/BISHOP, then update control along diagonal
        elif diag_backward_team[diag_index] == int(not team) \
                and diag_backward_type[diag_index] in [QUEEN, BISHOP]:
            # take steps backwards when checking backward diagonal
            for step in diag_index_to_steps[diag_index]:
                controlled[*step] = 1
                if neighbours[*step] != 0:
                    break


    # CHECK LATERALS
    for lat_index in range(6):
        # FORWARD LATERALS
        pin_index = 0 if lat_index == 1 else \
                    2 if lat_index == 4 else None
        
        # first check pin
        if pin_index is not None \
                and lat_forward_team[lat_index] == team \
                and lat_pinners[pin_index] in [QUEEN, BISHOP] \
                and lat_pinners_team[pin_index] == int(not team):
            pin_dir = lat_pin_index_to_dir[pin_index]
            pin_diff = lat_forward_diff[lat_index]
            pin_coords.append(king_coord + pin_diff)
            pin_dirs.append(pin_dir)
            if (np.abs(pin_diff) <= 1).all():  # if pinned piece is next to king, then set controlled to true (workaround)
                controlled[*(pin_diff + 1)] = 1
        
        elif lat_forward_team[lat_index] == int(not team) \
                and lat_forward_type[lat_index] in [QUEEN, ROOK]:
            for step in lat_index_to_steps[lat_index]:
                controlled[*step] = 1
                if neighbours[*step] != 0:
                    break

        # BACKWARD LATERALS
        pin_index = 1 if lat_index == 1 else \
                    3 if lat_index == 4 else None
        
        # first check pin
        if pin_index is not None \
                and lat_backward_team[lat_index] == team \
                and lat_pinners[pin_index] in [QUEEN, ROOK] \
                and lat_pinners_team[pin_index] == int(not team):
            pin_dir = lat_pin_index_to_dir[pin_index]
            pin_diff = lat_backward_diff[lat_index]
            pin_coords.append(king_coord + pin_diff)
            pin_dirs.append(pin_dir)
            if (np.abs(pin_diff) <= 1).all():  # if pinned piece is next to king, then set controlled to true (workaround)
                controlled[*(pin_diff + 1)] = 1
        
        if lat_backward_team[lat_index] == int(not team) \
                and lat_backward_type[lat_index] in [QUEEN, ROOK]:
            # take steps backwards when checking backward diagonal
            for step in lat_index_to_steps[lat_index][::-1]:
                controlled[*step] = 1
                if neighbours[*step] != 0:
                    break

    # remaining: knight control, opponent king control

    return controlled, pin_coords, pin_dirs



# board = Board()
# board.add_piece(KING, WHITE, 4, 4)
# board.add_piece(PAWN, WHITE, 3, 3)
# board.add_piece(BISHOP, BLACK, 1, 1)
# board.add_piece(BISHOP, BLACK, 2, 1)
# controlled, pin_coords, pin_dirs = king_state(board, WHITE)

# print(board)
# print()
# print(controlled)
# print()

# for coord, dir in zip(pin_coords, pin_dirs):
#     print(coord, dir)

