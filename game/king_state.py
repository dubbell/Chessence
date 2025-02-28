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


queen_contact = np.zeros((3, 3, 3, 3))
rook_contact = np.zeros((3, 3, 3, 3))
bishop_contact = np.zeros((3, 3, 3, 3))

for piece_contact, piece_corner, piece_side in zip(
        [queen_contact, rook_contact, bishop_contact],
        [queen_corner_, rook_corner_, bishop_corner_],
        [queen_side_, rook_side_, bishop_side_]):
    
    for i, corner in enumerate([[0, 0], [2, 0], [2, 2], [0, 2]]):
        piece_contact[*corner] = np.rot90(piece_corner, i)
    for i, side in enumerate([[0, 1], [1, 0], [2, 1], [1, 2]]):
        piece_contact[*side] = np.rot90(piece_side, i)

contact = [None, queen_contact, rook_contact, bishop_contact]

# map from opponent king diff to control matrix
diff_to_king_control = {}
edge_diffs_ = [[rank_diff, file_diff] for rank_diff in [-2, 2] for file_diff in range(-2, 3)] + \
             [[rank_diff, file_diff] for rank_diff in range(-1, 2) for file_diff in [-2, 2]]

for edge_rank_, edge_file_ in edge_diffs_:
    controlled = np.zeros((3, 3))
    for control_rank_diff_ in range(-1, 2):
        for control_file_diff_ in range(-1, 2):
            if abs(edge_rank_ - control_rank_diff_) <= 1 and abs(edge_file_ - control_file_diff_) <= 1:
                controlled[control_rank_diff_ + 1, control_file_diff_ + 1] = 1
    
    diff_to_king_control[(edge_rank_, edge_file_)] = controlled


def get_king_state(board : Board, team : Team):
    """Returns the current state of the team's king on the board. 
       State is defined as (controlled, pin_coords, pin_dirs) where `controlled` are 
       the locations in the 3x3 king square that are controlled by the 
       opponent, `pin_coords` are coordinates of the pieces that are pinned to the king,
       and `pin_dirs` are the directions from which these pieces are pinned."""
    
    controlled = np.zeros((3, 3), dtype=int)  # squares around king controlled by opponent
    pin_coords = []  # coordinates of team's pinned pieces
    pin_dirs = []  # direction from which the pieces are pinned

    # -1 if team piece, 0 if empty, otherwise opponent piece_type (0 piece_type impossible because king)
    neighbours = np.zeros((3, 3), dtype=int)
    
    king_pieces = board.of_team_and_type(team, KING)
    if len(king_pieces) == 0:
        return controlled, pin_coords, pin_dirs

    king_coord = king_pieces[0].coord

    # first pieces found in each diagonal line, only considering one direction
    diag_forward_diff = [None for _ in range(10)]
    diag_forward_type = [None for _ in range(10)]
    diag_forward_team = [None for _ in range(10)]
    diag_forward_dist = [15 for _ in range(10)]  # to keep track of distances
    # same for the other direction
    diag_backward_diff = [None for _ in range(10)]
    diag_backward_type = [None for _ in range(10)]
    diag_backward_team = [None for _ in range(10)]
    diag_backward_dist = [15 for _ in range(10)]  # to keep track of distances

    # potential diagonal pinners
    diag_pinners = [None for _ in range(4)]
    diag_pinners_team = [None for _ in range(4)]
    diag_pinners_dist = [15 for _ in range(4)]  # distance to potential pinners


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
    for piece in board.pieces:
        diff = piece.coord - king_coord
        piece_type = piece.piece_type

        diff_abs = np.abs(diff)
        diff_sum = diff.sum()
        diff_dif = diff[0] - diff[1]

        # distance to piece
        distance = diff_abs.sum()

        # ignore if it's the king itself
        if distance == 0:
            continue

        # if it's an opponent pawn
        if piece_type == PAWN and piece.team == other_team(team):
            pawn_coord = diff + 1  # convert diff to control matrix coordinates
            pawn_control_coords = pawn_coord + [[1 if team == WHITE else -1, file_diff] for file_diff in [-1, 1]]
            for pawn_control_coord in pawn_control_coords:
                if (pawn_control_coord <= 2).all() and (pawn_control_coord >= 0).all():
                    controlled[*pawn_control_coord] = 1

        # if it's an opponent knight
        if piece_type == KNIGHT and piece.team == other_team(team):
            # coordinates in the `controlled` space controlled by the knight
            for knight_control_coord in diff + 1 + knight_diffs:
                if (knight_control_coord >= 0).all() and (knight_control_coord <= 2).all():
                    controlled[*knight_control_coord] = 1

        # if in 3x3 square around the king
        if (diff_abs <= 1).all():
            neighbours[*(diff + 1)] = -1 if piece.team == team else piece_type.value
            continue  # handle neighbor pieces differently
        
        # if it's the opponent king
        if piece_type == KING:
            opponent_king_control = diff_to_king_control.get(tuple(diff))
            if opponent_king_control is not None:
                controlled = np.logical_or(controlled, opponent_king_control)

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
                    lat_pinners_team[pin_index] = lat_team[lat_index]
                    lat_pinners_dist[pin_index] = lat_dists[lat_index]
                # update closest piece and distance
                lat_coords[lat_index] = diff
                lat_types[lat_index] = piece_type
                lat_team[lat_index] = piece.team
                lat_dists[lat_index] = distance
            
            # if it's only the second closest piece, but it's on the king's lateral, then update second closest i.e. pinner
            elif pin_index is not None and distance < lat_pinners_dist[pin_index]:
                lat_pinners[pin_index] = piece_type
                lat_pinners_team[pin_index] = piece.team
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
                    diag_pinners_team[pin_index] = diag_team[diag_index]
                    diag_pinners_dist[pin_index] = diag_dists[diag_index]
                # update closest piece and distance
                diag_coords[diag_index] = diff
                diag_pieces[diag_index] = piece_type
                diag_team[diag_index] = piece.team
                diag_dists[diag_index] = distance

            # if it's only the second closest piece, and it's on the king's diagonal, then only update second closest i.e. pinner
            elif pin_index is not None and distance < diag_pinners_dist[pin_index]:
                diag_pinners[pin_index] = piece_type
                diag_pinners_team[pin_index] = piece.team
                diag_pinners_dist[pin_index] = distance


    # loop through neighbours to determine control
    for rank in range(3):
        for file in range(3):
            neighbour = neighbours[rank, file]

            # empty or team piece or pawn or knight
            if neighbour <= 0 or neighbour == PAWN.value or neighbour == KNIGHT.value:
                continue

            is_corner = not (rank == 1 or file == 1)

            if is_corner and neighbour in [QUEEN.value, ROOK.value]:
                # filter for neighbours that potentially blocks queen/rook sight along edges of 3x3 square
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
    
    # direction of pin by diagonal pin index
    diag_pin_index_to_dir = \
        np.array([[1, -1], [-1, 1], [1, 1], [-1, -1]])
    
    # direction of pin by lateral pin index
    lat_pin_index_to_dir = \
        np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])

    # CHECK DIAGONALS
    for diag_index in range(10):
        for check_direction in range(2):
            # CHECK FORWARD DIAGONALS
            if check_direction == 0:  
                steps = diag_index_to_steps[diag_index][::-1]
                pin_index = 0 if diag_index == 2 else \
                            2 if diag_index == 7 else None
                diag_team = diag_forward_team[diag_index]
                diag_type = diag_forward_type[diag_index]
                diag_diff = diag_forward_diff[diag_index]
            # CHECK BACKWARD DIAGONALS
            else:  
                steps = diag_index_to_steps[diag_index]
                pin_index = 1 if diag_index == 2 else \
                            3 if diag_index == 7 else None
                diag_team = diag_backward_team[diag_index]
                diag_type = diag_backward_type[diag_index]
                diag_diff = diag_backward_diff[diag_index]
        
            # if first piece on diagonal is not team
            if diag_team == other_team(team):
                # if piece is of type that controls the diagonal
                if diag_type in [QUEEN, BISHOP]:
                    # if neighbour blocks
                    if neighbours[*steps[0]] != 0:
                        controlled[*steps[0]] = 1
                        # if pin_index and of team, then it is pinned
                        if pin_index is not None and neighbours[*steps[0]] == -1:
                            pin_coords.append(king_coord + (steps[0] - [1, 1]))
                            pin_dirs.append(diag_pin_index_to_dir[pin_index])
                    else:  # if no piece blocks, then diagonal piece controls entire diagonal
                        for step in steps:
                            controlled[*step] = 1
                
                # add diagonal pins on opponent pawns since it matters for en passant
                elif diag_type == PAWN \
                        and pin_index is not None \
                        and diag_pinners_team[pin_index] == other_team(team) \
                        and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                        and neighbours[*steps[0]] == 0:
                    pin_coords.append(king_coord + diag_diff)
                    pin_dirs.append(diag_pin_index_to_dir[pin_index])


            # if first piece on diagonal is team and is being pinned, then add pin
            elif pin_index is not None \
                    and diag_pinners_team[pin_index] == other_team(team) \
                    and diag_pinners[pin_index] in [QUEEN, BISHOP] \
                    and neighbours[*steps[0]] == 0:
                pin_coords.append(king_coord + diag_diff)
                pin_dirs.append(diag_pin_index_to_dir[pin_index])



    # CHECK LATERALS
    for lat_index in range(6):
        for check_direction in range(2):
            # CHECK FORWARD LATERALS
            if check_direction == 0:
                steps = lat_index_to_steps[lat_index][::-1]
                pin_index = 0 if lat_index == 1 else \
                            2 if lat_index == 4 else None
                lat_team = lat_forward_team[lat_index]
                lat_type = lat_forward_type[lat_index]
                lat_diff = lat_forward_diff[lat_index]
            # CHECK BACKWARD LATERALS
            else:
                steps = lat_index_to_steps[lat_index]
                pin_index = 1 if lat_index == 1 else \
                            3 if lat_index == 4 else None
                lat_team = lat_backward_team[lat_index]
                lat_type = lat_backward_type[lat_index]
                lat_diff = lat_backward_diff[lat_index]
            
            # if first piece on lateral is not team
            if lat_team == other_team(team):
                # if piece is of type that controls the lateral
                if lat_type in [QUEEN, ROOK]:
                    # if neighbour blocks
                    if neighbours[*steps[0]] != 0:
                        controlled[*steps[0]] = 1
                        # if pin_index and of team, then it is pinned
                        if pin_index is not None and neighbours[*steps[0]] == -1:
                            pin_coords.append(king_coord + (steps[0] - [1, 1]))
                            pin_dirs.append(lat_pin_index_to_dir[pin_index])
                    else:  # if no piece blocks, then lateral piece controls entire lateral
                        for step in steps:
                            controlled[*step] = 1

            # if first piece on diagonal is team and is being pinned, then add pin
            elif pin_index is not None \
                    and lat_pinners_team[pin_index] == other_team(team) \
                    and lat_pinners[pin_index] in [QUEEN, ROOK] \
                    and neighbours[*steps[0]] == 0:
                pin_coords.append(king_coord + lat_diff)
                pin_dirs.append(lat_pin_index_to_dir[pin_index])
        

    


    return controlled, pin_coords, pin_dirs

