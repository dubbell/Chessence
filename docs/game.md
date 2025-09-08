# Chess Game Implementation

This document describes the chess game and logic implementation.

## Constants

- `Team` : enum
  - BLACK and WHITE
- `PieceType` : enum
  - KING, QUEEN, ROOK, BISHOP, KNIGHT and PAWN.
- `knight_diffs`
  - How the knight can move from a position.
  - 8 moves represented as 2D arrays, describing the relative movement from the knight location.

## Model

### `Piece`
- Coordinate, 2D numpy array.
- Piece type.
- Team, black or white.

### `Board`
- Attributes:
  - List of pieces.
  - Map of coordinates to pieces.
    - Used to lookup pieces given coordinate, e.g. in move generation for pawn to check surroundings.
  - Map from team and piecetype to list of pieces.
    - Used in move generation to lookup e.g. all knights of certain team, and generate their moves.
- Interface:
  - Lookup pieces.
  - Add pieces.
  - Remove pieces.
  - Move pieces.
  - get_state
    - Gets state for RL.

### `Move`
- Wrapper for 2D coordinate that a piece is moving to, stored as numpy array.
- Used in move calculation, as the returned moves is a dictionary mapping pieces to moves.

## King State

Implementation to check the "king state" for a certain team. A single function returning:
- 3x3 square around the king, describing which squares are controlled by the opponent.
- `pin_coords` - coordinates of pieces that are pinned to the king by opponent pieces.
- `pin_dirs` - directions from which the pinned pieces are pinned.

To do:
- De-spaghettification needed desperately.

## Move Calculation

Given the board, the team, and the possibility of en passant, generates all possible moves for the given team.

Returns:
- Map of `Piece` to `List[Move]`

To do:
- Also needs de-spaghettification.
