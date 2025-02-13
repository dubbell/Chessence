from constants import *
from typing import List
import numpy as np


class Board:
    coords : List[np.array] # list of 2 arrays, each (X, 2) of corresponding piece coords
    types : List[np.array]  # list of 2 arrays, each of length X representing corresponding piece type
    type_locs = np.array # (2, 6) array, each element representing the starting index of piece in types

    def __init__(self):
        self.coords = [np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=int)]
        self.types = [[], []]
        self.type_locs = np.zeros((2, 6), dtype=int)
    

    def __repr__(self):
        board = np.array([["  " for _ in range(8)] for _ in range(8)])
        for team in [WHITE, BLACK]:
            for coord, type in zip(self.coords[team], self.types[team]):
                name = ("W" if team == WHITE else "B") + \
                       ("K" if type == KING else 
                        "Q" if type == QUEEN else
                        "R" if type == ROOK else
                        "B" if type == BISHOP else 
                        "N" if type == KNIGHT else "P")
                
                board[*coord] = name

        return "[" + "]\n[".join([" ".join(rank) for rank in board]) + "]"


    def piece_slice(self, team : int, piece_type : int):
        return slice(self.type_locs[team, piece_type], 
                     self.type_locs[team, piece_type + 1] if piece_type != PAWN else None)


    def piece_coords(self, team : int, piece_type : int):
        return self.coords[team][self.piece_slice(team, piece_type)]


    def add_piece(self, piece_type : int, team : int, rank : int, file : int):
        location = self.type_locs[team, piece_type]
        self.coords[team] = np.insert(self.coords[team], location, [rank, file], axis=0)
        self.types[team] = np.insert(self.types[team], location, piece_type)
        self.type_locs[team, (piece_type+1):] += 1


    def remove_piece_at(self, team : int, rank : int, file : int):
        for piece_index in range(len(self.types[team])):
            if (self.coords[team][piece_index] == [rank, file]).all():
                # remove coordinate
                self.coords[team] = np.delete(self.coords[team], piece_index, axis=0)
                # remove corresponding type label
                self.types[team] = np.delete(self.types[team], piece_index, axis=0)
                # array type index locations must be updated
                for type_loc_index in range(len(self.type_locs[team]) - 1, -1, -1):
                    if self.type_locs[team, type_loc_index] <= piece_index or type_loc_index == 0:
                        self.type_locs[team, type_loc_index + 1:] -= 1
                        break
                
                break


class Move:
    piece_index : int
    to_coord : np.array

    def __init__(self, piece_index : int, to_coord : np.array | List[int]):
        self.piece_index = piece_index
        if type(to_coord) == np.array:
            self.to_coord = to_coord
        else:
            self.to_coord = np.array(to_coord)
    

    def __eq__(self, other : 'Move'):
        return self.piece_index == other.piece_index and (self.to_coord == other.to_coord).all()


    def __repr__(self):
        return f"({self.piece_index}, {self.to_coord})"