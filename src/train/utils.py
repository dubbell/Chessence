from enum import Enum
import torch 
from torch import Tensor


class MoveResult(Enum):
    CONTINUE = 0
    LOSS = 1
    DRAW = 2

    def __eq__(self, other):
        return isinstance(other, Enum) and self.value == other.value
    
    def __hash__(self):
        return self.value

CONTINUE = MoveResult.CONTINUE
LOSS = MoveResult.LOSS
DRAW = MoveResult.DRAW


def tensor_check(data):
    if not isinstance(data, Tensor):
        return torch.tensor(data)
    return data