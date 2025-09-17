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


def unsqueeze_check(tensor : Tensor, desired_dim : int):
    if tensor.dim() < desired_dim:
        return tensor.unsqueeze(0)


def validate_tensors(data, desired_dims = None):
    """
    data : list of inputs that should be converted to tensors
    desired_dims : desired dimensionality of the inputs"""

    mapped = [unsqueeze_check(tensor_check(x), desired_dim) for x, desired_dim in zip(data, desired_dims)]
    return mapped if len(mapped) > 1 else mapped[0]
        

