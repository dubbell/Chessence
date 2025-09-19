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


def tensor_check(x):
    if not isinstance(x, Tensor):
        return torch.tensor(x)
    return x


def to_ndim(tensor : Tensor, ndim : int):
    if ndim > tensor.dim():
        for _ in range(max(0, ndim - tensor.dim())):
            tensor = tensor.unsqueeze(0)
    elif ndim < tensor.dim():
        prev_dim = tensor.dim()
        dim_idx = 0
        while tensor.dim() > ndim:
            tensor = tensor.squeeze(dim_idx)
            if tensor.dim() == prev_dim:
                dim_idx += 1
                if dim_idx == tensor.dim():
                    break
            prev_dim = tensor.dim()

    return tensor


def validate_tensors(data, desired_dims = None):
    """
    data : list of inputs that should be converted to tensors
    desired_dims : desired dimensionality of the inputs"""

    mapped = [to_ndim(tensor_check(x), desired_dim) for x, desired_dim in zip(data, desired_dims)]
    return mapped if len(mapped) > 1 else mapped[0]