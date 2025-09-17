from train.utils import to_ndim
import torch


def test_to_ndim():
    small = torch.tensor(5)
    assert to_ndim(small, 2).dim() == 2
    assert to_ndim(small, 4).dim() == 4
    assert to_ndim(small, 5).dim() == 5

    big = torch.zeros((1, 1, 4))
    assert to_ndim(big, 2).shape == (1, 4)
    assert to_ndim(big, 1).shape == (4,)

    big2 = torch.zeros((1, 2, 1, 1, 5))
    assert to_ndim(big2, 5).shape == (1, 2, 1, 1, 5)
    assert to_ndim(big2, 4).shape == (2, 1, 1, 5)
    assert to_ndim(big2, 3).shape == (2, 1, 5)
    assert to_ndim(big2, 2).shape == (2, 5)
    assert to_ndim(big2, 1).shape == (2, 5)
