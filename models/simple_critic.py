import torch.nn as nn
from torch import Tensor


class SimpleConv(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, keep_size : bool):
        super().__init__()

        self.keep_size = keep_size

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x : Tensor):
        if self.keep_size:
            # add padding to keep input and output same size
            x = self.conv(nn.functional.pad(x, [0, 1, 0, 1]))
        else:
            x = self.conv(x)
        
        return self.relu(self.batch_norm(self.dropout(x)))
        


class SimpleCritic(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.constant_conv = nn.Sequential(
            SimpleConv(8, 64, True),
            SimpleConv(64, 64, True),
            SimpleConv(64, 64, True),
            SimpleConv(64, 64, True))
        
        self.decreasing_conv = nn.Sequential(
            SimpleConv(64, 64, False),    # outputs 7x7
            SimpleConv(64, 96, False),    # outputs 6x6
            SimpleConv(96, 96, False),    # outputs 5x5
            SimpleConv(96, 128, False),   # outputs 4x4
            SimpleConv(128, 128, False),  # outputs 3x3
            SimpleConv(128, 256, False),  # outputs 2x2
            SimpleConv(256, 512, False)   # outputs 1x1
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512), 
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1))
        
    
    def forward(self, x : Tensor):
        x = self.constant_conv(x)
        x = self.decreasing_conv(x)
        x = self.flatten(x)
        return self.linear(x)
        
