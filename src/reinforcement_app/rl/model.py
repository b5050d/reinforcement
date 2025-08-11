"""
Neural network model and associated methods
"""

import torch.nn as nn


class DQN(nn.Module):
    """
    Simple NN to train on our little reinforcement
    learning simulation
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)
        )

    def forward(self, x):
        return self.model(x)
