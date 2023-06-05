# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from agents.helpers import SinusoidalPosEmb


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels

        self.linear1 = nn.Linear(channels, channels * 2)
        self.linear2 = nn.Linear(channels * 2, channels)
        self.linear3 = nn.Linear(channels, int(channels / 2))
        self.linear4 = nn.Linear(int(channels / 2), channels)

    def forward(self, x):
        y = F.mish(self.linear1(x))
        y = self.linear2(y)
        z = F.mish(x + y)
        y2 = F.mish(self.linear3(z))
        y2 = self.linear4(y2)
        return F.mish(z + y2)


class MLP_res(nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):
        super(MLP_res, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       )

        self.resblock = ResidualBlock(256)
        self.resblock2 = ResidualBlock(256)
        self.resblock3 = ResidualBlock(256)
        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        x = self.resblock(x)
        x = self.resblock2(x)
        x = self.resblock3(x)

        return self.final_layer(x)


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       )

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
