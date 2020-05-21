import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, type='gan'):
        super(Discriminator, self).__init__()
        
        if type == 'wgan':  # wgan去掉sigmoid层
            self.layers = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(inplace=False),
                nn.Linear(256,64),
                nn.ReLU(inplace=False),
                nn.Linear(64,1),
            )
        else :
            self.layers = nn.Sequential(
                nn.Linear(2,256),
                nn.ReLU(inplace=False),
                nn.Linear(256,64),
                nn.ReLU(inplace=False),
                nn.Linear(64,1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, in_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)



