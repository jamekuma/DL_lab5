import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_size, middle_size, type='gan'):
        super(Discriminator, self).__init__()
        if type == 'gan':
            self.layers = nn.Sequential(
                nn.Linear(in_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, 1),  # 最后输出一维，判断是或否
                nn.Sigmoid(),
            )
        elif type == 'wgan':  # wgan去掉sigmoid层
            self.layers = nn.Sequential(
                nn.Linear(in_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, middle_size),
                nn.ReLU(True),
                nn.Linear(middle_size, 1),  # 最后输出一维，判断是或否
            )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, in_size, middle_size, out_size):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, middle_size),
            nn.ReLU(True),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(True),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(True),
            nn.Linear(middle_size, out_size)
        )

    def forward(self, x):
        return self.layers(x)



