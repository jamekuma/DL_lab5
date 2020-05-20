from torch import nn
import torch

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, real_out, z):


