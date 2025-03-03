import torch
import torch.nn as nn

import ipdb

import sys
sys.path.append("/home/fujita/work/eipl")
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax

class ImgEnc(nn.Module):
    def __init__(self, img_feat_dim):
        super(ImgEnc, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), # 8, 32, 32
                                  nn.ReLU(),
                                  nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # 16, 16, 16
                                  nn.ReLU(),
                                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 32, 8, 8
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(32*8*8, 64),
                                  nn.ReLU(),
                                  nn.Linear(64, img_feat_dim))
    def forward(self, xi):
        return self.conv(xi)


class SAImgEnc(nn.Module):
    def __init__(
        self, 
        key_dim=8,
        ):
        super(SAImgEnc, self).__init__()
        self.key_dim = key_dim
        activation = nn.LeakyReLU(negative_slope=0.3)
        
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )
    def forward(self, xi):
        return self.conv(xi)


class PosEnc(nn.Module):
    def __init__(
        self, 
        key_dim=8,
        temperature=1e-4,
        heatmap_size=0.1,
        img_size=[64, 64]
        ):
        super(PosEnc, self).__init__()
        self.key_dim = key_dim
        activation = nn.LeakyReLU(negative_slope=0.3)
        
        self.temperature = temperature
        self.heatmap_size = heatmap_size
        
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),  # Convolutional layer 2
            activation,
            nn.Conv2d(in_channels=32, out_channels=self.key_dim, kernel_size=3, stride=1, padding=1, padding_mode="replicate", groups=1),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=img_size[0],
                height=img_size[1],
                temperature=self.temperature,
                normalized=True,
            )  # Spatial Softmax layer
        )
    def forward(self, xi):
        ipdb.set_trace()
        return self.pos_encoder(xi)