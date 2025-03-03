import torch
import torch.nn as nn

import sys
sys.path.append("/home/fujita/work/eipl")
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax

class ImgDec(nn.Module):
    def __init__(
        self, 
        img_feat_dim
        ):
        super(ImgDec, self).__init__()
        
        activation = nn.LeakyReLU(negative_slope=0.3)
        
        self.deconv = nn.Sequential(
            nn.Linear(img_feat_dim, 64),            
            activation,
            nn.Linear(64, 32*8*8),        
            activation,
            nn.Unflatten(1, (32, 8, 8)),            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16, 16, 16
            activation,
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # 8, 32, 32
            activation,
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 3, 64, 64
            nn.Sigmoid()                       
        )
        
    def forward(self, zi):
        return self.deconv(zi)

class SAImgDec(nn.Module):
    def __init__(
        self,
        key_dim=8
        ):
        super(SAImgDec, self).__init__()
        
        self.key_dim = key_dim
        activation = nn.LeakyReLU(negative_slope=0.3)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                self.key_dim, 32, kernel_size=3, stride=1, padding=1), # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),  # Transposed Convolutional layer 3
            activation,
        )
        
    def forward(self, zi):
        return self.deconv(zi)
    
