#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn
from eipl.layer import GridMask
from eipl.utils import normalization

try:
    from libs.layer import HierarchicalRNNCell2
except:
    from layer import HierarchicalRNNCell2


class HSARNN2(nn.Module):
    def __init__(
        self,
        # rnn_dim,
        # union_dim,
        # joint_dim=14,
        srnn_hid_dim=50,
        urnn_hid_dim=20,
        k_dim=5,
        feat_dim=32,
        vec_dim=10,
        press_dim=4,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        img_size=[128, 128],
    ):
        super(HSARNN2, self).__init__()

        self.k_dim = k_dim

        """
        if isinstance(activation, str):
            activation = get_activation_fn(activation, inplace=True)
        """
        activation  = nn.LeakyReLU(negative_slope=0.3)
        
        sub_img_size = [
            img_size[0],
            img_size[1],
        ]
        self.temperature = temperature
        self.heatmap_size = heatmap_size
        
        # Image Encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # 3, 64, 64 -> 16, 64, 64
            activation,
            nn.Conv2d(16, 32, 3, 1, 1),  # 16, 64, 64 -> 32, 64, 64
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 1),  # 32, 64, 64 -> 8, 64, 64
            activation,
        )
        
        # Spatial Softmax
        self.ssm = SpatialSoftmax(
            width=sub_img_size[0],
            height=sub_img_size[1],
            temperature=self.temperature,
            normalized=True,
        )
        
        # Feat Encoder
        self.feat_encoder = nn.Sequential(
            nn.Conv2d(self.k_dim, 16, 3, 1, 1),  # 8, 64, 64 -> 16, 64, 64
            nn.MaxPool2d(2, stride=2),  # 16, 64, 64 -> 16, 32, 32
            activation,
            nn.Conv2d(16, 32, 3, 1, 1), # 16, 32, 32 -> 32, 32, 32
            nn.MaxPool2d(2, stride=2),  # 32, 32, 32 -> 32, 16, 16
            activation,
            nn.Conv2d(32, 64, 3, 1, 1), # 32, 16, 16 -> 64, 16, 16
            nn.MaxPool2d(2, stride=2),  # 64, 16, 16 -> 64, 8, 8
            activation,
            nn.Conv2d(64, 128, 3, 1, 1), # 64, 8, 8 -> 128, 8, 8
            nn.MaxPool2d(2, stride=2),  # 128, 8, 8 -> 128, 4, 4
            activation,
            nn.Conv2d(128, 256, 3, 1, 1), # 128, 4, 4 -> 256, 4, 4
            nn.MaxPool2d(2, stride=2),  # 256, 4, 4 -> 256, 2, 2
            activation,
            nn.Conv2d(256, 512, 3, 1, 1), # 256, 2, 2 -> 512, 2, 2
            nn.MaxPool2d(2, stride=2),  # 512, 2, 2 -> 512, 1, 1
            activation,
            nn.Flatten(),
            nn.Linear(512, feat_dim), # 512 -> 32
            activation,
        )
        key_dim = self.k_dim * 2

        self.hrnn = HierarchicalRNNCell2(
            srnn_input_dims={"f": feat_dim, "k": key_dim, "v": vec_dim, "p": press_dim}, 
            srnn_hid_dim=srnn_hid_dim,
            urnn_hid_dim=urnn_hid_dim
        )        
        
        # Point Decoder
        self.point_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, key_dim)
        )  # Linear layer and activation

        # Joint Decoder
        self.vec_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, vec_dim), 
            activation
        )  # Linear layer and activation
        
        # Pressure Decoder
        self.press_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, press_dim), 
            activation
        )  # Linear layer and activation
        
        # Inverse Spatial Softmax
        self.issm = InverseSpatialSoftmax(
            width=sub_img_size[0],
            height=sub_img_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Feat Decoder
        self.feat_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, 512),
            activation,
            nn.Unflatten(1, (512, 1, 1)),  # 512 -> 512, 1, 1
            nn.ConvTranspose2d(512, 256, 3, 1, 1),  # 512, 1, 1 -> 256, 1, 1
            nn.Upsample(scale_factor=2),  # 256, 1, 1 -> 256, 2, 2
            activation,
            nn.ConvTranspose2d(256, 128, 3, 1, 1),  # 256, 2, 2 -> 128, 2, 2
            nn.Upsample(scale_factor=2),  # 128, 2, 2 -> 128, 4, 4
            activation,
            nn.ConvTranspose2d(128, 64, 3, 1, 1),  # 128, 4, 4 -> 64, 4, 4
            nn.Upsample(scale_factor=2),  # 64, 4, 4 -> 128, 8, 8
            activation,
            nn.ConvTranspose2d(64, 32, 3, 1, 1),  # 64, 8, 8 -> 32, 8, 8
            nn.Upsample(scale_factor=2),  # 32, 8, 8 -> 32, 16, 16
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 1),  # 32, 16, 16 -> 16, 16, 16
            nn.Upsample(scale_factor=2),  # 16, 16, 16 -> 16, 32, 32
            activation,
            nn.ConvTranspose2d(16, self.k_dim, 3, 1, 1),  # 16, 32, 32 -> 8, 32, 32
            nn.Upsample(scale_factor=2),  # 8, 32, 32 -> 8, 64, 64
            activation,
            
        )
        
        # Image Decoder
        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.k_dim, 32, 3, 1, 1),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 1),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 1),  # Transposed Convolutional layer 3
            activation,
        )
        
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "rec" in name or "rnn" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    # n = p.size(0)
                    # p.data[(n // 4): (n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
            elif "decoder" in name or "encoder" in name:
                if "weight" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "bias" in name:
                    p.data.fill_(0)

    def forward(self, xi, xv, xp, states=None):
        # Encode input image
        xi_feat = self.img_encoder(xi)
        
        # Reshape encoded points and concatenate with input vector
        enc_pts, _ = self.ssm(xi_feat)
        
        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        xk = enc_pts
        
        xf = self.feat_encoder(xi_feat)
        
        # [new_fsrnn_state, new_ksrnn_state, new_vsrnn_state, new_psrnn_state, new_urnn_state]
        states = self.hrnn(xf, xk, xv, xp, states) # enc_pts

        yi_feat = self.feat_decoder(states[0][0])
        dec_pts = self.point_decoder(states[1][0])  # Decode points
        yv = self.vec_decoder(states[2][0])  # Decode joint prediction
        yp = self.press_decoder(states[3][0])  # Decode joint prediction
        
        # Reshape decoded points
        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        # hid
        yi_feat = torch.mul(heatmap, yi_feat)  # Multiply heatmap with image feature `im_hid`        
        
        yi = self.img_decoder(yi_feat)  # Decode image
        
        return yi, yv, yp, enc_pts, dec_pts, states


if __name__ == "__main__":
    from torchinfo import summary

    batch_size = 8
    joint_dim = 7

    # test RNNModel
    model = HSARNN(rnn_dim=50, union_dim=20, joint_dim=joint_dim)
    summary(
        model,
        input_size=[(batch_size, 3, 128, 128), (batch_size, joint_dim)]
    )
