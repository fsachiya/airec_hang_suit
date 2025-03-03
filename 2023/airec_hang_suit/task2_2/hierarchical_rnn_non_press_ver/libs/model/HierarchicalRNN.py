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
    from libs.layer import HierarchicalRNNCell
except:
    from layer import HierarchicalRNNCell


class HSARNN(nn.Module):
    def __init__(
        self,
        # rnn_dim,
        # union_dim,
        # joint_dim=14,
        srnn_hid_dim=50,
        urnn_hid_dim=20,
        k_dim=5,
        vec_dim=10,
        # press_dim=4,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        img_size=[128, 128],
    ):
        super(HSARNN, self).__init__()

        self.k_dim = k_dim

        """
        if isinstance(activation, str):
            activation = get_activation_fn(activation, inplace=True)
        """
        activation  = nn.LeakyReLU(negative_slope=0.3)
        
        sub_img_size = [
            img_size[0] - 3 * (kernel_size - 1),
            img_size[1] - 3 * (kernel_size - 1),
        ]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        #"""
        # Positional Encoder
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=sub_img_size[0],
                height=sub_img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
        """
        self.pos_encoder = SpatialSoftmax(
                width=sub_img_size[0],
                height=sub_img_size[1],
                temperature=self.temperature,
                normalized=True,
            )
        """

        # Image Encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )

        key_dim = self.k_dim * 2
        # self.rec
        self.hrnn = HierarchicalRNNCell(srnn_input_dims={"k": key_dim, "v": vec_dim},   # , "p": press_dim
                                       srnn_hid_dim=srnn_hid_dim,
                                       urnn_hid_dim=urnn_hid_dim
                                    #    input_dim1=key_dim, 
                                    #    input_dim2=key_dim, 
                                    #    rnn_dim=rnn_dim, 
                                    #    union_dim=union_dim
                                       )
        # Point Decoder
        self.decoder_point = nn.Sequential(
            # nn.Linear(rnn_dim, key_dim), activation
            nn.Linear(srnn_hid_dim, key_dim)
        )  # Linear layer and activation

        # Joint Decoder
        self.decoder_joint = nn.Sequential(
            # nn.Linear(rnn_dim, joint_dim), activation
            nn.Linear(srnn_hid_dim, vec_dim), 
            activation
        )  # Linear layer and activation
        
        # # Pressure Decoder
        # self.decoder_press = nn.Sequential(
        #     # nn.Linear(rnn_dim, joint_dim), activation
        #     nn.Linear(srnn_hid_dim, press_dim), 
        #     activation
        # )  # Linear layer and activation
        

        # Inverse Spatial Softmax
        self.issm = InverseSpatialSoftmax(
            width=sub_img_size[0],
            height=sub_img_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Image Decoder
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(
                self.k_dim, 32, 3, 1, 0
            ),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
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

    def forward(self, xi, xv, states=None): # , xp
        # Encode input image
        # im_hid
        xi_feat = self.img_encoder(xi)
        enc_pts, _ = self.pos_encoder(xi)
        #enc_pts, _ = self.pos_encoder(im_hid)

        # Reshape encoded points and concatenate with input vector
        # enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        xk = enc_pts
        # hid = torch.cat([enc_pts, xv], -1)
        # rnn_hid = self.rec(xv, enc_pts, state)  # LSTM forward pass
        
        # [new_ksrnn_state, new_vsrnn_state, new_psrnn_state, new_urnn_state]
        states = self.hrnn(xk, xv, states) # enc_pts　, xp
        # y_joint = self.decoder_joint(rnn_hid[0][0])  # Decode joint prediction
        # dec_pts = self.decoder_point(rnn_hid[1][0])  # Decode points
        dec_pts = self.decoder_point(states[0][0])  # Decode points
        yv = self.decoder_joint(states[1][0])  # Decode joint prediction
        # yp = self.decoder_press(states[2][0])  # Decode joint prediction

        # Reshape decoded points
        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        # hid
        xi_feat = torch.mul(heatmap, xi_feat)  # Multiply heatmap with image feature `im_hid`

        yi = self.decoder_image(xi_feat)  # Decode image
        return yi, yv, enc_pts, dec_pts, states# rnn_hid    # , yp


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
