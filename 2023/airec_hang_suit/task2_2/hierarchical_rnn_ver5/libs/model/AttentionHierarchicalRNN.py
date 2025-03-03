#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import math

from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn
from eipl.layer import GridMask
from eipl.utils import normalization

try:
    from libs.layer import HierachicalRNNCell
    # from libs.layer import HierarchicalRNNCellwithoutKey
except:
    from layer import HierarchicalRNNCell
    # from layer import HierarchicalRNNCellwithoutKey

# class SelfAttention(nn.Module):
#     def __init__(self, num_heads):
#         super(SelfAttention, self).__init__()
#         self.attn = nn.MultiheadAttention(1, num_heads)
        
#     def forward(self, x):
#         # Self-Attentionを適用
#         attn_x, wx = self.attn(x, x, x)
#         return attn_x, wx


class AttnHSARNN(nn.Module):
    def __init__(
        self,
        # rnn_dim,
        # union_dim,
        # joint_dim=14,
        srnn_hid_dim=50,
        urnn_hid_dim=20,
        k_dim=5,
        vec_dim=10,
        press_dim=4,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        batch_size=6,
        num_heads=2,
        activation="lrelu",
        img_size=[128, 128],
    ):
        super(AttnHSARNN, self).__init__()

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
        self.hrnn = HierarchicalRNNCell(    # HierarchicalRNNCell
            srnn_input_dims={"k": key_dim, "v": vec_dim, "p": press_dim}, 
            srnn_hid_dim=srnn_hid_dim,
            urnn_hid_dim=urnn_hid_dim
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
        
        # Pressure Decoder
        self.decoder_press = nn.Sequential(
            # nn.Linear(rnn_dim, joint_dim), activation
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

        # Image Decoder
        self.img_decoder = nn.Sequential(   # img_decoder
            nn.ConvTranspose2d(self.k_dim, 32, 3, 1, 0),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
            activation,
        )
        
        # self.K_v = nn.Linear(vec_dim, vec_dim, bias=False)
        # self.Q_v = nn.Linear(vec_dim, vec_dim, bias=False)
        # self.K_p = nn.Linear(press_dim, press_dim, bias=False)
        # self.Q_p = nn.Linear(press_dim, press_dim, bias=False)
        
        # self.vec_attn = nn.MultiheadAttention(embed_dim=batch_size, num_heads=num_heads)
        # self.press_attn = nn.MultiheadAttention(embed_dim=batch_size, num_heads=num_heads)
        self.vec_attn = nn.MultiheadAttention(embed_dim=vec_dim, num_heads=num_heads)
        self.press_attn = nn.MultiheadAttention(embed_dim=press_dim, num_heads=num_heads)
        
        
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):   #  or isinstance(m, nn.Linear)
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        
        if isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            nn.init.zeros_(m.in_proj_bias)

    
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

    def forward(self, 
                xi, xv, xp, 
                # step, 
                states=None,
                ):  # step is variable
        # Encode input image
        # im_hid
        xi_feat = self.img_encoder(xi)
        enc_pts, _ = self.pos_encoder(xi)

        # Reshape encoded points and concatenate with input vector
        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)
        xk = enc_pts
        
        # _xv = xv.permute(1,0)
        # _xp = xp.permute(1,0)
        # _attn_xv, _attn_wv = self.vec_attn(_xv, _xv, _xv)
        # _attn_xp, _attn_wp = self.vec_attn(_xp, _xp, _xp)
        # attn_xv = _attn_xv.permute(1,0)
        # attn_xp = _attn_xp.permute(1,0)
        
        attn_xv, attn_wv = self.vec_attn(xv, xv, xv)
        attn_xp, attn_wp = self.press_attn(xp, xp, xp)
                        
        states = self.hrnn(xk, xv, xp, states) # xv, xp , step, 
        # states = self.hrnn(xk, xv, xp, states) # xv, xp

        dec_pts = self.decoder_point(states[0][0])  # Decode points
        yv = self.decoder_joint(states[1][0])  # Decode joint prediction
        yp = self.decoder_press(states[2][0])  # Decode joint prediction

        # Reshape decoded points
        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        # hid
        xi_feat = torch.mul(heatmap, xi_feat)  # Multiply heatmap with image feature `im_hid`        
        
        yi = self.img_decoder(xi_feat)  # Decode image    img_decoder
        return yi, yv, yp, enc_pts, dec_pts, states, xi_feat


# if __name__ == "__main__":
#     from torchinfo import summary

#     batch_size = 8
#     joint_dim = 7

#     # test RNNModel
#     model = HSARNN(rnn_dim=50, union_dim=20, joint_dim=joint_dim)
#     summary(
#         model,
#         input_size=[(batch_size, 3, 128, 128), (batch_size, joint_dim)]
#     )
