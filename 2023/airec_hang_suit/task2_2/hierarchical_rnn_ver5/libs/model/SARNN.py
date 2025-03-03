#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn
import ipdb

from layer import MSA, Imgcropper, AutoEncoder, MSARNN


class SARNN(nn.Module):
    #:: SARNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        rec_dim (int): The dimension of the recurrent state in the LSTM cell.
        key_dim (int, optional): The dimension of the attention points.
        vec_dim (int, optional): The dimension of the joint angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        activation (str, optional): The name of activation function.
        img_size (list, optional): The size of the input image [height, width].
    """

    def __init__(
        self,
        rec_dim,
        key_dim=5,
        vec_dim=14,
        press_dim=7,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        img_size=[128, 128],
        device="cuda:0"
    ):
        super(SARNN, self).__init__()

        self.key_dim = key_dim
        self.vec_dim = vec_dim
        self.press_dim = press_dim

        if isinstance(activation, str):
            activation = get_activation_fn(activation, inplace=True)

        sub_img_size = [img_size[0] - 3 * (kernel_size - 1), img_size[1] - 3 * (kernel_size - 1)]
        self.temperature = temperature
        self.heatmap_size = heatmap_size

        # Positional Encoder
        self.pos_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
            SpatialSoftmax(
                width=sub_img_size[0],
                height=sub_img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )

        # Image Encoder
        self.im_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 0),  # Convolutional layer 1
            activation,
            nn.Conv2d(16, 32, 3, 1, 0),  # Convolutional layer 2
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 0),  # Convolutional layer 3
            activation,
        )

        self.rec_in = self.key_dim * 2 + self.vec_dim + self.press_dim
        self.rec = nn.LSTMCell(self.rec_in, rec_dim)  # LSTM cell

        # Joint Decoder
        self.decoder_joint = nn.Sequential(
            nn.Linear(rec_dim, vec_dim), activation
        )  # Linear layer and activation
        
        # Pressure Decoder
        self.decoder_pressure = nn.Sequential(
            nn.Linear(rec_dim, press_dim), activation
        )

        # Point Decoder
        self.decoder_point = nn.Sequential(
            nn.Linear(rec_dim, self.key_dim * 2), activation
        )  # Linear layer and activation

        # Inverse Spatial Softmax
        self.issm = InverseSpatialSoftmax(
            width=sub_img_size[0],
            height=sub_img_size[1],
            heatmap_size=self.heatmap_size,
            normalized=True,
        )

        # Image Decoder
        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(self.key_dim, 32, 3, 1, 0),  # Transposed Convolutional layer 1
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0),  # Transposed Convolutional layer 2
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 0),  # Transposed Convolutional layer 3
            activation,
        )
        
        ############################
        self.msarnn = MSARNN(device,
                 img_h=128,
                 img_w=128,
                 att_num=self.key_dim,
                 robot_dim=self.vec_dim+self.press_dim,
                 rnn_dim=50,
                 temperature=0.05,
                 heatmap_size=0.05,)
        ############################
        
    def forward(self, xi, xv, xp, state=None):
        """
        Forward pass of the SARNN module.
        Predicts the image, joint angle, and attention at the next time based on the image and joint angle at time t.
        Predict the image, joint angles, and attention points for the next state (t+1) based on
        the image and joint angles of the current state (t).
        By inputting the predicted joint angles as control commands for the robot,
        it is possible to generate sequential motion based on sensor information.

        Arguments:
            xi (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            xv (torch.Tensor): Input vector tensor of shape (batch_size, input_dim).
            state (tuple, optional): Initial hidden state and cell state of the LSTM cell.

        Returns:
            y_image (torch.Tensor): Decoded image tensor of shape (batch_size, channels, height, width).
            y_joint (torch.Tensor): Decoded joint prediction tensor of shape (batch_size, vec_dim).
            enc_pts (torch.Tensor): Encoded points tensor of shape (batch_size, key_dim * 2).
            dec_pts (torch.Tensor): Decoded points tensor of shape (batch_size, key_dim * 2).
            rnn_hid (tuple): Tuple containing the hidden state and cell state of the LSTM cell.
        """
        ########################
        x_rb = torch.cat([xv, xp], dim=1)
        [y_img, rec_img], y_rb, [x_pt, y_pt], state = cp.checkpoint(self.msarnn, xi, x_rb, state, use_reentrant=False) #cp.checkpoint()
        yi = y_img
        xf = rec_img
        yv, yp = torch.split(y_rb, [self.vec_dim, self.press_dim], dim=1)
        enc_pts = x_pt
        dec_pts = y_pt
        ########################
        
        # # Encode input image
        # xf = self.im_encoder(xi)
        # enc_pts, _ = self.pos_encoder(xi)

        # # Reshape encoded points and concatenate with input vector
        # enc_pts = enc_pts.reshape(-1, self.key_dim * 2)
        # hid = torch.cat([enc_pts, xv, xp], -1)
        
        # state = self.rec(hid, state)  # LSTM forward pass
        # yv = self.decoder_joint(state[0])  # Decode joint prediction
        # yp = self.decoder_pressure(state[0])
        # dec_pts = self.decoder_point(state[0])  # Decode points

        # # Reshape decoded points
        # dec_pts_in = dec_pts.reshape(-1, self.key_dim, 2)
        # heatmap = self.issm(dec_pts_in)  # Inverse Spatial Softmax
        # hid = torch.mul(heatmap, xf)  # Multiply heatmap with image feature `xf`

        # yi = self.decoder_image(hid)  # Decode image
        
        return yi, yv, yp, enc_pts, dec_pts, state
