#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn

import ipdb
import sys
sys.path.append("../")
from layer import HRNNCell

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

class ImgDec(nn.Module):
    def __init__(self, img_feat_dim):
        super(ImgDec, self).__init__()
        self.deconv = nn.Sequential(
            nn.Linear(img_feat_dim, 64),            
            nn.ReLU(),
            nn.Linear(64, 32*8*8),        
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),            
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16, 16, 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),   # 8, 32, 32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 3, 64, 64
            nn.Sigmoid()                       
        )
        
    def forward(self, zi):
        return self.deconv(zi)


class HRNN(nn.Module):
    #:: StackRNN
    """SARNN: Spatial Attention with Recurrent Neural Network.
    This model "explicitly" extracts positions from the image that are important to the task, such as the work object or arm position,
    and learns the time-series relationship between these positions and the robot's joint angles.
    The robot is able to generate robust motions in response to changes in object position and lighting.

    Arguments:
        union_dim (int): The dimension of the recurrent state in the LSTM cell.
        k_dim (int, optional): The dimension of the attention points.
        state_dim (int, optional): The dimension of the joint angles.
        temperature (float, optional): The temperature parameter for the softmax function.
        heatmap_size (float, optional): The size of the heatmap in the InverseSpatialSoftmax layer.
        kernel_size (int, optional): The size of the convolutional kernel.
        activation (str, optional): The name of activation function.
        im_size (list, optional): The size of the input image [height, width].
    """

    def __init__(
        self,
        img_feat_dim=8,
        state_dim=8,
        sensory_dim=64,
        union1_dim=64,
        union2_dim=64,
        img_size=[64, 64],
    ):
        super(HRNN, self).__init__()
        activation = nn.LeakyReLU(negative_slope=0.3)
        # input_dim = img_feat_dim + state_dim
        # output_dim = input_dim
        self.state_dim = state_dim
        self.img_feat_dim = img_feat_dim
        self.sensory_dim = sensory_dim

        self.img_enc = ImgEnc(img_feat_dim=img_feat_dim)
        self.img_dec = ImgDec(img_feat_dim=img_feat_dim)
        
        self.hrnncell = HRNNCell(img_feat_dim, state_dim, sensory_dim, union1_dim, union2_dim)

        self.apply(self._weights_init)

    def _weights_init(self, m):
        """
        Tensorflow/Keras-like initialization
        """
        if isinstance(m, nn.LSTMCell):
            nn.init.xavier_uniform_(m.weight_ih)
            nn.init.orthogonal_(m.weight_hh)
            nn.init.zeros_(m.bias_ih)
            nn.init.zeros_(m.bias_hh)

        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self, 
        x_img, 
        x_state, 
        hid_dict={"img_feat": None, "state": None, "union1": None, "union2": None},
        ):
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
            y_joint (torch.Tensor): Decoded joint prediction tensor of shape (batch_size, state_dim).
            enc_pts (torch.Tensor): Encoded points tensor of shape (batch_size, k_dim * 2).
            dec_pts (torch.Tensor): Decoded points tensor of shape (batch_size, k_dim * 2).
            rnn_hid (tuple): Tuple containing the hidden state and cell state of the LSTM cell.
        """
        
        batch, c, h, w = x_img.shape
        
        z_imgs_enc = self.img_enc(x_img)
        
        z_img_dec, y_state, prev_hid_dict = self.hrnncell(z_imgs_enc, x_state, hid_dict)
        
        y_img = self.img_dec(z_img_dec)
        
        return y_img, y_state, prev_hid_dict

