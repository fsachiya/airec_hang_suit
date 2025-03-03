#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn

import ipdb

import sys
sys.path.append("/home/fujita/work/eipl")
from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax

sys.path.append("../")
from layer import SAImgEnc, PosEnc, SAImgDec

class SAStackRNN(nn.Module):
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
        key_dim=8,
        state_dim=8,
        union1_dim=64,
        union2_dim=64,
        union3_dim=64,
        temperature=1e-4,
        heatmap_size=0.1,
        img_size=[64, 64],
        # num_layers=3
        ):        
        super(SAStackRNN, self).__init__()
        
        activation = nn.LeakyReLU(negative_slope=0.3)
        input_dim = key_dim + state_dim
        output_dim = input_dim
        self.state_dim = state_dim
        # self.img_feat_dim = img_feat_dim
        
        self.pos_enc = PosEnc(key_dim, temperature, heatmap_size, img_size)
        self.img_enc = SAImgEnc(key_dim)
        self.img_dec = SAImgDec(key_dim)
        
        # self.gru = nn.GRU(input_dim, union_dim, num_layers=num_layers, batch_first=True)
        self.union1_gru = nn.GRU(input_dim, union1_dim)
        self.union2_gru = nn.GRU(union1_dim, union2_dim)
        self.union3_gru = nn.GRU(union2_dim, union3_dim)
        
        self.fc = nn.Linear(union3_dim, output_dim)

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
        x_imgs, 
        x_states, 
        hid_dict = {"union1": None, 
                     "union2": None, 
                     "union3": None}):
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
        
        batch, seq, c, h, w = x_imgs.shape
        
        ipdb.set_trace()
        
        pos_enc = self.pos_enc(x_imgs.reshape(batch*seq, -1, h, w)).reshape(batch, seq, -1)
        
        # z_imgs_enc = self.img_enc(x_imgs.reshape(batch*seq, -1, h, w))
        x_cat = torch.concat([x_states, pos_enc], dim=2)
        
        # hids, _ = self.gru(x_cat)
        union1_hids, union1_hid = self.union1_gru(x_cat, hid_dict["union1"])
        union2_hids, union2_hid = self.union2_gru(union1_hids, hid_dict["union2"])
        union3_hids, union3_hid = self.union3_gru(union2_hids, hid_dict["union3"])
        
        outs = self.fc(union3_hids)
        
        y_states = outs[:,:,:self.state_dim]
        z_imgs_dec = outs[:,:,self.state_dim:]
        
        y_imgs = self.img_dec(z_imgs_dec.reshape(batch*seq, -1))
        y_imgs = y_imgs.reshape(batch, seq, -1, h, w)
        
        hids_dict = {"union1": union1_hids, 
                     "union2": union2_hids, 
                     "union3": union3_hids}
        hid_dict = {"union1": union1_hid, 
                     "union2": union2_hid, 
                     "union3": union3_hid}
        
        return y_imgs, y_states, hids_dict, hid_dict
