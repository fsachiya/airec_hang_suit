#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn

import ipdb

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


class FasterHRNN(nn.Module):
    #:: FasterHRNN
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
        super(FasterHRNN, self).__init__()
        activation = nn.LeakyReLU(negative_slope=0.3)
        # input_dim = img_feat_dim + state_dim
        # output_dim = input_dim
        self.state_dim = state_dim
        self.img_feat_dim = img_feat_dim
        self.sensory_dim = sensory_dim
        self.union1_dim = union1_dim
        self.union2_dim = union2_dim

        self.img_enc = ImgEnc(img_feat_dim=img_feat_dim)
        self.img_dec = ImgDec(img_feat_dim=img_feat_dim)
        
        
        self.img_feat_hid0 = nn.Parameter(torch.rand(1, sensory_dim), requires_grad=True)
        self.state_hid0 = nn.Parameter(torch.rand(1, sensory_dim), requires_grad=True)
        
        self.union1_hid0 = nn.Parameter(torch.rand(1, union1_dim), requires_grad=True)
        self.union2_hid0 = nn.Parameter(torch.rand(1, union2_dim), requires_grad=True)
        # _u1_hid0 == u2_hid0
        self._img_feat_hid0 = nn.Parameter(torch.rand(1, sensory_dim), requires_grad=True)
        self._state_hid0 = nn.Parameter(torch.rand(1, sensory_dim), requires_grad=True)

        # self.img_feat_gru = nn.GRUCell(img_feat_dim, sensory_dim)
        # self.state_gru = nn.GRUCell(state_dim, sensory_dim)
        # self.union1_gru = nn.GRUCell(sensory_dim*2, union1_dim)
        # self.union2_gru = nn.GRUCell(union1_dim, union2_dim)
        
        self.img_feat_gru = nn.GRU(self.img_feat_dim, sensory_dim, batch_first=True)
        self.state_gru = nn.GRU(self.state_dim, sensory_dim, batch_first=True)
        self.union1_gru = nn.GRU(sensory_dim*2, union1_dim, batch_first=True)
        self.union2_gru = nn.GRU(union1_dim, union2_dim, batch_first=True)
        
        self._union1_gru_cell = nn.GRUCell(union1_dim, union2_dim)
        self._img_feat_gru_cell = nn.GRUCell(img_feat_dim, sensory_dim)
        self._state_gru_cell = nn.GRUCell(state_dim, sensory_dim)
        
        self.union2_out = nn.Linear(union2_dim, union1_dim)
        self._union1_out = nn.Linear(union1_dim, sensory_dim*2)
        self._img_feat_out = nn.Linear(sensory_dim, self.img_feat_dim)
        self._state_out = nn.Linear(sensory_dim, self.state_dim)

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
        hid_dict = {"img_feat": None, "state": None,
                        "union1": None, "union2": None, "_union1": None, 
                        "_img_feat": None, "_state": None}):
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
        # if hid_dict["img_feat"] == None:
        #     img_feat_hid0 = self.img_feat_hid0.repeat(batch, 1).unsqueeze(0)
        #     state_hid0 = self.state_hid0.repeat(batch, 1).unsqueeze(0)
            
        #     union1_hid0 = self.union1_hid0.repeat(batch, 1).unsqueeze(0)
        #     union2_hid0 = self.union2_hid0.repeat(batch, 1).unsqueeze(0)
            
        #     _union1_hid0 = union2_hid0.squeeze(0)
        #     _img_feat_hid0 = self._img_feat_hid0.repeat(batch, 1)
        #     _state_hid0 = self._state_hid0.repeat(batch, 1)
            
        #     hid_dict["img_feat"] = img_feat_hid0
        #     hid_dict["state"] = state_hid0
        #     hid_dict["union1"] = union1_hid0
        #     hid_dict["union2"] = union2_hid0
        #     hid_dict["_union1"] = _union1_hid0
        #     hid_dict["_img_feat"] = _img_feat_hid0
        #     hid_dict["_state"] = _state_hid0
        
        # [5, 199, 8]
        z_imgs_enc = self.img_enc(x_imgs.reshape(batch*seq, -1, h, w)).reshape(batch, seq, -1)
        
        img_feat_hids, img_feat_hid = self.img_feat_gru(z_imgs_enc, hid_dict["img_feat"])
        state_hids, state_hid = self.state_gru(x_states, hid_dict["state"])
        sensory_hids = torch.cat([img_feat_hids, state_hids], dim=2)
        
        union1_hids, union1_hid = self.union1_gru(sensory_hids, hid_dict["union1"])
        union2_hids, union2_hid = self.union2_gru(union1_hids, hid_dict["union2"])
        
        # if seq == 1:
        #     _prev_union1_hids = hid_dict["_union1"]
        # else:
        #     _prev_union1_hids = torch.concat((_union1_hid0.unsqueeze(1), self.union2_out(union2_hids[:,:-1])), dim=1).reshape(-1,self.union1_dim)
        # _union1_hids = self._union1_gru_cell(union1_hids.reshape(-1,self.union1_dim), _prev_union1_hids)
        # _union1_hids = _union1_hids.reshape(batch, -1, self.union1_dim)

        # _sensory_hids = self._union1_out(_union1_hids)
        
        # if seq == 1:
        #     _prev_img_feat_hids = hid_dict["_img_feat"]
        # else:
        #     _prev_img_feat_hids = torch.concat((_img_feat_hid0.unsqueeze(1),_sensory_hids[:,:-1,:self.sensory_dim]), dim=1).reshape(-1,self.sensory_dim)
        # _img_feat_hids = self._img_feat_gru_cell(z_imgs_enc.reshape(-1,self.img_feat_dim), _prev_img_feat_hids)
        # _img_feat_hids = _img_feat_hids.reshape(batch, -1, self.sensory_dim)
        # z_imgs_dec = self._img_feat_out(_img_feat_hids)
        
        # if seq == 1:
        #     _prev_state_hids = hid_dict["_state"]
        # else:
        #     _prev_state_hids = torch.concat((_state_hid0.unsqueeze(1),_sensory_hids[:,:-1,self.sensory_dim:]), dim=1).reshape(-1,self.sensory_dim)
        # _state_hids = self._state_gru_cell(x_states.reshape(-1,self.state_dim), _prev_state_hids)
        # _state_hids = _state_hids.reshape(batch, -1, self.sensory_dim)
        # y_states = self._state_out(_state_hids)
        
        # y_imgs = self.img_dec(z_imgs_dec.reshape(-1, self.img_feat_dim))
        # y_imgs = y_imgs.reshape(batch, seq, -1, h, w)
        
        
        _union1_hid_list = []
        for i in reversed(range(seq)):
            if i == 0:
                _u1_hid = _u1_hid0
            else:
                _u1_hid = u2_hids[:,i]
            _u1_hid = self._u1_gru_cell(u1_hids[:,i], _u1_hid)
            _u1_hid_list.append(_u1_hid)
        _u1_hids = torch.stack(_u1_hid_list).permute([1,0,2])
        
        
        _s_hid_list = []
        for i in range(seq):
            if i == 0:
                _s_hid = _s_hid0
            else:
                _s_hid = _u1_hids[:,i]
            _s_hid = self._s_gru_cell(x[:,i], _s_hid)
            _s_hid_list.append(_s_hid)
        _s_hids = torch.stack(_s_hid_list).permute([1,0,2])
        
        
        hids_dict = {"img_feat": img_feat_hids, "state": state_hids,
                   "union1": union1_hids, "union2": union2_hids, "_union1": _union1_hids, 
                   "_img_feat": _img_feat_hids, "_state": _state_hids}
        
        hid_dict = {"img_feat": img_feat_hid, "state": state_hid,
                   "union1": union1_hid, "union2": union2_hid, "_union1": _union1_hids[:,-1], 
                   "_img_feat": _img_feat_hids[:,-1], "_state": _state_hids[:,-1]}
        
        return y_imgs, y_states, hids_dict, hid_dict