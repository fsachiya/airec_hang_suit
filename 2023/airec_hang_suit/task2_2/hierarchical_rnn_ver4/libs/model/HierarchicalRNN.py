#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from typing import Dict, List, Tuple
import ipdb
import pdb
import matplotlib.pyplot as plt
import time

from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn
from eipl.layer import GridMask
from eipl.utils import normalization

try:
    from libs.layer import HierachicalRNNCell
except:
    from layer import HierarchicalRNNCell


class HSARNN(nn.Module):
    def __init__(
        self,
        srnn_hid_dim=50,
        urnn_hid_dim=20,
        key_dim=5,
        vec_dim=10,
        press_dim=4,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        img_size=[128, 128],
    ):
        super(HSARNN, self).__init__()

        self.key_dim = key_dim
        """
        if isinstance(activation, str):
            activation = get_activation_fn(activation, inplace=True)
        """
        activation  = nn.LeakyReLU(negative_slope=0.3)
        self.img_size = img_size
        self.temperature = temperature
        self.heatmap_size = heatmap_size
        
        self.grid_side = 2
        self.grid_num = self.grid_side ** 2
        self.grid_size = [int(self.img_size[0]/self.grid_side), 
                     int(self.img_size[0]/self.grid_side)]
        
        self.overlaped_grid_side = self.grid_side*self.grid_side -1
        self.overlaped_grid_num = self.overlaped_grid_side ** 2
        ##################################################
        self.hrnn = HierarchicalRNNCell(
            srnn_input_dims={"k": 2*self.key_dim*self.overlaped_grid_num, "v": vec_dim, "p": press_dim},    # *self.grid_num
            srnn_hid_dim=srnn_hid_dim,
            urnn_hid_dim=urnn_hid_dim
        )
        ##################################################
        
        ##################################################
        self.heatmap2key = nn.Sequential(
            SpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
        self.key2heatmap = nn.Sequential(
            InverseSpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                heatmap_size=self.heatmap_size,
                normalized=True,
            )
        )
        ##################################################
        
        ##################################################
        # key encoder
        self.key_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # Convolutional layer 1
            # nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 32, 3, 1, 1),  # Convolutional layer 2
            # nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 1),  # Convolutional layer 3
            nn.GroupNorm(1, self.key_dim),
            activation,
            SpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
        
        # key cbam encoder
        self.key_cbam_encoder = nn.Sequential(
            CBAMEncoder(self.key_dim, self.img_size),
            # SpatialSoftmax(
            #     width=self.img_size[0],
            #     height=self.img_size[1],
            #     temperature=self.temperature,
            #     normalized=True,
            # ),
        )
        
        self.grid_key_cbam_encoder = nn.Sequential(
            GridCBAMEncoder(self.grid_num, 
                            self.grid_side,
                            self.grid_size,
                            self.img_size,
                            self.key_dim),
        )
        
        self.overlaped_grid_key_cbam_encoder = nn.Sequential(
            OverlapedGridCBAMEncoder(self.grid_num, 
                            self.grid_side,
                            self.grid_size,
                            self.img_size,
                            self.key_dim),
        )
        
        self.hierarchical_grid_key_cbam_encoder = nn.Sequential(
            HierarchicalGridCBAMEncoder(self.img_size, 
                                        self.key_dim,
                                        self.heatmap_size*0.1)
        )
        
        # double conv key encoder
        self.double_conv_key_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # Convolutional layer 1
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 32, 3, 1, 1),  # Convolutional layer 2
            nn.Conv2d(32, 32, 3, 1, 1),
            # nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 1),  # Convolutional layer 3
            nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1),
            nn.GroupNorm(1, self.key_dim),
            activation,
            SpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
        
        # key dw encoder
        self.key_dw_encoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1, groups=3),  # Convolutional layer 1
            nn.Conv2d(3, 16, 1),
            # nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 16, 3, 1, 1, groups=16),  # Convolutional layer 2
            nn.Conv2d(16, 32, 1),
            # nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, 32, 3, 1, 1, groups=32),  # Convolutional layer 3
            nn.Conv2d(32, self.key_dim, 1),
            # nn.GroupNorm(1, self.key_dim),
            activation,
            SpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                temperature=self.temperature,
                normalized=True,
            ),  # Spatial Softmax layer
        )
        
        # key decoder
        self.key_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, 2*self.key_dim*self.overlaped_grid_num, bias=False),    # *self.grid_num
            # nn.LayerNorm(2*self.key_dim),
            activation
        )  # Linear layer and activation
        ##################################################
        
        ##################################################
        # img encoder
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # Convolutional layer 1
            # nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 32, 3, 1, 1),  # Convolutional layer 2
            # nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 1),  # Convolutional layer 3
            nn.GroupNorm(1, self.key_dim),
            activation,
        )
        
        
        # img cbam encoder
        self.img_cbam_encoder = nn.Sequential(
            CBAMEncoder(self.key_dim*self.overlaped_grid_num, self.img_size),   # *self.grid_num
        )
        
        # double conv img encoder 
        self.double_conv_img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # Convolutional layer 1
            nn.Conv2d(16, 16, 3, 1, 1),
            # nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 32, 3, 1, 1),  # Convolutional layer 2
            nn.Conv2d(32, 32, 3, 1, 1),
            # nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, self.key_dim, 3, 1, 1),  # Convolutional layer 3
            nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1),
            # nn.GroupNorm(1, self.key_dim),
            activation,
        )
        
        # img dw encoder
        self.img_dw_encoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1, groups=3),  # Convolutional layer 1
            nn.Conv2d(3, 16, 1),
            nn.GroupNorm(1, 16),
            activation,
            nn.Conv2d(16, 16, 3, 1, 1, groups=16),  # Convolutional layer 2
            nn.Conv2d(16, 32, 1),
            nn.GroupNorm(1, 32),
            activation,
            nn.Conv2d(32, 32, 3, 1, 1, groups=32),  # Convolutional layer 3
            nn.Conv2d(32, self.key_dim, 1),
            nn.GroupNorm(1, self.key_dim),
            activation,
        )
        
        # img decoder
        self.img_decoder = nn.Sequential(   # img_decoder
            nn.ConvTranspose2d(self.key_dim*self.overlaped_grid_num, 32, 3, 1, 1),  # Transposed Convolutional layer 1  # *self.grid_num
            # nn.GroupNorm(1, 32),
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 1),  # Transposed Convolutional layer 2
            # nn.GroupNorm(1, 16),
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 1),  # Transposed Convolutional layer 3
            # nn.GroupNorm(1, 3),
            activation,
        )
        
        # double conv img decoder
        self.double_conv_img_decoder = nn.Sequential(   # img_decoder
            nn.ConvTranspose2d(self.key_dim, 32, 3, 1, 1),  # Transposed Convolutional layer 1
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            # nn.GroupNorm(1, 32),
            activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 1),  # Transposed Convolutional layer 2
            nn.ConvTranspose2d(16, 16, 3, 1, 1),
            # nn.GroupNorm(1, 16),
            activation,
            nn.ConvTranspose2d(16, 3, 3, 1, 1),  # Transposed Convolutional layer 3
            nn.ConvTranspose2d(3, 3, 3, 1, 1),
            # nn.GroupNorm(1, 3),
            activation,
        )
        ##################################################
        
        ##################################################
        # state decoder
        self.vec_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, vec_dim, bias=False), 
            # nn.LayerNorm(vec_dim),
            activation
        )  # Linear layer and activation
        
        # Pressure Decoder
        self.press_decoder = nn.Sequential(
            nn.Linear(srnn_hid_dim, press_dim, bias=False), 
            # nn.LayerNorm(press_dim),
            activation
        )  # Linear layer and activation
        ##################################################

        self._weights_init()

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.LSTMCell):
                nn.init.xavier_uniform_(m.weight_ih)
                nn.init.orthogonal_(m.weight_hh)
                if m.bias is not None:
                    nn.init.zeros_(m.bias_ih)
                    nn.init.zeros_(m.bias_hh)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
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
        # xf = self.img_encoder(xi)
        _, xf = cp.checkpoint(self.img_cbam_encoder, xi, use_reentrant=False)   #_coor, 
        
        # _enc_pts, _ = cp.checkpoint(self.overlaped_grid_key_cbam_encoder, xi, use_reentrant=False)
        _enc_pts, _, _ = cp.checkpoint(self.hierarchical_grid_key_cbam_encoder, xi, use_reentrant=False)
        ipdb.set_trace()
        
        # _enc_pts, _ = self.key_encoder(xi)
        # _enc_pts, _ = cp.checkpoint(self.key_cbam_encoder, xi)  #, _xf 
        # _enc_pts, _ = cp.checkpoint(self.grid_key_cbam_encoder, xi, use_reentrant=False)
        enc_pts = _enc_pts.flatten(1,2)
        xk = enc_pts
        
        states = self.hrnn(xk, xv, xp, states)
        
        dec_pts = self.key_decoder(states[0][0])  
        yv = self.vec_decoder(states[1][0])  
        yp = self.press_decoder(states[2][0])  
        _dec_pts = dec_pts.reshape(-1, self.key_dim*self.overlaped_grid_num, 2) # *self.grid_num

        # min_val = _dec_pts.min(dim=1)[0].unsqueeze(dim=1)
        # max_val = _dec_pts.max(dim=1)[0].unsqueeze(dim=1)
        # norm_dec_pts = (_dec_pts - min_val) / (max_val - min_val)
        heatmap = self.key2heatmap(_dec_pts) 
        yf = torch.mul(heatmap, xf)  
        
        # for i in range(32):
        #     plt.imshow(heatmap[0,i].detach().clone().cpu().numpy())
        #     plt.savefig(f"./fig/grid_feat/heat_enc_{i}.png")
        
        yi = cp.checkpoint(self.img_decoder, yf, use_reentrant=False)
        
        del xf
        torch.cuda.empty_cache()
        return yi, yv, yp, enc_pts, dec_pts, yf, states

# grid_enc_pts_list = []
        # for i in range(self.grid_side):
        #     for j in range(self.grid_side):
        #         grid = xi[:,:,i*self.grid_size[0]:(i+1)*self.grid_size[0],
        #                      j*self.grid_size[0]:(j+1)*self.grid_size[0]]
        #         _grid_enc_pts, _ = cp.checkpoint(self.grid_key_cbam_encoder, grid)
        #         _grid_enc_pts = _grid_enc_pts * self.grid_size[0]
        #         _grid_enc_pts[:,:,0] += i*self.grid_size[0]
        #         _grid_enc_pts[:,:,1] += j*self.grid_size[0]
        #         _grid_enc_pts = _grid_enc_pts / self.img_size[0]
        #         grid_enc_pts_list.append(_grid_enc_pts)
        # _grid_enc_pts = torch.stack(grid_enc_pts_list)
        # _grid_enc_pts = torch.einsum("abcd -> bacd", _grid_enc_pts).flatten(1,2)
        



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.activation  = nn.LeakyReLU(negative_slope=0.3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes//ratio, 1, bias=False, padding_mode='replicate'),
                            #    nn.ReLU(),
                                self.activation,
                               nn.Conv2d(in_planes//ratio, in_planes, 1, bias=False, padding_mode='replicate'))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMEncoder(nn.Module):
    def __init__(self, key_dim, img_size
                #  clip_edge=True
                 ):
        super(CBAMEncoder, self).__init__()

        self.key_dim = key_dim
        self.activation  = nn.LeakyReLU(negative_slope=0.3)
        self.img_size = img_size
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, self.key_dim, 3, 1, 1, padding_mode='replicate')
        
        self.ca1 = ChannelAttention(in_planes=16, ratio=4)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(in_planes=32, ratio=4)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(in_planes=self.key_dim, ratio=4)
        self.sa3 = SpatialAttention()
        
        self.norm = nn.GroupNorm(1, self.key_dim)
        self.softmax2d = nn.Softmax2d()

    def forward(self, xi):
        # out = cp.checkpoint(self.conv1, xi)
        out = self.conv1(xi)
        residual = out
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        out += residual
        out = self.activation(out)
        
        # out = cp.checkpoint(self.conv2, out)
        out = self.conv2(out)
        residual = out
        out = self.ca2(out) * out
        out = self.sa2(out) * out
        out += residual
        out = self.activation(out)
        
        # out = cp.checkpoint(self.conv3, out)
        out = self.conv3(out)
        residual = out
        out = self.ca3(out) * out
        out = self.sa3(out) * out
        out += residual
        out = self.activation(out)

        _out = self.softmax2d(out)
        _flatten =_out.flatten(2,3)
        idx = _flatten.argmax(dim=2)
        y = idx // self.img_size[0]
        x = idx % self.img_size[0]
        y = y/self.img_size[0]
        x = x/self.img_size[0]
        pts = torch.stack((x,y)).permute((1,2,0))
        
        # plt.figure()
        # for i in range(8):
        #     plt.imshow(_out[0,i].detach().clone().cpu().numpy())
        #     plt.savefig(f"./fig/custom_re_attn_map_{i}_0")
        
        # ipdb.set_trace()
        
        return pts, _out #coor, 
   
class SimpleCBAMEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 img_size
                 ):
        super(SimpleCBAMEncoder, self).__init__()

        # self.key_dim = key_dim
        self.activation  = nn.LeakyReLU(negative_slope=0.3)
        self.img_size = img_size
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.conv1 = nn.Conv2d(self.input_dim, self.output_dim, 3, 1, 1, padding_mode='replicate')
        
        self.ca1 = ChannelAttention(in_planes=self.output_dim, ratio=4)
        self.sa1 = SpatialAttention()
        
        self.norm = nn.GroupNorm(1, self.output_dim)
        self.softmax2d = nn.Softmax2d()

    def forward(self, xi):
        out = self.conv1(xi)
        residual = out
        out = self.ca1(out) * out
        out = self.sa1(out) * out
        out += residual
        out = self.activation(out)

        _out = self.softmax2d(out)
        _flatten =_out.flatten(2,3)
        idx = _flatten.argmax(dim=2)
        y = idx // self.img_size[0]
        x = idx % self.img_size[0]
        y = y/self.img_size[0]
        x = x/self.img_size[0]
        pts = torch.stack((x,y)).permute((1,2,0))
        
        return pts, _out


class GridCBAMEncoder(nn.Module):
    def __init__(self, 
                 grid_num,
                 grid_side, 
                 grid_size, 
                 img_size, 
                 key_dim
                 ):
        super(GridCBAMEncoder, self).__init__()
        self.grid_num = grid_num
        self.grid_side = grid_side
        self.grid_size = grid_size
        self.img_size = img_size
        self.key_dim = key_dim
        self.grid_cbam_encoder = CBAMEncoder(self.key_dim, grid_size)

    def forward(self, xi):
        grid_enc_pts_list = []
        grid_list = []
        for i in range(self.grid_side):
            for j in range(self.grid_side):
                grid = xi[:,:,i*self.grid_size[0]:(i+1)*self.grid_size[0],
                             j*self.grid_size[0]:(j+1)*self.grid_size[0]]
                _grid_enc_pts, _grid = self.grid_cbam_encoder(grid)
                _grid_enc_pts = _grid_enc_pts * self.grid_size[0]
                _grid_enc_pts[:,:,0] += j*self.grid_size[0]
                _grid_enc_pts[:,:,1] += i*self.grid_size[0]
                _grid_enc_pts = _grid_enc_pts / self.img_size[0]
                grid_enc_pts_list.append(_grid_enc_pts)
                grid_list.append(_grid)
        _grid_enc_pts = torch.stack(grid_enc_pts_list)
        _grid_enc_pts = torch.einsum("abcd -> bacd", _grid_enc_pts).flatten(1,2)
        _grid = torch.stack(grid_list)
        _grid = torch.einsum("abcde -> bacde", _grid).flatten(1,2)
        
        pts = _grid_enc_pts
        _out = _grid
        return pts, _out
    
    


class OverlapedGridCBAMEncoder(nn.Module):
    def __init__(self, 
                 grid_num,
                 grid_side, 
                 grid_size, 
                 img_size, 
                 key_dim
                 ):
        super(OverlapedGridCBAMEncoder, self).__init__()
        self.grid_num = grid_num
        self.grid_side = grid_side
        self.grid_size = grid_size
        self.half_grid_size = [int(self.grid_size[0]/2), int(self.grid_size[0]/2)]
        self.img_size = img_size
        self.key_dim = key_dim
        self.grid_cbam_encoder = CBAMEncoder(self.key_dim, grid_size)

        # 128
        # 64
        # 0~64, 32~64+32, 64~128
        # i*32 i:0~2, 2*2-1=3
        
        # 32
        # 0~32, 16~16+32, 32~32+32, 
        # 4*4-1
        
    def forward(self, xi):
        grid_enc_pts_list = []
        grid_list = []
        for i in range(self.grid_side*self.grid_side -1):
            for j in range(self.grid_side*self.grid_side -1):
                grid = xi[:,:,i*self.half_grid_size[0]:i*self.half_grid_size[0]+self.grid_size[0],
                             j*self.half_grid_size[0]:j*self.half_grid_size[0]+self.grid_size[0]]
                _grid_enc_pts, _grid = self.grid_cbam_encoder(grid)
                _grid_enc_pts = _grid_enc_pts * self.grid_size[0]
                _grid_enc_pts[:,:,0] += j*self.grid_size[0]
                _grid_enc_pts[:,:,1] += i*self.grid_size[0]
                _grid_enc_pts = _grid_enc_pts / self.img_size[0]
                grid_enc_pts_list.append(_grid_enc_pts)
                grid_list.append(_grid)
        _grid_enc_pts = torch.stack(grid_enc_pts_list)
        _grid_enc_pts = torch.einsum("abcd -> bacd", _grid_enc_pts).flatten(1,2)
        _grid = torch.stack(grid_list)
        _grid = torch.einsum("abcde -> bacde", _grid).flatten(1,2)
        
        pts = _grid_enc_pts
        _out = _grid
        return pts, _out





class SimpleGridCBAMEncoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                #  grid_num,
                 grid_side, 
                 grid_size, 
                 img_size, 
                 heatmap_size,
                 ):
        super(SimpleGridCBAMEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.grid_num = grid_num
        self.grid_side = grid_side
        self.grid_size = grid_size
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.simple_grid_cbam_encoder = SimpleCBAMEncoder(self.input_dim, self.output_dim, self.grid_size)
        self.key2heatmap = nn.Sequential(
            InverseSpatialSoftmax(
                width=self.grid_size[0],
                height=self.grid_size[1],
                heatmap_size=self.heatmap_size,
                normalized=True,
            )
        )
        
    def forward(self, xi):
        grid_enc_pts_list = []
        grid_list = []
        full_img = torch.zeros((xi.shape[0], self.output_dim, self.img_size[0], self.img_size[0])).to(xi.device)
        full_heatmap = torch.zeros((xi.shape[0], self.output_dim, self.img_size[0], self.img_size[0])).to(xi.device)
        for i in range(self.grid_side):
            for j in range(self.grid_side):
                grid = xi[:,:,i*self.grid_size[0]:(i+1)*self.grid_size[0],
                             j*self.grid_size[0]:(j+1)*self.grid_size[0]]
                _grid_enc_pts, _grid = self.simple_grid_cbam_encoder(grid)
                
                full_img[:,:,i*self.grid_size[0]:(i+1)*self.grid_size[0],
                             j*self.grid_size[0]:(j+1)*self.grid_size[0]] = _grid
                heatmap = self.key2heatmap(_grid_enc_pts)
                full_heatmap[:,:,i*self.grid_size[0]:(i+1)*self.grid_size[0],
                        j*self.grid_size[0]:(j+1)*self.grid_size[0]] = heatmap
                
                _grid_enc_pts = _grid_enc_pts * self.grid_size[0]
                _grid_enc_pts[:,:,0] += j*self.grid_size[0]
                _grid_enc_pts[:,:,1] += i*self.grid_size[0]
                _grid_enc_pts = _grid_enc_pts / self.img_size[0]
                grid_enc_pts_list.append(_grid_enc_pts)
                grid_list.append(_grid)
        _grid_enc_pts = torch.stack(grid_enc_pts_list)
        _grid_enc_pts = torch.einsum("abcd -> bacd", _grid_enc_pts).flatten(1,2)
        _grid = torch.stack(grid_list)
        _grid = torch.einsum("abcde -> bacde", _grid).flatten(1,2)
                
        pts = _grid_enc_pts
        grid = _grid
        return pts, grid, full_img








class HierarchicalGridCBAMEncoder(nn.Module):
    def __init__(self, 
                #  grid_num,
                #  grid_side, 
                #  grid_size, 
                 img_size, 
                 key_dim,
                 heatmap_size
                 ):
        super(HierarchicalGridCBAMEncoder, self).__init__()
        # self.grid_num = grid_num
        # self.grid_side = grid_side
        # self.grid_size = grid_size
        self.img_size = img_size
        self.key_dim = key_dim
        self.heatmap_size = heatmap_size
        # 128
        self.grid_side1 = 8
        self.grid_side2 = 4
        self.grid_side3 = 1
        self.grid_size1 = [int(img_size[0]/self.grid_side1), int(img_size[0]/self.grid_side1)]
        self.grid_size2 = [int(img_size[0]/self.grid_side2), int(img_size[0]/self.grid_side2)]
        self.grid_size3 = [int(img_size[0]/self.grid_side3), int(img_size[0]/self.grid_side3)]
        
        self.simple_grid_cbam_encoder1 = SimpleGridCBAMEncoder(5, 16, 
                                                               self.grid_side1, self.grid_size1, img_size, self.heatmap_size)
        self.simple_grid_cbam_encoder2 = SimpleGridCBAMEncoder(16, 32, 
                                                               self.grid_side2, self.grid_size2, img_size, self.heatmap_size)
        self.simple_grid_cbam_encoder3 = SimpleGridCBAMEncoder(32, self.key_dim, 
                                                               self.grid_side3, self.grid_size3, img_size, self.heatmap_size)
        
        self.key2heatmap = nn.Sequential(
            InverseSpatialSoftmax(
                width=self.img_size[0],
                height=self.img_size[1],
                heatmap_size=self.heatmap_size,
                normalized=True,
            )
        )
        self.softmax2d = nn.Softmax2d()
        
    def forward(self, xi):
        # 0~1
        x_coords = torch.linspace(0, 1, steps=self.img_size[0]).unsqueeze(0).repeat(self.img_size[0], 1)
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).repeat(xi.shape[0], 1, 1, 1).to(xi.device)
        
        y_coords = torch.linspace(-1, 1, steps=self.img_size[0]).unsqueeze(1).repeat(1, self.img_size[0])
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).repeat(xi.shape[0], 1, 1, 1).to(xi.device)
        
        _xi = torch.cat([xi, x_coords, y_coords], dim=1)
        _pts, _grid, _full = self.simple_grid_cbam_encoder1(_xi)
        heatmap = self.key2heatmap(_pts) 
        # mean_heatmap = heatmap.mean(dim=1).unsqueeze(dim=1)
        # mean_val = heatmap.mean()
        # threshold = mean_val
        # thresh_heatmap = torch.where(mean_heatmap < threshold, torch.zeros_like(mean_heatmap), mean_heatmap)
        # _full = self.softmax2d(torch.mul(_full, mean_heatmap))
        
        # _xi = torch.cat([xi, mean_heatmap], dim=1)
        # plt.imshow(heatmap[0,i].detach().clone().cpu().numpy())
        # plt.savefig(f"./fig/h_grid_feat/overlap_heatmap_{i}.png")

        _pts, _grid, _full = self.simple_grid_cbam_encoder2(_full)
        # heatmap = self.key2heatmap(_pts) 
        # mean_heatmap = heatmap.mean(dim=1).unsqueeze(dim=1)
        # mean_val = heatmap.mean()
        # threshold = mean_val
        # thresh_heatmap = torch.where(mean_heatmap < threshold, torch.zeros_like(mean_heatmap), mean_heatmap)
        # _full = self.softmax2d(torch.mul(_full, mean_heatmap))
        # heatmap = self.key2heatmap(_pts) 
        # mean_heatmap = heatmap.mean(dim=1).unsqueeze(dim=1)
        # _xi = torch.cat([xi, mean_heatmap], dim=1)
        
        _pts, _grid, _full = self.simple_grid_cbam_encoder3(_full)
        ipdb.set_trace()
        
        return _pts, _grid, _full

"""
for i in range(32):
        plt.imshow(_full[0,i].detach().clone().cpu().numpy())
        plt.savefig(f"./fig/h_grid_feat/full_img_lay2_{i}.png")
"""

