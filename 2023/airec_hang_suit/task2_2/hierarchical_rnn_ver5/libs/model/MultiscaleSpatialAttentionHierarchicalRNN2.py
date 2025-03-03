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
import numpy as np
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import KMeans
from pyclustering.cluster import gmeans, xmeans
from torch_kmeans import KMeans

import logging
logging.basicConfig(level=logging.INFO)

from eipl.layer import SpatialSoftmax, InverseSpatialSoftmax
from eipl.utils import get_activation_fn
from eipl.layer import GridMask
from eipl.utils import normalization

try:
    from libs.layer import HierachicalRNNCell
    from libs.layer import MultiscaleSpatialAttention
    # from libs.utils import XMeans
except:
    from layer import HierarchicalRNNCell
    from layer import MSA, Imgcropper, AutoEncoder
    # from utils import XMeans

import numpy, warnings
numpy.warnings = warnings

class MSAHSARNN2(nn.Module):
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
        device="cpu"
    ):
        super(MSAHSARNN2, self).__init__()

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
            srnn_input_dims={"k": 2*self.key_dim, "v": vec_dim, "p": press_dim},    # *self.grid_num    *self.overlaped_grid_num
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
        self.msa = MSA(att_num=self.key_dim, 
                       img_h=img_size[0], 
                       img_w=img_size[0], 
                       temperature=self.temperature, 
                       device=device,
                       type="concat")
        self.imgcroper = Imgcropper(att_num=self.key_dim, 
                                    img_h=img_size[0], 
                                    img_w=img_size[0], 
                                    temp=self.heatmap_size, 
                                    gpu=device)
        self.autodecoder = AutoEncoder()
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
        self.key_cbam_encoder = HeatCBAMEncoder(self.key_dim, self.img_size)
        
        self.key_heat_encoder = HeatEncoder(self.key_dim, self.img_size)
        
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
            nn.Linear(srnn_hid_dim, 2*self.key_dim, bias=False),    # *self.grid_num    *self.overlaped_grid_num
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
        self.img_cbam_encoder = CBAMEncoder(self.key_dim, self.img_size)
        
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
            nn.ConvTranspose2d(self.key_dim, 32, 3, 1, 1),  # Transposed Convolutional layer 1  # *self.grid_num    *self.overlaped_grid_num
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

    def forward(self, xi, xv, xp, states=None, prev_heatmap=None, prev_select_pts=None, time=0, epoch=0):
        _, xf = cp.checkpoint(self.img_cbam_encoder, xi, use_reentrant=False)   #_coor, 
        
        # _xf, _enc_pts = cp.checkpoint(self.msa, xi)
        
        _enc_pts, _xf, feat_list = cp.checkpoint(self.key_cbam_encoder, xi, prev_heatmap, prev_select_pts, time, epoch, use_reentrant=False)
        # _enc_pts, _xf, feat_list = cp.checkpoint(self.key_heat_encoder, xi, prev_heatmap, time, epoch, use_reentrant=False)
        prev_heatmap = self.key2heatmap(_enc_pts) 
        prev_select_pts = _enc_pts
        # prev_mean_heatmap = prev_heatmap.mean(dim=1).unsqueeze(dim=1).repeat(1,8,1,1)
        
        # _enc_pts, _ = cp.checkpoint(self.overlaped_grid_key_cbam_encoder, xi, use_reentrant=False)
        # _enc_pts, _, _ = cp.checkpoint(self.hierarchical_grid_key_cbam_encoder, xi, use_reentrant=False)
        
        # _enc_pts, _ = self.key_encoder(xi)
        # _enc_pts, _ = cp.checkpoint(self.key_cbam_encoder, xi)  #, _xf 
        # _enc_pts, _ = cp.checkpoint(self.grid_key_cbam_encoder, xi, use_reentrant=False)
        
        enc_pts = _enc_pts.flatten(1,2)  # _enc_pts.flatten(1,2)
        xk = enc_pts
        states = self.hrnn(xk, xv, xp, states)
        
        dec_pts = self.key_decoder(states[0][0])  
        yv = self.vec_decoder(states[1][0])  
        yp = self.press_decoder(states[2][0])
        _dec_pts = dec_pts.reshape(-1, self.key_dim, 2) # *self.grid_num    *self.overlaped_grid_num
        ######
        # yf = cp.checkpoint(self.imgcroper, xi, dec_pts)
        # yi = cp.checkpoint(self.autodecoder, yf)
        # xf = cp.checkpoint(self.autodecoder, xi)
        ######

        # min_val = _dec_pts.min(dim=1)[0].unsqueeze(dim=1)
        # max_val = _dec_pts.max(dim=1)[0].unsqueeze(dim=1)
        # norm_dec_pts = (_dec_pts - min_val) / (max_val - min_val)
        heatmap = self.key2heatmap(_dec_pts) 
        # ipdb.set_trace()
        yf = torch.mul(heatmap, xf)  
        yi = cp.checkpoint(self.img_decoder, yf, use_reentrant=False)
        
        del xf    #, _xf
        torch.cuda.empty_cache()
        return yi, yv, yp, enc_pts, dec_pts, yf, states, prev_heatmap, prev_select_pts  #, _xf


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
        
        # ################
        # _out = self.softmax2d(out)
        # clip_out = _out[:,:,3:-3, 3:-3]
        # batch_size, channels, height, width = clip_out.size()
        # _flatten = clip_out.flatten(2,3)
        
        # if t == 0:
        #     _, idxs = _flatten.topk(20, dim=2)  # 値が高い順に10個の要素のインデックスを取得
        #     idxs += 3
        #     y, x = idxs // width, idxs % width  # y, x 座標に変換
        #     _y = y/128
        #     _x = x/128
        #     pts = torch.stack((_x,_y)).permute((1,2,3,0))
        
        #     model = KMeans(n_clusters=8)
        #     _pts = pts.flatten(1,2)
        #     result = model(_pts)
        #     centers = result.centers
        #     pts = centers
        
        # else:
        #     idx = _flatten.argmax(dim=2)
        #     idxs += 3
        #     y = idx // self.img_size[0]
        #     x = idx % self.img_size[0]
        #     _y = y/self.img_size[0]
        #     _x = x/self.img_size[0]
        #     pts = torch.stack((_x,_y)).permute((1,2,3,0))
        # ################
        
        _out = self.softmax2d(out)
        clip_out = _out[:,:,3:-3, 3:-3]
        batch_size, channels, height, width = clip_out.size()
        _flatten = clip_out.flatten(2,3)
        idx = _flatten.argmax(dim=2)
        idx += 3
        y = idx // self.img_size[0]
        x = idx % self.img_size[0]
        _y = y/self.img_size[0]
        _x = x/self.img_size[0]
        pts = torch.stack((_x,_y)).permute((1,2,0))
        
        return pts, _out #coor, # b*key_dim*10*2


class HeatCBAMEncoder(nn.Module):
    def __init__(self, key_dim, img_size):
        super(HeatCBAMEncoder, self).__init__()

        # model = MSAHSARNN()
        self.key_dim = key_dim
        self.img_size = img_size
        
        self.activation  = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        self.conv2 = nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        self.conv3 = nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        
        self.ca1 = ChannelAttention(in_planes=self.key_dim, ratio=4)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(in_planes=self.key_dim, ratio=4)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(in_planes=self.key_dim, ratio=4)
        self.sa3 = SpatialAttention()
        
        self.norm = nn.GroupNorm(1, self.key_dim)
        self.softmax2d = nn.Softmax2d()
        
    def get_gray_img(self,rgb_img):
        gray = 0.114*rgb_img[:,0,:,:] + 0.587*rgb_img[:,1,:,:] + 0.299*rgb_img[:,2,:,:]
        gray = torch.unsqueeze(gray,1)
        return gray
    
    def sort_cluster_centers(self, centers):
        # 各クラスタ中心の総和を計算し、その順序でソート
        centers_sum = torch.sum(centers, dim=-1)
        sorted_indices = torch.argsort(centers_sum, dim=1)
        sorted_centers = torch.stack([centers[i, sorted_indices[i]] for i in range(centers.size(0))])
        return sorted_centers

        
    def forward(self, xi, prev_heatmap, prev_select_pts, time, epoch):
        gray_xi = self.get_gray_img(xi).repeat(1,8,1,1)
        
        # x_coords = torch.linspace(0, 1, steps=self.img_size[0]).unsqueeze(0).repeat(self.img_size[0], 1)
        # x_coords = x_coords.unsqueeze(0).unsqueeze(0).repeat(xi.shape[0], 1, 1, 1).to(xi.device)
        # y_coords = torch.linspace(-1, 1, steps=self.img_size[0]).unsqueeze(1).repeat(1, self.img_size[0])
        # y_coords = y_coords.unsqueeze(0).unsqueeze(0).repeat(xi.shape[0], 1, 1, 1).to(xi.device)
        # _xi = torch.cat([xi, x_coords, y_coords], dim=1)
        
        feat_list = []
        
        if prev_heatmap == None:
            prev_heatmap = 1.0
        # else:
        #     prev_heatmap[prev_heatmap < 0.1] = 0.0
        
        # out = torch.mul(prev_heatmap, gray_xi)
        out = prev_heatmap + gray_xi
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        # out = xi
        out = self.conv1(out)    
        # residual = out    
        # out = self.ca1(out) * out
        # out = self.sa1(out) * out
        # out += residual
        out = self.activation(out)
        
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        # out = torch.mul(prev_heatmap, out)
        out += prev_heatmap
        out = self.conv2(out)
        # residual = out
        # out = self.ca2(out) * out
        # out = self.sa2(out) * out
        # out += residual
        out = self.activation(out)
        
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        # out = torch.mul(prev_heatmap, out)
        out += prev_heatmap
        out = self.conv3(out)
        # residual = out
        # out = self.ca3(out) * out
        # out = self.sa3(out) * out
        # out += residual
        out = self.activation(out)
                
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
            
        """
        for i in range(8):
            plt.figure()
            plt.imshow(out[0,i].detach().clone().cpu().numpy())
            plt.savefig(f"./fig/hoge4/_out_feat_lay0_{i}.png")
            plt.close()
            
            # plt.savefig(f"./fig/_prev_heatmap_{i}.png")
            
        for i in range(8):
            plt.figure()
            plt.imshow(out[0,i].detach().clone().cpu().numpy())
            plt.savefig(f"./fig/_out_feat_lay2_{i}.png")
            plt.close()
            
        """
        
        ################
        _out = self.softmax2d(out)
        clip_out = _out[:,:,3:-3, 3:-3]
        batch_size, channels, height, width = clip_out.size()
        _flatten = clip_out.flatten(2,3)
        
        _, idxs = _flatten.topk(100, dim=2)  # 値が高い順に10個の要素のインデックスを取得
        idxs += 3
        y, x = idxs // width, idxs % width  # y, x 座標に変換
        _y = y/128
        _x = x/128
        full_pts = pts = torch.stack((_x,_y)).permute((1,2,3,0))
        
        torch.manual_seed(42)
        model = KMeans(n_clusters=8, mode='euclidean')
        _pts = pts.flatten(1,2)
        result = model(_pts)
        centers = result.centers
        select_pts = pts = centers
        
        if prev_select_pts == None:
            pts = self.sort_cluster_centers(select_pts)
            
        else:
            pts = torch.empty_like(select_pts)
            for batch in range(batch_size):
                # データAとデータBのペア間の距離行列を計算
                cost_matrix = torch.cdist(prev_select_pts[batch], select_pts[batch])

                # ハンガリアンアルゴリズムを使用して最適な対応を求める
                row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().clone().cpu().numpy()) 

                # 最適な対応に基づいてデータBを並び替える
                pts[batch] = select_pts[batch, col_ind]
        
        full_pts = (full_pts*128).flatten(1,2).to(int).detach().clone().cpu().numpy()
        select_pts = (select_pts*128).to(int).detach().clone().cpu().numpy()
        
        # if epoch == 2 and time%5 == 0:
        # if epoch == 10 and time%5 == 0:
        #     plt.figure()
        #     plt.imshow(xi[0].permute(1,2,0).detach().clone().cpu().numpy(), origin='upper')
        #     plt.scatter(full_pts[0,:,0], full_pts[0,:,1])
        #     plt.scatter(select_pts[0,:,0], select_pts[0,:,1])
        #     plt.savefig(f"./fig/sample_key_trend/scatter_plot_{epoch}_{time}.png")
        #     plt.close()
            
            # if epoch == 2:
            #     ipdb.set_trace()
        # else:
            # idx = _flatten.argmax(dim=2)
            # idx += 3
            # y = idx // self.img_size[0]
            # x = idx % self.img_size[0]
            # _y = y/self.img_size[0]
            # _x = x/self.img_size[0]
            # pts = torch.stack((_x,_y)).permute((1,2,0))
            
            # select_pts = (pts*128).to(int).detach().clone().cpu().numpy()
            # if time == 1:
            #     plt.figure()
            #     plt.imshow(xi[0].permute(1,2,0).detach().clone().cpu().numpy(), origin='upper')
            #     plt.scatter(select_pts[0,:,0], select_pts[0,:,1])
            #     plt.savefig(f"./fig/_scatter_plot_{epoch}.png")
            #     plt.close()
        # ################
        
        
        # _, idxs = _flatten.topk(20, dim=2)  # 値が高い順に10個の要素のインデックスを取得
        # idxs += 3
        # y, x = idxs // width, idxs % width  # y, x 座標に変換
        # _y = y/128
        # _x = x/128
        # pts = torch.stack((_x,_y)).permute((1,2,3,0))
        
        # model = KMeans(n_clusters=8)
        # _pts = pts.flatten(1,2)
        # result = model(_pts)
        # centers = result.centers
        # pts = centers
        
        return pts, _out, feat_list #coor, # b*key_dim*10*2




class HeatEncoder(nn.Module):
    def __init__(self, key_dim, img_size):
        super(HeatEncoder, self).__init__()

        # model = MSAHSARNN()
        self.key_dim = key_dim
        self.img_size = img_size
        
        self.activation  = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv2d(3, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        self.conv2 = nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        self.conv3 = nn.Conv2d(self.key_dim, self.key_dim, 3, 1, 1, bias=False, padding_mode='replicate')
        
        self.norm = nn.GroupNorm(1, self.key_dim)
        self.softmax2d = nn.Softmax2d()
        
    
    def sort_cluster_centers(self, centers):
        # 各クラスタ中心の総和を計算し、その順序でソート
        centers_sum = torch.sum(centers, dim=-1)
        sorted_indices = torch.argsort(centers_sum, dim=1)
        sorted_centers = torch.stack([centers[i, sorted_indices[i]] for i in range(centers.size(0))])
        return sorted_centers
    
    def forward(self, xi, prev_heatmap, time, epoch):        
        feat_list = []
        
        if prev_heatmap == None:
            prev_heatmap = 1.0
        
        out = self.conv1(xi) 
        out = self.activation(out)
        
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        # out = torch.mul(prev_heatmap, out)
        out = self.conv2(out)
        out = self.activation(out)
        
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        # out = torch.mul(prev_heatmap, out)
        out = self.conv3(out)
        out = self.activation(out)
                
        feat_list.append(out)
        if epoch == 2 and time == 1:
            ipdb.set_trace()
        
        ################
        _out = self.softmax2d(out)
        clip_out = _out[:,:,3:-3, 3:-3]
        batch_size, channels, height, width = clip_out.size()
        _flatten = clip_out.flatten(2,3)
        
        _, idxs = _flatten.topk(100, dim=2)  # 値が高い順に10個の要素のインデックスを取得
        idxs += 3
        y, x = idxs // width, idxs % width  # y, x 座標に変換
        _y = y/128
        _x = x/128
        full_pts = pts = torch.stack((_x,_y)).permute((1,2,3,0))
        
        torch.manual_seed(42)
        model = KMeans(n_clusters=8, mode='euclidean')
        _pts = pts.flatten(1,2)
        result = model(_pts)
        centers = result.centers
        select_pts = pts = centers
        
        pts = self.sort_cluster_centers(select_pts)
        # if prev_select_pts == None:
        # else:
        #     # pts = sorted(changed_coords, key=lambda x: [y for y in original_coords if abs(y[0] - x[0]) < 1 and abs(y[1] - x[1]) < 1][0])
        #     # print()
            
        full_pts = (full_pts*128).flatten(1,2).to(int).detach().clone().cpu().numpy()
        select_pts = (select_pts*128).to(int).detach().clone().cpu().numpy()
        
        # # if epoch == 2 and time%5 == 0:
        # if epoch == 10 and time%5 == 0:
        #     plt.figure()
        #     plt.imshow(xi[0].permute(1,2,0).detach().clone().cpu().numpy(), origin='upper')
        #     plt.scatter(full_pts[0,:,0], full_pts[0,:,1])
        #     plt.scatter(select_pts[0,:,0], select_pts[0,:,1])
        #     plt.savefig(f"./fig/sample_key_trend/scatter_plot_{epoch}_{time}.png")
        #     plt.close()
        
        return pts, _out, feat_list
    










"""
for i in range(8):
    plt.figure()
    plt.imshow(out[0,i].detach().clone().cpu().numpy())
    plt.savefig(f"./fig/simple_encoder/_out_feat_lay2_{i}.png")
    plt.close()

"""















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
        ipdb.set_trace()
        heatmap = self.key2heatmap(_pts) 
        # mean_heatmap = heatmap.mean(dim=1).unsqueeze(dim=1)
        # mean_val = heatmap.mean()
        # threshold = mean_val
        # thresh_heatmap = torch.where(mean_heatmap < threshold, torch.zeros_like(mean_heatmap), mean_heatmap)
        # _full = self.softmax2d(torch.mul(_full, mean_heatmap))
        
        # _xi = torch.cat([xi, mean_heatmap], dim=1)
        # plt.imshow(_xf[0,i].detach().clone().cpu().numpy())
        # plt.savefig(f"./fig/msa_feat/msa_feat_{i}.png")

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
for i in range(8):
        plt.figure()
        plt.imshow(out[0,i].detach().clone().cpu().numpy())
        plt.savefig(f"./fig/cba_feat_{i}.png")
        plt.close()
"""

# x = _pts[:,1].detach().clone().cpu().numpy()
# y = _pts[:,0].detach().clone().cpu().numpy()
# plt.figure()
# plt.imshow(out[0,i].detach().clone().cpu().numpy())
# plt.savefig(f"./fig/enc_feat_{i}.png")
# plt.close()









"""
[[ 64,  40],                                                                                                                                                                           
        [ 89,  40],                                                                                                                                                                           
        [ 62,  40],                                                                                                                                                                           
        [ 70,  40],                                                                                                                                                                           
        [ 97,  40],                                                                                                                                                                           
        [ 67,  40],                                                                                                                                                                           
        [ 91,  40],                                                                                                                                                                           
        [ 76,  40],                                                                                                                                                                           
        [ 93,  40],                                                                                                                                                                           
        [ 71,  40],                                                                                                                                                                           
        [102,  14],                                                                                                                                                                           
        [102,  17],                                                                                                                                                                           
        [ 69,  30],                                                                                                                                                                           
        [102,  34],                                                                                                                                                                           
        [102,  21],                                                                                                                                                                           
        [102,  25],                                                                                                                                                                           
        [102,  27],                                                                                                                                                                           
        [ 42,  14],                                                                                                                                                                           
        [ 90,  29],                                                                                                                                                                           
        [ 65,  24],                                                                                                                                                                           
        [103,  30], 
        [103,  22],                                                                                                                                                                           
        [103,  14],                                                                                                                                                                           
        [103,  24],                                                                                                                                                                           
        [103,  29],                                                                                                                                                                           
        [103,  27],
        [103,  23],
        [103,  18],
        [103,  31],
        [103,  15],
        [ 94,  38],
        [ 73,  38],
        [ 75,  38],
        [ 64,  38],
        [ 90,  38],
        [ 66,  38],
        [101,  38],
        [ 97,  38],
        [ 68,  38],
        [102,  38],
        [102,  32],
        [102,  19],
        [102,  22],
        [102,  30],
        [102,  17],
        [102,  16],
        [102,  33],
        [102,  25],
        [102,  29],
        [102,  34],
        [ 26,  12],
        [ 26,  13],
        [ 26,  14],
        [ 26,  22],
        [ 26,  27],
        [ 26,  21],
        [  4,  29],
        [ 26,  28],
        [ 26,  19],
        [ 26,  17],
        [ 69,  38],
        [ 78,  38],
        [ 66,  38],
        [ 65,  38],
        [ 72,  38],
        [ 67,  38],
        [ 93,  38],
        [ 82,  38],
        [ 68,  38],
        [ 83,  38],
        [102,  37],
        [ 87,  37],
        [ 61,  37],
        [ 70,  37],
        [103,  37],
        [ 99,  37],
        [ 73,  37],
        [ 93,  37],
        [ 90,  37],
        [ 91,  37]]
"""


# plt.figure()
# for centers in centers_list:
#     for center in centers:
#         plt.scatter(center)
# plt.savefig("./fig/claster_center.png")
