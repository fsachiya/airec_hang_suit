#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from eipl.layer import CoordConv2d, AddCoords
import ipdb
import matplotlib.pyplot as plt
import torch.nn.functional as F


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
        # print("ca", out.min(), out.max())
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
        # print("sa", x.min(), x.max())
        return self.sigmoid(x)



class AbSCAE(nn.Module):
    def __init__(self):
        super(AbSCAE, self).__init__()

        # encoder
        self.encoder = AbSCAEEncoder()
        
        # decoder
        self.decoder = AbSCAEDecoder()
    
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

    def forward(self, x, prev_enc_xf_last=None, prev_dec_xfs=None, epoch=0, time=0):
        xf, enc_xf_last = cp.checkpoint(self.encoder, x, prev_dec_xfs, prev_enc_xf_last, epoch, time, use_reentrant=False)
        y, dec_xfs, pts = cp.checkpoint(self.decoder, xf, epoch, time, use_reentrant=False)
        
        return y, enc_xf_last, dec_xfs, pts
    

class AbSCAEEncoder(nn.Module):
    def __init__(self):
        super(AbSCAEEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, padding_mode='replicate')
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, padding_mode='replicate')
        self.conv3 = nn.Conv2d(32, 8, 3, 1, 1, padding_mode='replicate')
        
        self.ca1 = ChannelAttention(in_planes=16, ratio=4)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(in_planes=32, ratio=4)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(in_planes=8, ratio=4)
        self.sa3 = SpatialAttention()
        
        self.softmax2d = nn.Softmax2d()
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.lrelu  = nn.LeakyReLU(negative_slope=0.3)
        
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=8)
        
        self.sigmoid = nn.Sigmoid()
        self.img_size = 128
    
    def forward(self, x, prev_dec_xfs, prev_enc_xf_last, epoch, time):
        if not prev_dec_xfs==None:
            dec_xf1, dec_xf2, dec_xf3 = prev_dec_xfs[0], prev_dec_xfs[1], prev_dec_xfs[2]
        
        x = self.conv1(x)    
        residual = x    
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        x += residual
        x = self.bn1(x)
        x = self.lrelu(x)
        # print("enclay1", x.min(), x.max())
        if not prev_dec_xfs==None:
            _channel_map = self.gap(dec_xf3)
            channel_map = self.sigmoid(_channel_map)
            x = torch.mul(x, channel_map)
            _layer_map = torch.mean(x, dim=1, keepdim=True)
            layer_map = self.sigmoid(_layer_map)
            x = torch.mul(x, layer_map)
            # x += dec_xf3
            # # x = self.bn1(x)
            # x = self.lrelu(x)
            # heatmap = self.softmax2d(dec_xf3)
            # x = torch.mul(x, heatmap)
        if torch.any(x.isnan()):
            ipdb.set_trace()
        
        x = self.conv2(x)    
        residual = x    
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        x += residual
        x = self.bn2(x)
        x = self.lrelu(x)
        # print("enclay2", x.min(), x.max())
        if not prev_dec_xfs==None:
            _channel_map = self.gap(dec_xf2)
            channel_map = self.sigmoid(_channel_map)
            x = torch.mul(x, channel_map)
            _layer_map = torch.mean(x, dim=1, keepdim=True)
            layer_map = self.sigmoid(_layer_map)
            x = torch.mul(x, layer_map)
            # heatmap = self.softmax2d(dec_xf2)
            # x = torch.mul(x, heatmap)
            # x += dec_xf2
            # # x = self.bn2(x)
            # x = self.lrelu(x)
            # heatmap = self.softmax2d(dec_xf2)
            # x = torch.mul(x, heatmap)
        if torch.any(x.isnan()):
            ipdb.set_trace()
        
        x = self.conv3(x)    
        residual = x    
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        x += residual
        x = self.bn3(x)
        x = self.lrelu(x)
        # print("enclay3", x.min(), x.max())
        if not prev_dec_xfs==None:
            _channel_map = self.gap(dec_xf1)
            channel_map = self.sigmoid(_channel_map)
            x = torch.mul(x, channel_map)
            _layer_map = torch.mean(x, dim=1, keepdim=True)
            layer_map = self.sigmoid(_layer_map)
            x = torch.mul(x, layer_map)
            # heatmap = self.softmax2d(dec_xf1)
            # x = torch.mul(x, heatmap)
            # x += dec_xf1
            # # x = self.bn3(x)
            # x = self.lrelu(x)
            # heatmap = self.softmax2d(dec_xf1)
            # x = torch.mul(x, heatmap)
        if torch.any(x.isnan()):
            ipdb.set_trace()
        
        # if time == 5:
        #     ipdb.set_trace()
        
        enc_xf_last = x
        
        if not prev_enc_xf_last==None:
            # channel_map = self.gap(prev_enc_xf_last)
            _channel_map = self.gap(prev_enc_xf_last)
            channel_map = self.sigmoid(_channel_map)
            x = torch.mul(x, channel_map)
        else:
            # channel_map = self.gap(enc_xf_last)
            _channel_map = self.gap(enc_xf_last)
            channel_map = self.sigmoid(_channel_map)
            x = torch.mul(x, channel_map)
        
        # if time == 5:
        #     ipdb.set_trace()
        
        return x, enc_xf_last.detach().clone()
    

class AbSCAEDecoder(nn.Module):
    def __init__(self):
        super(AbSCAEDecoder, self).__init__()
        
        # self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        # self.conv3 = nn.Conv2d(64, 8, 3, 2, 1)
        
        self.deconv1 = nn.ConvTranspose2d(8, 32, 3, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 3, 1, 1)
        self.deconv3 = nn.ConvTranspose2d(16, 3, 3, 1, 1)
        
        self.ca1 = ChannelAttention(in_planes=8, ratio=4)
        self.sa1 = SpatialAttention()
        self.ca2 = ChannelAttention(in_planes=32, ratio=4)
        self.sa2 = SpatialAttention()
        self.ca3 = ChannelAttention(in_planes=16, ratio=4)
        self.sa3 = SpatialAttention()
        
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        self.softmax2d = nn.Softmax2d()
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.lrelu  = nn.LeakyReLU(negative_slope=0.3)
        
        self.img_size = 128
    
    def forward(self, x, epoch, time):
        dec_xfs = []
        
        # _x = self.ca1(x) * x
        # _x = self.sa1(_x) * _x
        dec_xfs.append(x.detach().clone())
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        
        _x = self.softmax2d(x)
        pad_x = torch.zeros_like(_x)
        pad_x[:,:,3:-3, 3:-3] = _x[:,:,3:-3, 3:-3]
        _flatten =pad_x.flatten(2,3)
        idx = _flatten.argmax(dim=2)
        coord_y = idx // self.img_size
        coord_x = idx % self.img_size
        coord_y = coord_y/self.img_size
        coord_x = coord_x/self.img_size
        pts = torch.stack((coord_x,coord_y)).permute((1,2,0))
        
        ipdb.set_trace()
        
        # if time == 5:
        #     ipdb.set_trace()
        
        # print("declay1", x.min(), x.max())
        
        # _x = self.ca2(x) * x
        # _x = self.sa2(_x) * _x
        dec_xfs.append(x.detach().clone())
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        # print("declay2", x.min(), x.max())
        
        # if time == 5:
        #     ipdb.set_trace()

        # _x = self.ca3(x) * x
        # _x = self.sa3(_x) * _x
        dec_xfs.append(x.detach().clone())
        x = self.deconv3(x)
        # x = self.norm(x)
        x = self.lrelu(x)
        # print("declay3", x.min(), x.max())
        
        # if time == 5:
        #     ipdb.set_trace()
        
        return x, dec_xfs, pts

# for i in range(16):
#     plt.figure()
#     plt.imshow(x[0,i].detach().clone().cpu().numpy())
#     plt.savefig(f"./fig/abs9/dec_feat{i}.png")
#     plt.close()

# for i in range(32):
#     x = pts_data[0,:,i,0].detach().clone().cpu().numpy()*128
#     y = pts_data[0,:,i,1].detach().clone().cpu().numpy()*128
#     plt.figure()
#     plt.plot(x,y)
#     plt.xlim(0,128)
#     plt.ylim(0,128)
#     plt.savefig(f"./fig/pts_trend_{i}.png")
#     plt.close()
