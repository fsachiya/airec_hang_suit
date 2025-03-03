#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import sys
import torch
import torch.nn as nn
from torch.autograd import detect_anomaly, set_detect_anomaly
import numpy as np
from typing import Dict, List, Tuple
from eipl.utils import LossScheduler, tensor2numpy
import ipdb
import gc
import time
import matplotlib.pyplot as plt

try:
    from libs.utils import moving_average
except:
    sys.path.append("./libs/")
    from utils import moving_average


class simple_fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, 
                 model, 
                 optimizer, 
                 scaler, 
                 loss_w_dic:Dict[str, float]={"i": 0.1, "k": 1.0, "v": 1.0, "p": 1.0}, 
                 device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.loss_w_dic = loss_w_dic
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)
        self.criterion = nn.MSELoss()
        
        self.layer_norm = nn.LayerNorm(207-10).to(self.device)
        self.attn = nn.MultiheadAttention(
            embed_dim=20, 
            num_heads=2,
            bias=False,
            batch_first=True).to(self.device)
        self.maxpool = nn.MaxPool1d(20)

        
    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                # 'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            savename,
        )
        
    def tensor_moving_average(self,
                              tensor_data,
                              kernel_size=7,
                              padding=3):
        conv = nn.Conv1d(1,1,kernel_size,padding=padding,bias=False,padding_mode="replicate").to(tensor_data.device)
        conv.weight.data.fill_(1/kernel_size)
        roll_pts_list = []
        for i in range(tensor_data.shape[0]):
            roll_pt_list = []
            for j in range(tensor_data.shape[2]):
                pt = tensor_data[i,:,j].view(1,1,-1)
                roll_pt = conv(pt)
                roll_pt_list.append(roll_pt[0])
            roll_pts = torch.stack(roll_pt_list,dim=-1)
            roll_pts_list.append(roll_pts[0])
        roll_tensor_data = torch.stack(roll_pts_list)
        return roll_tensor_data

    def process_epoch(self, data, step, training=True):   # step is variable
        # with set_detect_anomaly(False):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((xi, xv, xp), (yi, yv, yp)) in enumerate(data):   # x_joint, y_joint            
            if not xi.is_cuda:
                xi = xi.to(self.device)
                xv = xv.to(self.device)
                xp = xp.to(self.device)
                yi = yi.to(self.device)
                yv = yv.to(self.device)
                yp = yp.to(self.device)
            
            states = None
            yi_hat_list, yv_hat_list, yp_hat_list = [], [], []
            dec_pts_list, enc_pts_list = [], []
            # ksrnn_hid_list, vsrnn_hid_list, psrnn_hid_list, urnn_hid_list = [], [], [], []
            # ksrnn_cell_list, vsrnn_cell_list, psrnn_cell_list, urnn_cell_list = [], [], [], []
            
            # yf_list = []
            T = xi.shape[1]
                            
            with torch.cuda.amp.autocast(enabled=False):
                start_time = time.time()
                for t in range(T - 1):
                    # print(t)
                    # self.print_allocated_memory()
                    
                    # hrnn_s_time = time.time()
                    yi_hat, yv_hat, yp_hat, enc_pts, dec_pts, states = self.model(
                        xi[:, t], xv[:, t], xp[:, t], states # step, 
                    )   # step is variable
                    # hrnn_e_time = time.time()
                    # print(hrnn_e_time - hrnn_s_time)
                    
                    yi_hat_list.append(yi_hat.to("cpu"))
                    yv_hat_list.append(yv_hat.to("cpu"))
                    yp_hat_list.append(yp_hat.to("cpu"))
                    enc_pts_list.append(enc_pts.to("cpu"))
                    dec_pts_list.append(dec_pts.to("cpu"))
                    # yf_list.append(yf.to("cpu"))
                    
                    # ipdb.set_trace()
                    # ksrnn_hid_list.append(states[0][0].to("cpu"))
                    # vsrnn_hid_list.append(states[1][0].to("cpu"))
                    # psrnn_hid_list.append(states[2][0].to("cpu"))
                    # urnn_hid_list.append(states[3][0].to("cpu"))
                    
                    # ksrnn_cell_list.append(states[0][1].to("cpu"))
                    # vsrnn_cell_list.append(states[1][1].to("cpu"))
                    # psrnn_cell_list.append(states[2][1].to("cpu"))
                    # urnn_cell_list.append(states[3][1].to("cpu"))
                    
                    # self.print_allocated_memory()
                    del yi_hat, yv_hat, yp_hat, enc_pts, dec_pts    #, yf
                    torch.cuda.empty_cache()

                end_time = time.time()
                
                yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
                yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
                yp_hat_data = torch.permute(torch.stack(yp_hat_list), (1, 0, 2))
                enc_pts_data = torch.permute(torch.stack(enc_pts_list[1:]), (1,0,2))
                dec_pts_data = torch.permute(torch.stack(dec_pts_list[:-1]), (1,0,2))
                # yf_data = torch.permute(torch.stack(yf_list), (1, 0, 2, 3, 4))

                _enc_pts_data = enc_pts_data.reshape(enc_pts_data.shape[0],enc_pts_data.shape[1],-1,2)
                _dec_pts_data = dec_pts_data.reshape(dec_pts_data.shape[0],dec_pts_data.shape[1],-1,2)
                
                # _ksrnn_hid = torch.permute(torch.stack(ksrnn_hid_list), (1,0,2))
                # _vsrnn_hid = torch.permute(torch.stack(vsrnn_hid_list), (1,0,2))
                # _psrnn_hid = torch.permute(torch.stack(psrnn_hid_list), (1,0,2))
                # _urnn_hid = torch.permute(torch.stack(urnn_hid_list), (1,0,2))
                # _ksrnn_cell = torch.permute(torch.stack(ksrnn_cell_list), (1,0,2))
                # _vsrnn_cell = torch.permute(torch.stack(vsrnn_cell_list), (1,0,2))
                # _psrnn_cell = torch.permute(torch.stack(psrnn_cell_list), (1,0,2))
                # _urnn_cell = torch.permute(torch.stack(urnn_cell_list), (1,0,2))
                
                # _urnn_hid = _urnn_hid[:,10:]
                # _urnn_hid = _urnn_hid.to(self.device)
                # _urnn_hid = _urnn_hid.transpose(1,2)
                # _urnn_hid = self.layer_norm(_urnn_hid)
                # _urnn_hid = _urnn_hid.transpose(1,2)
                # Q = K = V = _urnn_hid
                # attn_x, attn_weight = self.attn(Q,K,V)
                # attn_frame_map = self.maxpool(attn_x)
                
                img_loss = self.criterion(yi_hat_data, yi[:, 1:].to("cpu")) * self.loss_w_dic["i"]
                vec_loss = self.criterion(yv_hat_data, yv[:, 1:].to("cpu")) * self.loss_w_dic["v"]
                press_loss = self.criterion(yp_hat_data, yp[:, 1:].to("cpu")) * self.loss_w_dic["p"]
                key_loss = self.criterion(enc_pts_data, dec_pts_data) * self.loss_w_dic["k"]
                
                modality_loss = img_loss + vec_loss + press_loss + key_loss    # + roll_pt_loss + delta_pt_loss
                print("modality", 
                    img_loss.item(), 
                    vec_loss.item(), 
                    press_loss.item(), 
                    key_loss.item(),
                    # roll_pt_loss.item(), 
                    # delta_pt_loss.item()
                    )
                
                # attention coordinate var
                _enc_pts_data = enc_pts_data.reshape((enc_pts_data.shape[0],
                                                        enc_pts_data.shape[1],
                                                        -1,2))
                _dec_pts_data = dec_pts_data.reshape((dec_pts_data.shape[0],
                                                        dec_pts_data.shape[1],
                                                        -1,2))
                enc_var_loss = 1/torch.var(_enc_pts_data, dim=2).mean() * 0.0001
                dec_var_loss = 1/torch.var(_dec_pts_data, dim=2).mean() * 0.0001
                # enc_var_loss = torch.var(_enc_pts_data, dim=2).mean()
                # dec_var_loss = torch.var(_dec_pts_data, dim=2).mean()
                key_var_loss = enc_var_loss + dec_var_loss     
                print("point_key", key_var_loss.item())           
                
                # ipdb.set_trace()
                # ksrnn_state_var = torch.var(_ksrnn_hid, dim=1).mean()
                # vsrnn_state_var = torch.var(_vsrnn_hid, dim=1).mean()
                # psrnn_state_var = torch.var(_psrnn_hid, dim=1).mean()
                # urnn_state_var = torch.var(_urnn_hid, dim=1).mean()

                loss = modality_loss    # + key_var_loss
                total_loss += tensor2numpy(loss)
                
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                del loss
                self.optimizer.step()
                
        return total_loss / (n_batch + 1)
        
    def print_allocated_memory(self):
        print("{:.2f} GB".format(torch.cuda.memory_allocated(self.device) / 1024 ** 3))

