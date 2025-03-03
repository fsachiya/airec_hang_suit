#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import sys
import torch
import torch.nn as nn
from torch.autograd import detect_anomaly
import numpy as np
from typing import Dict, List, Tuple
from eipl.utils import LossScheduler, tensor2numpy


try:
    from libs.utils import moving_average
except:
    sys.path.append("./libs/")
    from utils import moving_average


class fullBPTTtrainer:
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
                 # loss_weights = [0.1, 1]
                 loss_w_dic:Dict[str, float]={"i": 0.1, "k": 0.1, "v": 1.0, "p": 1.0}, 
                 device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_w_dic = loss_w_dic
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)
        
        # self.vec_attn = nn.MultiheadAttention(6, 2).to(self.device)

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
        with detect_anomaly(False):
            if not training:
                self.model.eval()
            else:
                self.model.train()

            total_loss = 0.0
            for n_batch, ((x_img, x_vec, x_press), (y_img, y_vec, y_press)) in enumerate(data):   # x_joint, y_joint
                states = None
                yi_hat_list, yv_hat_list, yp_hat_list = [], [], []
                dec_pts_list, enc_pts_list = [], []
                ksrnn_state_list, vsrnn_state_list, psrnn_state_list, urnn_state_list = [], [], [], []
                xi_feat_list = []
                T = x_img.shape[1]
                
                for t in range(T - 1):
                    yi_hat, yv_hat, yp_hat, enc_pts, dec_pts, states, xi_feat = self.model(
                        x_img[:, t], x_vec[:, t], x_press[:, t], states # step, 
                    )   # step is variable
                    yi_hat_list.append(yi_hat)
                    yv_hat_list.append(yv_hat)
                    yp_hat_list.append(yp_hat)
                    enc_pts_list.append(enc_pts)
                    dec_pts_list.append(dec_pts)
                    
                    ksrnn_state_list.append(states[0])
                    vsrnn_state_list.append(states[1])
                    psrnn_state_list.append(states[2])
                    urnn_state_list.append(states[3])
                    xi_feat_list.append(xi_feat)

                yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
                yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
                yp_hat_data = torch.permute(torch.stack(yp_hat_list), (1, 0, 2))
                enc_pts_data = torch.permute(torch.stack(enc_pts_list[1:]), (1,0,2))
                dec_pts_data = torch.permute(torch.stack(dec_pts_list[:-1]), (1,0,2))
                
                roll_enc_pts_data = self.tensor_moving_average(enc_pts_data)
                roll_dec_pts_data = self.tensor_moving_average(dec_pts_data)
                
                _enc_pts_data = enc_pts_data.reshape(enc_pts_data.shape[0],enc_pts_data.shape[1],-1,2)
                _dec_pts_data = dec_pts_data.reshape(dec_pts_data.shape[0],dec_pts_data.shape[1],-1,2)
                _roll_enc_pts_data = roll_enc_pts_data.reshape(enc_pts_data.shape[0],enc_pts_data.shape[1],-1,2)
                _roll_dec_pts_data = roll_dec_pts_data.reshape(dec_pts_data.shape[0],dec_pts_data.shape[1],-1,2)
                
                # list(tuple(tensor(tensor())))    # 27*2*5*50
                ksrnn_states = torch.stack([torch.stack(t) for t in ksrnn_state_list])
                vsrnn_states = torch.stack([torch.stack(t) for t in vsrnn_state_list])
                psrnn_states = torch.stack([torch.stack(t) for t in psrnn_state_list])
                urnn_states = torch.stack([torch.stack(t) for t in urnn_state_list])
                _ksrnn_state = ksrnn_states[:,0]    # torch.Size([206, 1, 50]), 206, 5, 50
                _vsrnn_state = vsrnn_states[:,0]
                _psrnn_state = psrnn_states[:,0]
                _urnn_state = urnn_states[:,0]
                _xi_feat = torch.stack(xi_feat_list)
                _ksrnn_cell = ksrnn_states[:,1]
                _vsrnn_cell = vsrnn_states[:,1]
                _psrnn_cell = psrnn_states[:,1]
                _urnn_cell = urnn_states[:,1]
                
                criterion = nn.MSELoss()
                img_loss = criterion(yi_hat_data, y_img[:, 1:]) * self.loss_w_dic["i"]
                vec_loss = criterion(yv_hat_data, y_vec[:, 1:]) * self.loss_w_dic["v"]
                press_loss = criterion(yp_hat_data, y_press[:, 1:]) * self.loss_w_dic["p"]
                pt_loss = criterion(enc_pts_data, dec_pts_data) * self.loss_w_dic["k"]
                roll_pt_loss = criterion(roll_enc_pts_data, roll_dec_pts_data) * self.loss_w_dic["k"]
                # loss between roll_pt and pt
                delta_enc_pt_loss = criterion(enc_pts_data, roll_enc_pts_data) * self.loss_w_dic["k"]
                delta_dec_pt_loss = criterion(dec_pts_data, roll_dec_pts_data) * self.loss_w_dic["k"]
                delta_pt_loss = delta_enc_pt_loss + delta_dec_pt_loss
                modality_loss = img_loss + vec_loss + press_loss + roll_pt_loss + delta_pt_loss    # img_loss +  + roll_pt_loss + delta_pt_loss
                print("modality", 
                      img_loss.item(), 
                      vec_loss.item(), 
                      press_loss.item(), 
                      roll_pt_loss.item(), 
                      delta_pt_loss.item())
                
                # attention coordinate var
                enc_var_loss = 1/torch.var(_enc_pts_data, dim=2).mean() * 0.0001
                dec_var_loss = 1/torch.var(_dec_pts_data, dim=2).mean() * 0.0001
                pt_var_loss = enc_var_loss + dec_var_loss     
                print("point_key", 
                      pt_var_loss.item())           
                
                # state var
                state_var_loss_w = 1e-5   # * 0.001                
                ksrnn_state_var_loss = 1/torch.var(_ksrnn_state, dim=1).mean() * state_var_loss_w  #  + _ksrnn_state.var(dim=1).var(dim=0).mean() * state_seq_var_loss_w
                vsrnn_state_var_loss = 1/torch.var(_vsrnn_state, dim=1).mean() * state_var_loss_w
                psrnn_state_var_loss = 1/torch.var(_psrnn_state, dim=1).mean() * state_var_loss_w
                urnn_state_var_loss = 1/torch.var(_urnn_state, dim=1).mean() * state_var_loss_w
                state_var_loss = vsrnn_state_var_loss + psrnn_state_var_loss + urnn_state_var_loss  # ksrnn_state_var_loss + 
                print("state", 
                      vsrnn_state_var_loss.item(), 
                      psrnn_state_var_loss.item(), 
                      urnn_state_var_loss.item())
                #######################################
                # ksrnn_state_var = torch.var(_ksrnn_state, dim=1).mean()
                # vsrnn_state_var = torch.var(_vsrnn_state, dim=1).mean()
                # psrnn_state_var = torch.var(_psrnn_state, dim=1).mean()
                # urnn_state_var = torch.var(_urnn_state, dim=1).mean()
                # state_var_loss = 1/(ksrnn_state_var + vsrnn_state_var + psrnn_state_var + urnn_state_var) * state_var_loss_w
                
                # seq state var
                seq_state_var_loss_w = 1e+6
                ksrnn_seq_state_var_loss = _ksrnn_state.var(dim=1).var(dim=0).mean() * seq_state_var_loss_w  #  + _ksrnn_state.var(dim=1).var(dim=0).mean() * state_seq_var_loss_w
                vsrnn_seq_state_var_loss = _vsrnn_state.var(dim=1).var(dim=0).mean() * seq_state_var_loss_w
                psrnn_seq_state_var_loss = _psrnn_state.var(dim=1).var(dim=0).mean() * seq_state_var_loss_w
                urnn_seq_state_var_loss = _urnn_state.var(dim=1).var(dim=0).mean() * seq_state_var_loss_w
                seq_state_var_loss = vsrnn_seq_state_var_loss + psrnn_seq_state_var_loss + urnn_seq_state_var_loss  # ksrnn_seq_state_var_loss + 
                print("seq_state", 
                      vsrnn_seq_state_var_loss.item(), 
                      psrnn_seq_state_var_loss.item(), 
                      urnn_seq_state_var_loss.item())
                
                # cell var
                cell_var_loss_w = 1e-5
                ksrnn_cell_var_loss = 1/torch.var(_ksrnn_cell, dim=1).mean() * cell_var_loss_w
                vsrnn_cell_var_loss = 1/torch.var(_vsrnn_cell, dim=1).mean() * cell_var_loss_w
                psrnn_cell_var_loss = 1/torch.var(_psrnn_cell, dim=1).mean() * cell_var_loss_w
                urnn_cell_var_loss = 1/torch.var(_urnn_cell, dim=1).mean() * cell_var_loss_w
                cell_var_loss = vsrnn_cell_var_loss + psrnn_cell_var_loss + urnn_cell_var_loss    # ksrnn_cell_var_loss +         
                print("cell", 
                      vsrnn_cell_var_loss.item(), 
                      psrnn_cell_var_loss.item(), 
                      urnn_cell_var_loss.item())
                #######################################
                # ksrnn_cell_var = torch.var(_ksrnn_cell, dim=1).mean()
                # vsrnn_cell_var = torch.var(_vsrnn_cell, dim=1).mean()
                # psrnn_cell_var = torch.var(_psrnn_cell, dim=1).mean()
                # urnn_cell_var = torch.var(_urnn_cell, dim=1).mean()
                # cell_var_loss = 1/(ksrnn_cell_var + vsrnn_cell_var + psrnn_cell_var + urnn_cell_var) * cell_var_loss_w
                
                # loss = img_loss + vec_loss + press_loss + pt_loss + state_var_loss + cell_var_loss# + var_loss
                # loss = img_loss + vec_loss + press_loss + roll_pt_loss + delta_pt_loss + state_var_loss + cell_var_loss + pt_var_loss# + pt_var_loss
                loss = modality_loss + pt_var_loss + state_var_loss + cell_var_loss + seq_state_var_loss  # + cell_var_loss # + pt_var_loss    cell_var_loss  + state_var_loss + cell_var_loss
                
                total_loss += tensor2numpy(loss)
                
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()

            return total_loss / (n_batch + 1)
        
        
