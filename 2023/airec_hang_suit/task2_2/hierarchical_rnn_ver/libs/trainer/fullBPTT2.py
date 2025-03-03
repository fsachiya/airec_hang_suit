#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler, tensor2numpy
from typing import Dict, List, Tuple


class fullBPTTtrainer2:
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
                 loss_w_dic:Dict[str, float]={"i": 0.1, "k": 0.1, "v": 1.0, "p": 1.0}, 
                 device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_w_dic = loss_w_dic
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)

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

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_img, x_vec, x_press), (y_img, y_vec, y_press)) in enumerate(data):   # x_joint, y_joint
            states = None
            yi_hat_list, yv_hat_list, yp_hat_list = [], [], []
            dec_pts_list, enc_pts_list = [], []
            fsrnn_state_list, ksrnn_state_list, vsrnn_state_list, psrnn_state_list, urnn_state_list = [], [], [], [], []
            T = x_img.shape[1]
            for t in range(T - 1):
                yi_hat, yv_hat, yp_hat, enc_pts, dec_pts, states = self.model(
                    x_img[:, t], x_vec[:, t], x_press[:, t], states
                )
                yi_hat_list.append(yi_hat)
                yv_hat_list.append(yv_hat)
                yp_hat_list.append(yp_hat)
                enc_pts_list.append(enc_pts)
                dec_pts_list.append(dec_pts)
                
                fsrnn_state_list.append(states[0])
                ksrnn_state_list.append(states[1])
                vsrnn_state_list.append(states[2])
                psrnn_state_list.append(states[3])
                urnn_state_list.append(states[4])

            yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
            yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
            yp_hat_data = torch.permute(torch.stack(yp_hat_list), (1, 0, 2))
            enc_pts_data = torch.permute(torch.stack(enc_pts_list[1:]), (1,0,2))
            dec_pts_data = torch.permute(torch.stack(dec_pts_list[:-1]), (1,0,2))
            _enc_pts_data = enc_pts_data.reshape(enc_pts_data.shape[0],enc_pts_data.shape[1],-1,2)
            _dec_pts_data = dec_pts_data.reshape(dec_pts_data.shape[0],dec_pts_data.shape[1],-1,2)
            
            # list(tuple(tensor(tensor())))    # 27*2*5*50
            fsrnn_states = torch.stack([torch.stack(t) for t in fsrnn_state_list])
            ksrnn_states = torch.stack([torch.stack(t) for t in ksrnn_state_list])
            vsrnn_states = torch.stack([torch.stack(t) for t in vsrnn_state_list])
            psrnn_states = torch.stack([torch.stack(t) for t in psrnn_state_list])
            urnn_states = torch.stack([torch.stack(t) for t in urnn_state_list])
            _fsrnn_state = fsrnn_states[:,0]
            _ksrnn_state = ksrnn_states[:,0]
            _vsrnn_state = vsrnn_states[:,0]
            _psrnn_state = psrnn_states[:,0]
            _urnn_state = urnn_states[:,0]
            _fsrnn_cell = fsrnn_states[:,1]
            _ksrnn_cell = ksrnn_states[:,1]
            _vsrnn_cell = vsrnn_states[:,1]
            _psrnn_cell = psrnn_states[:,1]
            _urnn_cell = urnn_states[:,1]
            
            # Gradually change the loss value using the LossScheluder class.
            criterion = nn.MSELoss()
            img_loss = criterion(yi_hat_data, y_img[:, 1:]) * self.loss_w_dic["i"]
            pt_loss = criterion(enc_pts_data, dec_pts_data)  * self.loss_w_dic["k"]
            vec_loss = criterion(yv_hat_data, y_vec[:, 1:]) * self.loss_w_dic["v"]
            press_loss = criterion(yp_hat_data, y_press[:, 1:]) * self.loss_w_dic["p"]
            
            # # attention coordinate var
            # enc_var_loss = 1/torch.var(_enc_pts_data) * 0.00001
            # dec_var_loss = 1/torch.var(_dec_pts_data) * 0.00001
            # var_loss = enc_var_loss + dec_var_loss
            
            # state var
            state_var_loss_w = 0.00001
            fsrnn_state_var_loss = 1/torch.var(_fsrnn_state, dim=1).mean() * state_var_loss_w
            ksrnn_state_var_loss = 1/torch.var(_ksrnn_state, dim=1).mean() * state_var_loss_w
            vsrnn_state_var_loss = 1/torch.var(_vsrnn_state, dim=1).mean() * state_var_loss_w
            psrnn_state_var_loss = 1/torch.var(_psrnn_state, dim=1).mean() * state_var_loss_w
            urnn_state_var_loss = 1/torch.var(_urnn_state, dim=1).mean() * state_var_loss_w
            state_var_loss = fsrnn_state_var_loss + ksrnn_state_var_loss + vsrnn_state_var_loss + psrnn_state_var_loss + urnn_state_var_loss

            # cell var
            cell_var_loss_w = 0.0001
            fsrnn_cell_var_loss = 1/torch.var(_fsrnn_cell, dim=1).mean() * cell_var_loss_w
            ksrnn_cell_var_loss = 1/torch.var(_ksrnn_cell, dim=1).mean() * cell_var_loss_w
            vsrnn_cell_var_loss = 1/torch.var(_vsrnn_cell, dim=1).mean() * cell_var_loss_w
            psrnn_cell_var_loss = 1/torch.var(_psrnn_cell, dim=1).mean() * cell_var_loss_w
            urnn_cell_var_loss = 1/torch.var(_urnn_cell, dim=1).mean() * cell_var_loss_w
            cell_var_loss = fsrnn_cell_var_loss + ksrnn_cell_var_loss + vsrnn_cell_var_loss + psrnn_cell_var_loss + urnn_cell_var_loss            
            
            loss = img_loss + vec_loss + press_loss + pt_loss + state_var_loss + cell_var_loss # + var_loss
            total_loss += tensor2numpy(loss)
            
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
