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
import math


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
    
    def order_of_magnitude(self, number):
        if number == 0:
            return 0
        return int(math.floor(math.log10(abs(number))))

    def process_epoch(self, data, step, training=True): # with step
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_img, x_vec, x_press), (y_img, y_vec, y_press)) in enumerate(data):   # x_joint, y_joint
            states = None
            yi_hat_list, yv_hat_list, yp_hat_list = [], [], []
            dec_pts_list, enc_pts_list = [], []
            ksrnn_state_list, vsrnn_state_list, psrnn_state_list, urnn_state_list = [], [], [], []  # fsrnn_state_list, , []
            T = x_img.shape[1]
            for t in range(T - 1):
                yi_hat, yv_hat, yp_hat, enc_pts, dec_pts, states = self.model(
                    x_img[:, t], x_vec[:, t], x_press[:, t], step, states   # with step
                )
                yi_hat_list.append(yi_hat)
                yv_hat_list.append(yv_hat)
                yp_hat_list.append(yp_hat)
                enc_pts_list.append(enc_pts)
                dec_pts_list.append(dec_pts)
                
                # fsrnn_state_list.append(states[0])
                ksrnn_state_list.append(states[0])
                vsrnn_state_list.append(states[1])
                psrnn_state_list.append(states[2])
                urnn_state_list.append(states[3])

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
            # fsrnn_states = torch.stack([torch.stack(t) for t in fsrnn_state_list])
            ksrnn_states = torch.stack([torch.stack(t) for t in ksrnn_state_list])
            vsrnn_states = torch.stack([torch.stack(t) for t in vsrnn_state_list])
            psrnn_states = torch.stack([torch.stack(t) for t in psrnn_state_list])
            urnn_states = torch.stack([torch.stack(t) for t in urnn_state_list])
            # _fsrnn_state = fsrnn_states[:,0]
            _ksrnn_state = ksrnn_states[:,0]
            _vsrnn_state = vsrnn_states[:,0]
            _psrnn_state = psrnn_states[:,0]
            _urnn_state = urnn_states[:,0]
            # _fsrnn_cell = fsrnn_states[:,1]
            _ksrnn_cell = ksrnn_states[:,1]
            _vsrnn_cell = vsrnn_states[:,1]
            _psrnn_cell = psrnn_states[:,1]
            _urnn_cell = urnn_states[:,1]
            
            # Gradually change the loss value using the LossScheluder class.
            modality_loss_w = 1
            criterion = nn.MSELoss()
            img_loss = criterion(yi_hat_data, y_img[:, 1:]) * self.loss_w_dic["i"]
            pt_loss = criterion(enc_pts_data, dec_pts_data)  * self.loss_w_dic["k"]
            roll_pt_loss = criterion(roll_enc_pts_data, roll_dec_pts_data) * self.loss_w_dic["k"]
            vec_loss = criterion(yv_hat_data, y_vec[:, 1:]) * self.loss_w_dic["v"]
            press_loss = criterion(yp_hat_data, y_press[:, 1:]) * self.loss_w_dic["p"]
            delta_enc_pt_loss = criterion(enc_pts_data, roll_enc_pts_data) * self.loss_w_dic["k"]
            delta_dec_pt_loss = criterion(dec_pts_data, roll_dec_pts_data) * self.loss_w_dic["k"]
            delta_pt_loss = delta_enc_pt_loss + delta_dec_pt_loss
            # if step > 20: 
            #     modality_loss = img_loss + pt_loss + vec_loss + press_loss    # roll_pt_loss +  + delta_pt_loss
            # else:
            #     modality_loss = vec_loss + press_loss
            modality_loss = img_loss + pt_loss + vec_loss + press_loss
            # print("modality",
            #       img_loss.item(), 
            #       roll_pt_loss.item(),
            #       vec_loss.item(), 
            #       press_loss.item(),
            #       delta_pt_loss.item())
            # modality_order = self.order_of_magnitude(modality_loss.item())
            # if modality_order > 0:
            #     modality_loss_w = 10 ** (0 - modality_order)
            #     modality_loss *= modality_loss_w
            #     modality_order = 0
            
            # # attention coordinate var
            # enc_var_loss = 1/torch.var(_enc_pts_data) * 0.00001
            # dec_var_loss = 1/torch.var(_dec_pts_data) * 0.00001
            # var_loss = enc_var_loss + dec_var_loss
            
            e = 1e-6
            # state var
            state_var_loss_w = 1e-6
            # fsrnn_state_var_loss = (1/(_fsrnn_state.var(dim=1)+e)).mean() 
            ksrnn_state_var_loss = (1/(_ksrnn_state.var(dim=1)+e)).mean() 
            vsrnn_state_var_loss = (1/(_vsrnn_state.var(dim=1)+e)).mean() 
            psrnn_state_var_loss = (1/(_psrnn_state.var(dim=1)+e)).mean() 
            urnn_state_var_loss = (1/(_urnn_state.var(dim=1)+e)).mean() 
            state_var_loss = vsrnn_state_var_loss  + urnn_state_var_loss    # + psrnn_state_var_loss + ksrnn_state_var_loss   #  + ksrnn_state_var_loss fsrnn_state_var_loss +  + psrnn_state_var_loss
            # print("state", 
            #       fsrnn_state_var_loss.item(), 
            #       vsrnn_state_var_loss.item(), 
            #       psrnn_state_var_loss.item(), 
            #       urnn_state_var_loss.item())    
            # state_var_order = self.order_of_magnitude(state_var_loss.item())   
            # state_var_loss_w = 10 ** (modality_order - state_var_order)
            state_var_loss *= state_var_loss_w
            
                        
            # attention coordinate var
            pt_var_loss_w = 1e-5    # 1e-3
            enc_var_loss = (1/(_enc_pts_data.var(dim=2)+e)).mean()
            dec_var_loss = (1/(_dec_pts_data.var(dim=2)+e)).mean()
            pt_var_loss = enc_var_loss + dec_var_loss   
            # print("point_key", 
            #       pt_var_loss.item())
            # pt_var_order = self.order_of_magnitude(pt_var_loss.item())   
            # pt_var_loss_w = 10 ** (modality_order - pt_var_order - 2)   # if step > 20 else 0
            pt_var_loss *= pt_var_loss_w
            
            
            seq_pt_var_loss_w = 1e-9    # 1e-7
            seq_enc_var_loss = (1/(_enc_pts_data.var(dim=1)+e)).mean()
            seq_dec_var_loss = (1/(_dec_pts_data.var(dim=1)+e)).mean()
            seq_pt_var_loss = seq_enc_var_loss + seq_dec_var_loss     
            # print("seq_point_key", 
            #       seq_pt_var_loss.item())
            # seq_pt_var_order = self.order_of_magnitude(seq_pt_var_loss.item())   
            # seq_pt_var_loss_w = 10 ** (modality_order - seq_pt_var_order - 2)   # if step > 20 else 0
            seq_pt_var_loss *= seq_pt_var_loss_w
            
            
            # seq state var
            seq_state_var_loss_w = 1e+3     # 1e+6
            # fsrnn_seq_state_var_loss = _fsrnn_state.var(dim=1).var(dim=0).mean() 
            ksrnn_seq_state_var_loss = _ksrnn_state.var(dim=1).var(dim=0).mean() 
            vsrnn_seq_state_var_loss = _vsrnn_state.var(dim=1).var(dim=0).mean() 
            psrnn_seq_state_var_loss = _psrnn_state.var(dim=1).var(dim=0).mean() 
            urnn_seq_state_var_loss = _urnn_state.var(dim=1).var(dim=0).mean() 
            seq_state_var_loss = vsrnn_seq_state_var_loss  + urnn_seq_state_var_loss    # + psrnn_seq_state_var_loss + ksrnn_seq_state_var_loss  # ksrnn_seq_state_var_loss + fsrnn_seq_state_var_loss +  + psrnn_seq_state_var_loss
            # print("seq_state", 
            #       fsrnn_seq_state_var_loss.item(),
            #       vsrnn_seq_state_var_loss.item(), 
            #       psrnn_seq_state_var_loss.item(), 
            #       urnn_seq_state_var_loss.item())
            # seq_state_var_order = self.order_of_magnitude(seq_state_var_loss.item())   
            # seq_state_var_loss_w = 10 ** (modality_order - seq_state_var_order - 2)
            seq_state_var_loss *= seq_state_var_loss_w
            
            
            # cell var
            cell_var_loss_w = 1e-7
            # fsrnn_cell_var_loss = (1/(_fsrnn_cell.var(dim=1)+e)).mean() * cell_var_loss_w
            ksrnn_cell_var_loss = (1/(_ksrnn_cell.var(dim=1)+e)).mean() * cell_var_loss_w
            vsrnn_cell_var_loss = (1/(_vsrnn_cell.var(dim=1)+e)).mean() * cell_var_loss_w
            psrnn_cell_var_loss = (1/(_psrnn_cell.var(dim=1)+e)).mean() * cell_var_loss_w
            urnn_cell_var_loss = (1/(_urnn_cell.var(dim=1)+e)).mean() * cell_var_loss_w
            cell_var_loss = vsrnn_cell_var_loss + urnn_cell_var_loss      # + ksrnn_cell_var_loss    fsrnn_cell_var_loss +  + psrnn_cell_var_loss
            # print("cell", 
            #       fsrnn_cell_var_loss.item(), 
            #       vsrnn_cell_var_loss.item(), 
            #       psrnn_cell_var_loss.item(), 
            #       urnn_cell_var_loss.item())
            # cell_var_order = self.order_of_magnitude(cell_var_loss.item())   
            # cell_var_loss_w = 10 ** (modality_order - cell_var_order - 1)
            cell_var_loss *= cell_var_loss_w
            
            # print(state_var_loss_w, pt_var_loss_w, seq_pt_var_loss_w, seq_state_var_loss_w)
            # print(modality_loss, state_var_loss, seq_state_var_loss, pt_var_loss, seq_pt_var_loss)
                     
            loss = modality_loss + state_var_loss + seq_state_var_loss + pt_var_loss + seq_pt_var_loss #  + cell_var_loss   + pt_var_loss + seq_pt_var_loss
            total_loss += tensor2numpy(loss)
            
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
