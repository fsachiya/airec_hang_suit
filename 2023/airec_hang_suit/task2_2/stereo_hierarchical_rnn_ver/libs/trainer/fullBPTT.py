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
                 approx_parallax_ratio=0.0,
                 device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_w_dic = loss_w_dic
        self.approx_parallax_ratio = approx_parallax_ratio
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

    def process_epoch(self, data, epoch, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_left_img, x_right_img, x_vec, x_press), 
                      (y_left_img, y_right_img, y_vec, y_press)) in enumerate(data):   # x_joint, y_joint
            # x_img = x_img.to(self.device)
            # y_img = y_img.to(self.device)
            # x_joint = x_joint.to(self.device)
            # y_joint = y_joint.to(self.device)

            states = None
            yli_hat_list, yri_hat_list, yv_hat_list, yp_hat_list = [], [], [], []
            left_dec_pts_list, left_enc_pts_list, right_dec_pts_list, right_enc_pts_list = [], [], [], []
            left_ksrnn_state_list, right_ksrnn_state_list, vsrnn_state_list, psrnn_state_list, urnn_state_list = [], [], [], [], []
            T = x_left_img.shape[1]
            for t in range(T - 1):
                [yli_hat, yri_hat, yv_hat, yp_hat, 
                 left_enc_pts, left_dec_pts, right_enc_pts, right_dec_pts, states] = self.model(
                    x_left_img[:, t], x_right_img[:, t], x_vec[:, t], x_press[:, t], states
                )
                yli_hat_list.append(yli_hat)
                yri_hat_list.append(yri_hat)
                yv_hat_list.append(yv_hat)
                yp_hat_list.append(yp_hat)
                left_enc_pts_list.append(left_enc_pts)
                left_dec_pts_list.append(left_dec_pts)
                right_enc_pts_list.append(right_enc_pts)
                right_dec_pts_list.append(right_dec_pts)
                
                left_ksrnn_state_list.append(states[0])
                right_ksrnn_state_list.append(states[1])
                vsrnn_state_list.append(states[2])
                psrnn_state_list.append(states[3])
                urnn_state_list.append(states[4])
                
            yli_hat_data = torch.permute(torch.stack(yli_hat_list), (1, 0, 2, 3, 4))
            yri_hat_data = torch.permute(torch.stack(yri_hat_list), (1, 0, 2, 3, 4))
            yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
            yp_hat_data = torch.permute(torch.stack(yp_hat_list), (1, 0, 2))
            left_enc_pts_data = torch.permute(torch.stack(left_enc_pts_list[1:]), (1, 0, 2))    # [:-1]
            left_dec_pts_data = torch.permute(torch.stack(left_dec_pts_list[:-1]), (1, 0, 2))    # [1:]
            right_enc_pts_data = torch.permute(torch.stack(right_enc_pts_list[1:]), (1, 0, 2))  # [:-1]
            right_dec_pts_data = torch.permute(torch.stack(right_dec_pts_list[:-1]), (1, 0, 2))  # [1:]
            
            _left_enc_pts_data = left_enc_pts_data.reshape(left_enc_pts_data.shape[0],left_enc_pts_data.shape[1],-1,2)
            _right_enc_pts_data = right_enc_pts_data.reshape(right_enc_pts_data.shape[0],right_enc_pts_data.shape[1],-1,2)
            _left_dec_pts_data = left_dec_pts_data.reshape(left_dec_pts_data.shape[0],left_dec_pts_data.shape[1],-1,2)
            _right_dec_pts_data = right_dec_pts_data.reshape(right_dec_pts_data.shape[0],right_dec_pts_data.shape[1],-1,2)
            
            # list(tuple(tensor(tensor())))    # 27*2*5*50
            left_ksrnn_states = torch.stack([torch.stack(t) for t in left_ksrnn_state_list])
            right_ksrnn_states = torch.stack([torch.stack(t) for t in right_ksrnn_state_list])
            vsrnn_states = torch.stack([torch.stack(t) for t in vsrnn_state_list])
            psrnn_states = torch.stack([torch.stack(t) for t in psrnn_state_list])
            urnn_states = torch.stack([torch.stack(t) for t in urnn_state_list])
            
            _left_ksrnn_state = left_ksrnn_states[:,0]
            _right_ksrnn_state = right_ksrnn_states[:,0]
            _vsrnn_state = vsrnn_states[:,0]
            _psrnn_state = psrnn_states[:,0]
            _urnn_state = urnn_states[:,0]
            
            # [[x,y], [x,y], [x,y],...]
            mse = nn.MSELoss()
            left_img_loss = mse(yli_hat_data, y_left_img[:, 1:]) * self.loss_w_dic["i"]
            right_img_loss = mse(yri_hat_data, y_right_img[:, 1:]) * self.loss_w_dic["i"]
            vec_loss = mse(yv_hat_data, y_vec[:, 1:]) * self.loss_w_dic["v"]
            press_loss = mse(yp_hat_data, y_press[:, 1:]) * self.loss_w_dic["p"]
            
            # Gradually change the loss value using the LossScheluder class.
            left_pt_loss = mse(left_enc_pts_data, left_dec_pts_data) * self.loss_w_dic["k"] # * self.scheduler(self.loss_weights[2])
            right_pt_loss = mse(right_enc_pts_data, right_dec_pts_data) * self.loss_w_dic["k"] # * self.scheduler(self.loss_weights[2])
            pt_loss = left_pt_loss + right_pt_loss
            # pt_loss = left_pt_loss * right_pt_loss
            
            
            ## not omit side attention constrain
            # stereo_enc_pts_x_loss = mse(_left_enc_pts_data[:,:,:,0], _right_enc_pts_data[:,:,:,0]) * (self.loss_w_dic["k"] * 0.1)
            # stereo_dec_pts_x_loss = mse(_left_dec_pts_data[:,:,:,0], _right_dec_pts_data[:,:,:,0]) * (self.loss_w_dic["k"] * 0.1)
            # stereo_enc_pts_y_loss = mse(_left_enc_pts_data[:,:,:,1], _right_enc_pts_data[:,:,:,1]) * (self.loss_w_dic["k"])
            # stereo_dec_pts_y_loss = mse(_left_dec_pts_data[:,:,:,1], _right_dec_pts_data[:,:,:,1]) * (self.loss_w_dic["k"])
            # stereo_xy_loss = stereo_enc_pts_x_loss + stereo_dec_pts_x_loss + stereo_enc_pts_y_loss + stereo_dec_pts_y_loss
            # # stereo_y_loss = stereo_enc_pts_y_loss + stereo_dec_pts_y_loss
            
            # omit side attention constrain
            stereo_enc_pt_x_loss_list, stereo_dec_pt_x_loss_list, stereo_enc_pt_y_loss_list, stereo_dec_pt_y_loss_list = [], [], [], []
            for i in range(_left_enc_pts_data.shape[2]):
                if (_left_enc_pts_data[:,:,i,0].mean() > self.approx_parallax_ratio) and (_right_enc_pts_data[:,:,i,0].mean() < (1-self.approx_parallax_ratio)):
                    stereo_enc_pt_x_loss = mse(_left_enc_pts_data[:,:,i,0], _right_enc_pts_data[:,:,i,0])
                    stereo_dec_pt_x_loss = mse(_left_dec_pts_data[:,:,i,0], _right_dec_pts_data[:,:,i,0])
                    stereo_enc_pt_y_loss = mse(_left_enc_pts_data[:,:,i,1], _right_enc_pts_data[:,:,i,1])
                    stereo_dec_pt_y_loss = mse(_left_dec_pts_data[:,:,i,1], _right_dec_pts_data[:,:,i,1])
                    stereo_enc_pt_x_loss_list.append(stereo_enc_pt_x_loss)
                    stereo_dec_pt_x_loss_list.append(stereo_dec_pt_x_loss)
                    stereo_enc_pt_y_loss_list.append(stereo_enc_pt_y_loss)
                    stereo_dec_pt_y_loss_list.append(stereo_dec_pt_y_loss)
            stereo_enc_pts_x_loss = torch.stack(stereo_enc_pt_x_loss_list).mean() * (self.loss_w_dic["k"] * 0.1)
            stereo_dec_pts_x_loss = torch.stack(stereo_dec_pt_x_loss_list).mean() * (self.loss_w_dic["k"] * 0.1)
            stereo_enc_pts_y_loss = torch.stack(stereo_enc_pt_y_loss_list).mean() * (self.loss_w_dic["k"])
            stereo_dec_pts_y_loss = torch.stack(stereo_dec_pt_y_loss_list).mean() * (self.loss_w_dic["k"])
            stereo_xy_loss = stereo_enc_pts_x_loss + stereo_dec_pts_x_loss + stereo_enc_pts_y_loss + stereo_dec_pts_y_loss
            # stereo_y_loss = stereo_enc_pts_y_loss + stereo_dec_pts_y_loss
            
            
            # attention coordinate var
            attn_var_loss_w = 0.0001
            left_enc_var_loss = 1/torch.var(_left_enc_pts_data) * attn_var_loss_w
            left_dec_var_loss = 1/torch.var(_left_dec_pts_data) * attn_var_loss_w
            right_enc_var_loss = 1/torch.var(_right_enc_pts_data) * attn_var_loss_w
            right_dec_var_loss = 1/torch.var(_right_dec_pts_data) * attn_var_loss_w
            attn_var_loss = left_enc_var_loss + left_dec_var_loss + right_enc_var_loss + right_dec_var_loss
            

            # state var
            state_var_loss_w = 0.00001
            left_ksrnn_state_var_loss = 1/torch.var(_left_ksrnn_state, dim=1).mean() * state_var_loss_w * 0.01
            right_ksrnn_state_var_loss = 1/torch.var(_right_ksrnn_state, dim=1).mean() * state_var_loss_w * 0.01
            vsrnn_state_var_loss = 1/torch.var(_vsrnn_state, dim=1).mean() * state_var_loss_w * 0.1
            psrnn_state_var_loss = 1/torch.var(_psrnn_state, dim=1).mean() * state_var_loss_w * 0.1
            urnn_state_var_loss = 1/torch.var(_urnn_state, dim=1).mean() * state_var_loss_w
            # if epoch > 10:
            #     state_var_loss = left_ksrnn_state_var_loss + right_ksrnn_state_var_loss + vsrnn_state_var_loss + psrnn_state_var_loss + urnn_state_var_loss
            # else:
            #     state_var_loss = 0
            state_var_loss = left_ksrnn_state_var_loss + right_ksrnn_state_var_loss + vsrnn_state_var_loss + psrnn_state_var_loss + urnn_state_var_loss

            
            loss = left_img_loss + right_img_loss + pt_loss + vec_loss + press_loss + stereo_xy_loss + attn_var_loss + state_var_loss
            # loss = left_img_loss * right_img_loss * pt_loss * vec_loss * press_loss * stereo_y_loss #  + stereo_xy_loss

            total_loss += tensor2numpy(loss)
                        
            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
