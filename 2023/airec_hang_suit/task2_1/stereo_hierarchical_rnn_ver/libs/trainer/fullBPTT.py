#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

# from typing import Dict
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from eipl.utils import LossScheduler
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
                 scaler, 
                 max_norm, 
                 loss_w_dic:Dict[str, float]={"i": 0.1, "k": 0.1, "v": 1.0, "c": 1.0}, 
                 device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.scaler = scaler
        self.max_norm = max_norm
        self.loss_w_dic = loss_w_dic
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)

    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
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
            
        [total_loss, 
         total_left_img_loss, total_right_img_loss, 
         total_left_key_point_loss, total_right_key_point_loss, 
         total_stereo_enc_key_point_loss, total_stereo_dec_key_point_loss,
         total_joint_loss, total_cmd_loss] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for n_batch, ((x_left_img, x_right_img, x_joint, x_cmd), 
                      (y_left_img, y_right_img, y_joint, y_cmd)) in enumerate(data):
            xli_data = x_left_img.to(self.device)
            yli_data = y_left_img.to(self.device)
            xri_data = x_right_img.to(self.device)
            yri_data = y_right_img.to(self.device)
            
            xv_data = x_joint.to(self.device)
            yv_data = y_joint.to(self.device)
            xc_data = x_cmd.to(self.device)
            yc_data = y_cmd.to(self.device)
            
            states = None
            yli_hat_list, yri_hat_list, ylk_hat_list, yrk_hat_list, yv_hat_list, yc_hat_list = [], [], [], [], [], []
            left_enc_pts_list, right_enc_pts_list, left_dec_pts_list, right_dec_pts_list = [], [], [], []
            T = xli_data.shape[1]
            
            with autocast(): 
                for t in range(T - 1):
                    xli = xli_data[:, t]
                    xri = xri_data[:, t]
                    xi_dic = {"left": xli, "right": xri}
                    xv = xv_data[:, t]
                    xc = xc_data[:, t]
                    yi_hat_dic, yk_hat_dic, yv_hat, yc_hat, enc_pts_dic, dec_pts_dic, states = self.model(
                        xi_dic, xv, xc, states
                    )    
                    yli_hat_list.append(yi_hat_dic["left"])
                    yri_hat_list.append(yi_hat_dic["right"])
                    ylk_hat_list.append(yk_hat_dic["left"])
                    yrk_hat_list.append(yk_hat_dic["right"])
                    yv_hat_list.append(yv_hat)
                    yc_hat_list.append(yc_hat)  
                    left_enc_pts_list.append(enc_pts_dic["left"])
                    right_enc_pts_list.append(enc_pts_dic["right"])
                    left_dec_pts_list.append(dec_pts_dic["left"])
                    right_dec_pts_list.append(dec_pts_dic["right"])

                yli_hat_data = torch.permute(torch.stack(yli_hat_list), (1, 0, 2, 3, 4))
                yri_hat_data = torch.permute(torch.stack(yri_hat_list), (1, 0, 2, 3, 4))
                yrk_hat_data = torch.permute(torch.stack(ylk_hat_list), (1, 0, 2))
                ylk_hat_data = torch.permute(torch.stack(yrk_hat_list), (1, 0, 2))
                yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
                yc_hat_data = torch.permute(torch.stack(yc_hat_list), (1, 0, 2))
                left_enc_pts_data = torch.permute(torch.stack(left_enc_pts_list), (1, 0, 2))
                right_enc_pts_data = torch.permute(torch.stack(right_enc_pts_list), (1, 0, 2))
                left_dec_pts_data = torch.permute(torch.stack(left_dec_pts_list), (1, 0, 2))
                right_dec_pts_data = torch.permute(torch.stack(right_dec_pts_list), (1, 0, 2))
                                
                criterion = nn.MSELoss()
                left_img_loss = criterion(yli_hat_data, yli_data[:, 1:]) * self.loss_w_dic["i"]
                right_img_loss = criterion(yri_hat_data, yri_data[:, 1:]) * self.loss_w_dic["i"]
                left_key_point_loss = criterion(left_enc_pts_data, left_dec_pts_data) * self.loss_w_dic["k"]
                right_key_point_loss = criterion(right_enc_pts_data, right_dec_pts_data) * self.loss_w_dic["k"]
                
                _left_enc_pts_data = left_enc_pts_data.reshape(left_enc_pts_data.shape[0],
                                                               left_enc_pts_data.shape[1],
                                                               -1,2)
                _right_enc_pts_data = right_enc_pts_data.reshape(right_enc_pts_data.shape[0],
                                                               right_enc_pts_data.shape[1],
                                                               -1,2)
                _left_dec_pts_data = left_dec_pts_data.reshape(left_dec_pts_data.shape[0],
                                                               left_dec_pts_data.shape[1],
                                                               -1,2)
                _right_dec_pts_data = right_dec_pts_data.reshape(right_dec_pts_data.shape[0],
                                                               right_dec_pts_data.shape[1],
                                                               -1,2)

                stereo_enc_key_point_loss = criterion(_left_enc_pts_data[:,:,:,1], _right_enc_pts_data[:,:,:,1]) * (self.loss_w_dic["k"] * 0.1)
                stereo_dec_key_point_loss = criterion(_left_dec_pts_data[:,:,:,1], _right_dec_pts_data[:,:,:,1]) * (self.loss_w_dic["k"] * 0.1)
                joint_loss = criterion(yv_hat_data, yv_data[:, 1:]) * self.loss_w_dic["v"]
                cmd_loss = criterion(yc_hat_data, yc_data[:, 1:]) * self.loss_w_dic["c"]
                    
                loss = left_img_loss + right_img_loss + left_key_point_loss + right_key_point_loss + joint_loss + cmd_loss + stereo_enc_key_point_loss + stereo_dec_key_point_loss
                total_loss += loss.item()
                total_left_img_loss += left_img_loss.item()
                total_right_img_loss += right_img_loss.item()
                total_left_key_point_loss += left_key_point_loss.item()
                total_right_key_point_loss += right_key_point_loss.item()
                total_stereo_enc_key_point_loss += stereo_enc_key_point_loss.item()
                total_stereo_dec_key_point_loss += stereo_dec_key_point_loss.item()
                total_joint_loss += joint_loss.item()
                total_cmd_loss += cmd_loss.item()

            if training:
                # self.optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                # self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

        total_loss_dic = {"total_loss": total_loss/(n_batch + 1),
                          "total_left_img_loss": total_left_img_loss/(n_batch + 1),
                          "total_right_img_loss": total_right_img_loss/(n_batch + 1),
                          "total_left_key_point_loss": total_left_key_point_loss/(n_batch + 1),
                          "total_right_key_point_loss": total_right_key_point_loss/(n_batch + 1),
                          "total_stereo_enc_key_point_loss": total_stereo_enc_key_point_loss/(n_batch + 1),
                          "total_stereo_dec_key_point_loss": total_stereo_dec_key_point_loss/(n_batch + 1),
                          "total_joint_loss": total_joint_loss/(n_batch + 1),
                          "total_cmd_loss": total_cmd_loss/(n_batch + 1),
                          }
        
        return total_loss_dic
