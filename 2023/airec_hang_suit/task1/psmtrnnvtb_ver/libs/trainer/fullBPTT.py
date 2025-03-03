#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

from typing import Dict
import torch
import torch.nn as nn
from eipl.utils import LossScheduler


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, 
                 loss_weights=[0.1, 1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
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

        # total_loss = 0.0
        total_loss, total_left_loss, total_right_loss = 0.0, 0.0, 0.0
        total_left_img_loss, total_left_joint_loss, total_left_state_loss = 0.0, 0.0, 0.0
        total_right_img_loss, total_right_joint_loss, total_right_state_loss = 0.0, 0.0, 0.0
        for n_batch, ((x_left_img, x_right_img, x_joint, x_state), 
                      (y_left_img, y_right_img, y_joint, y_state)) in enumerate(data):
            lxi_data = x_left_img.to(self.device)
            lyi_data = y_left_img.to(self.device)
            rxi_data = x_right_img.to(self.device)
            ryi_data = y_right_img.to(self.device)
            
            x_joint = x_joint.to(self.device)
            lxv_data = x_joint[:,:,:7]
            rxv_data = x_joint[:,:,7:]
            y_joint = y_joint.to(self.device)
            lyv_data = x_joint[:,:,:7]
            ryv_data = x_joint[:,:,7:]
            
            x_state = x_state.to(self.device)
            lxs_data = x_state[:,:,:2]
            rxs_data = x_state[:,:,2:]
            y_state = y_state.to(self.device)
            lys_data = y_state[:,:,:2]
            rys_data = y_state[:,:,2:]
                        
            state = None
            lyi_hat_list, lyv_hat_list, lys_hat_list = [], [], []   # , lydv_hat_list, []
            ryi_hat_list, ryv_hat_list, rys_hat_list = [], [], []   # , rydv_hat_list, []
            left_fast_tau_list, right_fast_tau_list = [], []
            
            T = xi_data.shape[1]
            for t in range(T - 1):
                xi = {"left": lxi_data[:, t], "right": rxi_data[:, t]}
                xv = {"left": lxv_data[:, t], "right": rxv_data[:, t]}
                xs = {"left": lxs_data[:, t], "right": rxs_data[:, t]}
                
                yi_hat, yv_hat, ys_hat, state, fast_tau = self.model( # , ydv_hat
                    xi, xv, xs, state   # , xdv
                )
                lyi_hat_list.append(yi_hat["left"])
                lyv_hat_list.append(yv_hat["left"])
                lys_hat_list.append(ys_hat["left"])
                
                ryi_hat_list.append(yi_hat["right"])
                ryv_hat_list.append(yv_hat["right"])
                rys_hat_list.append(ys_hat["right"])
                
                left_fast_tau_list.append(fast_tau["left"])
                right_fast_tau_list.append(fast_tau["right"])

            lyi_hat_data = torch.permute(torch.stack(lyi_hat_list), (1, 0, 2, 3, 4))
            lyv_hat_data = torch.permute(torch.stack(lyv_hat_list), (1, 0, 2))
            lys_hat_data = torch.permute(torch.stack(lys_hat_list), (1, 0, 2))
            
            ryi_hat_data = torch.permute(torch.stack(ryi_hat_list), (1, 0, 2, 3, 4))
            ryv_hat_data = torch.permute(torch.stack(ryv_hat_list), (1, 0, 2))
            rys_hat_data = torch.permute(torch.stack(rys_hat_list), (1, 0, 2))
            
            left_fast_tau_data = torch.permute(torch.stack(left_fast_tau_list), (1,0,2))[:,:,0]
            right_fast_tau_data = torch.permute(torch.stack(right_fast_tau_list), (1,0,2))[:,:,0]
            
            left_img_loss = nn.MSELoss()(lyi_hat_data, lyi_data[:, 1:]) * self.loss_weights[0]
            left_joint_loss = nn.MSELoss()(lyv_hat_data, lyv_data[:, 1:]) * self.loss_weights[1]
            left_state_loss = nn.MSELoss()(lys_hat_data, lys_data[:, 1:]) * self.loss_weights[2]
            
            right_img_loss = nn.MSELoss()(ryi_hat_data, ryi_data[:, 1:]) * self.loss_weights[0]
            right_joint_loss = nn.MSELoss()(ryv_hat_data, ryv_data[:, 1:]) * self.loss_weights[1]
            right_state_loss = nn.MSELoss()(rys_hat_data, rys_data[:, 1:]) * self.loss_weights[2]
            
            # Gradually change the loss value using the LossScheluder class.
            # pt_loss = nn.MSELoss()(
            #     torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            # ) * self.scheduler(self.loss_weights[2])
            left_loss = left_img_loss + left_joint_loss + left_state_loss   #  + left_delta_joint_loss
            right_loss = right_img_loss + right_joint_loss + right_state_loss   #  + right_delta_joint_loss
            
            loss = left_loss + right_loss
            total_loss += loss.item()
            total_left_loss += left_loss.item()
            total_right_loss += right_loss.item()
            total_left_img_loss += left_img_loss.item()
            total_left_joint_loss += left_joint_loss.item()
            total_left_state_loss += left_state_loss.item()
            total_right_img_loss += right_img_loss.item()
            total_right_joint_loss += right_joint_loss.item()
            total_right_state_loss += right_state_loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        total_loss_dir = {"total_loss": total_loss/(n_batch + 1),
                          "total_left_loss": total_left_loss/(n_batch + 1),
                          "total_right_loss": total_right_loss/(n_batch + 1),
                          "total_left_img_loss": total_left_img_loss/(n_batch + 1),
                          "total_left_joint_loss": total_left_joint_loss/(n_batch + 1),
                          "total_left_state_loss": total_left_state_loss/(n_batch + 1),
                          "total_right_img_loss": total_right_img_loss/(n_batch + 1),
                          "total_right_joint_loss": total_right_joint_loss/(n_batch + 1),
                          "total_right_state_loss": total_right_state_loss/(n_batch + 1),}
        return total_loss_dir

