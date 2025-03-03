#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

# from typing import Dict
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
                 loss_weights=[0.1, 1.0, 1.0, 1.0], device="cpu"):
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

        total_loss, total_img_loss, total_key_point_loss, total_joint_loss, total_cmd_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        for n_batch, ((x_img, x_joint, x_cmd), 
                      (y_img, y_joint, y_cmd)) in enumerate(data):
            xi_data = x_img.to(self.device)
            yi_data = y_img.to(self.device)
            
            xv_data = x_joint.to(self.device)
            yv_data = y_joint.to(self.device)
            xc_data = x_cmd.to(self.device)
            yc_data = y_cmd.to(self.device)
            
            states = None
            yi_hat_list, yk_hat_list, yv_hat_list, yc_hat_list = [], [], [], []
            enc_pts_list, dec_pts_list = [], []
            T = xi_data.shape[1]
            for t in range(T - 1):
                xi = xi_data[:, t]
                xv = xv_data[:, t]
                xc = xc_data[:, t]
                yi_hat, yk_hat, yv_hat, yc_hat, enc_pts, dec_pts, states = self.model(
                    xi, xv, xc, states
                )
                yi_hat_list.append(yi_hat)
                yk_hat_list.append(yk_hat)
                yv_hat_list.append(yv_hat)
                yc_hat_list.append(yc_hat)  
                enc_pts_list.append(enc_pts)
                dec_pts_list.append(dec_pts)

            yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
            yk_hat_data = torch.permute(torch.stack(yk_hat_list), (1, 0, 2))
            yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
            yc_hat_data = torch.permute(torch.stack(yc_hat_list), (1, 0, 2))
            enc_pts_data = torch.permute(torch.stack(enc_pts_list), (1, 0, 2))
            dec_pts_data = torch.permute(torch.stack(dec_pts_list), (1, 0, 2))
            
            img_loss = nn.MSELoss()(yi_hat_data, yi_data[:, 1:]) * self.loss_weights[0]
            key_point_loss = nn.MSELoss()(enc_pts_data, dec_pts_data) * self.loss_weights[1]
            joint_loss = nn.MSELoss()(yv_hat_data, yv_data[:, 1:]) * self.loss_weights[2]
            cmd_loss = nn.MSELoss()(yc_hat_data, yc_data[:, 1:]) * self.loss_weights[3]
            # Gradually change the loss value using the LossScheluder class.
            # pt_loss = nn.MSELoss()(
            #     torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            # ) * self.scheduler(self.loss_weights[2])
            loss = img_loss + key_point_loss + joint_loss + cmd_loss
            total_loss += loss.item()
            total_img_loss += img_loss.item()
            total_key_point_loss += key_point_loss.item()
            total_joint_loss += joint_loss.item()
            total_cmd_loss += cmd_loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        total_loss_dir = {"total_loss": total_loss/(n_batch + 1),
                          "total_img_loss": total_img_loss/(n_batch + 1),
                          "total_key_point_loss": total_key_point_loss/(n_batch + 1),
                          "total_joint_loss": total_joint_loss/(n_batch + 1),
                          "total_cmd_loss": total_cmd_loss/(n_batch + 1),
                          }
        
        return total_loss_dir
