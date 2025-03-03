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

import sys
try:
    from libs.model import CNNMTRNNVTB
except:
    sys.path.append("./libs/")
    from model import CNNMTRNNVTB

class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizers, loss_weights=[1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizers = optimizers
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

        total_loss = 0.0
        for n_batch, ((x_img, x_joint), (y_img, y_joint)) in enumerate(data):
            x_img = x_img.to(self.device)
            y_img = y_img.to(self.device)
            x_joint = x_joint.to(self.device)
            y_joint = y_joint.to(self.device)
            
            # import ipdb; ipdb.set_trace()

            state = None
            yi_list, yv_list = [], []
            fast_tau_list = []
            T = x_img.shape[1]
            for t in range(T - 1):
                _yi_hat, _yv_hat, state, fast_tau = self.model(
                    x_img[:, t], x_joint[:, t], state
                )
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                fast_tau_list.append(fast_tau)
                
            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))
            fast_tau_list = torch.stack(fast_tau_list).reshape(len(fast_tau_list),-1)
            fast_tau_list = torch.mean(fast_tau_list, dim=1)
            img_mses = torch.zeros(yi_hat.shape[0], yi_hat.shape[1], 1)       # batch * seq * 1
            joint_mses = torch.zeros(yv_hat.shape[0], yv_hat.shape[1], 1)     # batch * seq * 1
            for i in range(yi_hat.shape[0]):
                for j in range(yi_hat.shape[1]):
                    img_mses[i,j] = nn.MSELoss()(yi_hat[i,j], y_img[:, 1:][i,j])
                    joint_mses[i,j] = nn.MSELoss()(yv_hat[i,j], y_joint[:, 1:][i,j])
            img_vars = torch.zeros(yi_hat.shape[0], 1)
            joint_vars = torch.zeros(yv_hat.shape[0], 1)
            for i in range(yi_hat.shape[0]):
                img_vars[i] = torch.var(img_mses[i])
                joint_vars[i] = torch.var(joint_mses[i])
            img_losses = torch.mean(img_mses, 1) + img_vars*10.0
            joint_losses = torch.mean(joint_mses, 1) + joint_vars*10.0
            losses = img_losses + joint_losses
            loss = torch.mean(losses)
            # import ipdb; ipdb.set_trace()
            # img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]
            # joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]
            # loss = img_loss + joint_loss
            total_loss += loss.item()

            if training:
                # self.optimizer.zero_grad(set_to_none=True)
                # loss.backward()
                # self.optimizer.step()
                self.optimizers[0].zero_grad()
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.optimizers[1].step()

        return total_loss/(n_batch + 1), fast_tau_list
