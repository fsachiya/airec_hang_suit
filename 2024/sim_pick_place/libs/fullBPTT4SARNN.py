#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler, tensor2numpy

import ipdb

class fullBPTTtrainer4SARNN:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0], device="cpu"):
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
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_imgs, x_states), (y_imgs, y_states)) in enumerate(data):
            if "cpu" in self.device:
                x_imgs = x_imgs.to(self.device)
                y_imgs = y_imgs.to(self.device)
                x_states = x_states.to(self.device)
                y_states = y_states.to(self.device)

            state = None
            yi_list, yv_list = [], []
            dec_pts_list, enc_pts_list = [], []
            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_imgs.shape[1] - 1):
                _yi_hat, _yv_hat, enc_ij, dec_ij, state = self.model(
                    x_imgs[:, t], x_states[:, t], state
                )
                yi_list.append(_yi_hat)
                yv_list.append(_yv_hat)
                enc_pts_list.append(enc_ij)
                dec_pts_list.append(dec_ij)

            yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

            img_loss = nn.MSELoss()(yi_hat, y_imgs[:, 1:]) * self.loss_weights[0]
            state_loss = nn.MSELoss()(yv_hat, y_states[:, 1:]) * self.loss_weights[1]
            # Gradually change the loss value using the LossScheluder class.
            pt_loss = nn.MSELoss()(
                torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
            ) * self.scheduler(self.loss_weights[2])
            loss = img_loss + state_loss + pt_loss
            total_loss += tensor2numpy(loss)

            if training:
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)