#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler, tensor2numpy

import ipdb

class fullBPTTtrainer4SAStackRNN:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(
        self, model, 
        optimizer, 
        loss_weight_dict={"img": 0.1, "state": 1.0, "key": 1.0}, 
        device="cpu"
        ):
        assert len(loss_weight_dict.keys()) == 3, "num of loss weights is not matched"
        
        self.device = device
        self.optimizer = optimizer
        self.loss_weight_dict = loss_weight_dict
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)
        
        self.criterion = nn.MSELoss()

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
            if "cpu" == self.device:
                x_imgs = x_imgs.to(self.device)
                y_imgs = y_imgs.to(self.device)
                x_states = x_states.to(self.device)
                y_states = y_states.to(self.device)
                
            x_imgs = x_imgs[:,:-1]
            x_states = x_states[:,:-1]
            y_imgs = y_imgs[:,1:]
            y_states = y_states[:,1:]
            assert x_imgs.shape[1] == y_imgs.shape[1], "shape of x_imgs/y_imgs is not same"
            assert x_states.shape[1] == y_states.shape[1], "shape of x_states/y_states is not same"
            
            hid_dict = {"union1": None, "union2": None, "union3": None}

            if training:
                self.optimizer.zero_grad(set_to_none=True)
            
            y_imgs_hat, y_states_hat, hids_dict, hid_dict = self.model(x_imgs, x_states, hid_dict=hid_dict)

            img_loss = self.criterion(y_imgs_hat, y_imgs) * self.loss_weight_dict["img"]
            state_loss = self.criterion(y_states_hat, y_states) * self.loss_weight_dict["state"]
            loss = img_loss + state_loss
            total_loss += tensor2numpy(loss)

            if training:
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)