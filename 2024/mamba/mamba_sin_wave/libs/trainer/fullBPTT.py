import torch.nn as nn
import torch
import sys
from typing import Dict
import ipdb

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import LossScheduler

try:
    from libs.utils.data import tensor2numpy
except:
    sys.path.append("libs/")
    from utils.data import tensor2numpy


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, device="cpu"):
        self.device = device
        self.optimizer = optimizer
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

    def process_epoch(self, data, ext_hs_list, training=True):
        if not training:
            self.model.eval()
        
        total_loss = 0.0
        for n_batch, (x, y) in enumerate(data):
            x = x.to(self.device)   # 3, 119, 1
            y = y.to(self.device)
            x = x.unsqueeze(dim=2).float()
            y = y.unsqueeze(dim=2).float()
            
            # state = None
            # fast_tau_list = []
            # y_hat_list = []
            # T = x.shape[1]
            
            # for t in range(T):
            #     _y = self.model(    # , state
            #         x[:,t]  #, state
            #     )
            #     y_hat_list.append(_y)
            #     # fast_tau_list.append(fast_tau)
            # # fast_tau = torch.stack([fast_tau for fast_tau in fast_tau_list])
            # # fast_tau = fast_tau.permute(1,0,2)
            # y_hat = torch.stack([y_hat for y_hat in y_hat_list])
            # y_hat = y_hat.permute(1,0,2)
            # y_hat, ssm_state, conv_state = self.model(x)
            # 3, 119, 2, 16 / 3, 119, 1
            # 3, 119, 2, 16
            y_hat, hs_list = self.model(x, ext_hs_list=ext_hs_list)
                                
            loss = nn.MSELoss()(y_hat, y)
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
        return total_loss/(n_batch + 1), hs_list
