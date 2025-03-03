import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import ipdb

class Inf4StackRNN:
    def __init__(
        self, 
        model,
        open_ratio=1.0, 
        img_size=64,
        print_log=True
        ):
        self.model = model
        self.img_size = img_size
        self.open_ratio = open_ratio
        self.print_log = print_log
        
    def inf(self, x_imgs, x_states):
        y_img_hat_list, y_state_hat_list = [], []
        hids_dict = {"union1": [], "union2": [], "union3": []}
        hid_dict = {"union1": None, "union2": None, "union3": None}
        
        nloop = x_imgs.shape[1]
        for loop_ct in range(nloop):
            x_img = x_imgs[:,loop_ct].unsqueeze(1)
            x_state = x_states[:,loop_ct].unsqueeze(1)
            
            # closed loop
            if loop_ct > 0:
                x_img = self.open_ratio * x_img + (1.0 - self.open_ratio) * y_img_hat_list[-1]
                x_state = self.open_ratio * x_state + (1.0 - self.open_ratio) * y_state_hat_list[-1]

            # predict rnn
            y_img_hat, y_state_hat, _hids_dict, hid_dict = self.model(x_img, x_state, hid_dict=hid_dict)
            
            y_img_hat_list.append(y_img_hat)
            y_state_hat_list.append(y_state_hat)
            
            for key in _hids_dict.keys():
                hids_dict[key].append(_hids_dict[key])

            if self.print_log:
                print(f"loop_ct:{loop_ct}, joint:{y_state_hat.detach().clone().cpu().numpy()}")
        
        y_imgs_hat = torch.cat(y_img_hat_list, dim=1)
        y_states_hat = torch.cat(y_state_hat_list, dim=1)
        
        return y_imgs_hat, y_states_hat, hids_dict