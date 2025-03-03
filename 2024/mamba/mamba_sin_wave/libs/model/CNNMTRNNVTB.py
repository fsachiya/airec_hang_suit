import torch
import torch.nn as nn
from typing import Dict
from eipl.utils import get_activation_fn

import sys
try:
    from libs.layer import MTRNNVTBCell
except:
    sys.path.append("./libs/")
    from layer import MTRNNVTBCell
    
class CNNMTRNNVTB(nn.Module):
    def __init__(
        self, 
        context_size:Dict[str, int], 
        fast_tau_range:Dict[str, float],
        slow_tau:float,
        feat_size=10,
        trainInitialValue=True
    ):
        super(CNNMTRNNVTB, self).__init__()
        # RNN
        input_size = 1
        out_size = 1

        self.h2h = MTRNNVTBCell(input_size, context_size["cf"], context_size["cs"], fast_tau_range, slow_tau)
        
        # self.h_fast2out = nn.Linear(context_size["cf"], out_size)
        self.h_fast2out = nn.Sequential(
            nn.Linear(context_size["cf"], 10),
            nn.Tanh(),
            nn.Linear(10, out_size),
        )

    def forward(self, x, context=None):
        # Encoder
        # im_hid = self.encoder(xi)
        
        # RNN
        # x = torch.cat((im_hid, xv), dim=1)
        new_h_fast, new_h_slow, new_u_fast, new_u_slow, fast_tau = self.h2h(x, context)
        y = self.h_fast2out(new_h_fast)

        state = new_h_fast, new_h_slow, new_u_fast, new_u_slow

        return y, state, fast_tau
