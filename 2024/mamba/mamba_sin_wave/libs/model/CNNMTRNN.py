import torch
import torch.nn as nn
from typing import Dict

import sys
from eipl.layer import MTRNNCell
# from eipl.layer import MTRNNwithAttenCell

class CNNMTRNN(nn.Module):
    def __init__(
        self, 
        context_size:Dict[str, int], 
        tau:Dict[str, torch.Tensor], 
        feat_size=10,
        trainInitialValue=True
    ):
        super(CNNMTRNN, self).__init__()
        
        self.activation = torch.tanh
        
        # RNN
        input_size   = 1
        out_size = 1
        
        self.h2h = MTRNNCell(input_size, context_size["cf"], context_size["cs"], tau["cf"], tau["cs"])
        
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
        new_h_fast, new_h_slow, new_u_fast, new_u_slow = self.h2h(x, context)
        y = self.h_fast2out(new_h_fast)
        
        state = new_h_fast, new_h_slow, new_u_fast, new_u_slow

        return y, state


# if __name__ == '__main__':
#     from torchinfo import summary
#     batch_size = 50

#     rnn_model = CNNMTRNN(context_size={"cf": 50, "cs": 10}, tau={"cf": 2.0, "cs": 5.0})
#     summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
