import torch
import torch.nn as nn
from typing import Dict

import sys
try:
    from libs.layer import VMTRNNCell
except:
    sys.path.append("libs/")
    from layer import VMTRNNCell
    

class CNNVMTRNN(nn.Module):
    def __init__(
        self, 
        context_size:Dict[str, int], 
        init_tau:Dict[str, float],
        fast_tau_range:Dict[str, float], 
        slow_tau_range:Dict[str, float], 
        feat_size=10,
        trainInitialValue=True
    ):
        super(CNNVMTRNN, self).__init__()
        
        self.activation = torch.tanh

        # Encoder
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1)       #Output Shape [8, 14, 14]
        self.conv2 = nn.Conv2d(8, 16, 4, 2, 1)       #Output Shape [16, 7, 7]
        self.conv3 = nn.Conv2d(16,32, 4, 2, 1)       #Output Shape [32, 2, 2]
        self.l4    = nn.Linear(3*3*32, 100)          #Output Shape [100]
        self.l5    = nn.Linear(100, 10)              #Output Shape [feat_size]

        # RNN
        input_size   = 10+2
        out_vec_size = 2
        out_img_size = 100

        self.h2h = VMTRNNCell(input_size, 
                              context_size["cf"], 
                              context_size["cs"],
                              init_tau["cf"],
                              init_tau["cs"],
                              fast_tau_range, 
                              slow_tau_range)
        
        # self.h2v = nn.Linear(rec_size, out_vec_size)  # output for vector
        # self.h2d = nn.Linear(rec_size, out_img_size)  # output for decoder
        self.h_fast2v = nn.Linear(context_size["cf"], out_vec_size)
        self.h_fast2d = nn.Linear(context_size["cf"], out_img_size)

        # Decoder
        self.l7     =  nn.Linear(100, 3*3*32)         # Output Shape [288]
        self.conv8  = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=1)      #Output Shape [16, 7, 7]
        self.conv9  = nn.ConvTranspose2d(16,  8, 4, 2, padding=1, output_padding=0)                         #Output Shape [8, 14, 14]
        self.conv10 = nn.ConvTranspose2d(8,   3, 4, 2, padding=1, output_padding=0)                         #Output Shape [3, 28, 28]


    def encoder(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (x.shape[0],-1))
        x = self.l4(x)
        x = self.l5(x)
        return x

    def decoder(self, x):
        x = self.l7(x)
        x = torch.reshape(x, (x.shape[0],-1,3,3))
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        return x

    def forward(self, xi, xv, context=None):
        # Encoder
        im_hid = self.encoder(xi)
        
        # RNN
        x = torch.cat((im_hid, xv), dim=1)
        ### hogehoge
        new_h_fast, new_h_slow, new_u_fast, new_u_slow, fast_tau, slow_tau = self.h2h(x, context)
        
        out_vec     = self.h_fast2v(new_h_fast)
        out_im_feat = self.h_fast2d(new_h_fast)
        
        # Decoder
        out_img = self.decoder(out_im_feat)

        return out_img, out_vec, new_h_fast, new_h_slow, new_u_fast, new_u_slow, fast_tau, slow_tau


# if __name__ == '__main__':
#     from torchinfo import summary
#     batch_size = 50

#     rnn_model = CNNMTRNN(context_size={"cf": 50, "cs": 10}, tau={"cf": 2.0, "cs": 5.0})
#     summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
