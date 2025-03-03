import torch
import torch.nn as nn
from typing import Dict

import sys
from eipl.layer import MTRNNCell
from eipl.utils import get_activation_fn
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
        
        self.activation = nn.ReLU()

        # Encoder: Input Shape [3, 64, 64]
        self.conv_encoder = nn.Sequential(
            # in, out, kernel, stride, padding
            nn.Conv2d(  3,  8, 4, 2, 1), # [  8, 32, 32]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(  8, 16, 4, 2, 1), # [ 16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d( 16, 32, 4, 2, 1), # [ 32,  8,  8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d( 32, 64, 4, 2, 1), # [ 64,  4,  4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d( 64,128, 4, 2, 1), # [128,  2,  2],
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256, 4, 2, 1), # [256,  1,  1]
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.linear_encoder = nn.Sequential(
            nn.Linear(1*1*256, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )

        # RNN
        input_size   = 10+5
        out_vec_size = 5
        out_img_size = 10

        self.h2h = MTRNNCell(input_size, context_size["cf"], context_size["cs"], tau["cf"], tau["cs"])
        
        self.h_fast2v = nn.Linear(context_size["cf"], out_vec_size)
        self.h_fast2d = nn.Linear(context_size["cf"], out_img_size)

        # Decoder
        self.linear_decoder = nn.Sequential(
            nn.Linear(10, 100),
            nn.Sigmoid(),
            nn.Linear(100, 1*1*256),
            nn.Sigmoid()
        )
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, 0), # [128,  2,  2]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1, 0), # [ 64,  4,  4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, 0), # [ 32,  8,  8]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d( 32,  16, 4, 2, 1, 0), # [ 16, 16, 16]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d( 16,   8, 4, 2, 1, 0), # [  8, 32, 32]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(  8,   3, 4, 2, 1, 0), # [  3, 64, 64]
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.conv_encoder(x)
        x = torch.reshape(x, (x.shape[0],-1))
        x = self.linear_encoder(x)
        return x

    def decoder(self, x):
        x = self.linear_decoder(x)
        x = torch.reshape(x, (x.shape[0],-1,1,1))
        x = self.conv_decoder(x)
        return x

    def forward(self, xi, xv, context=None):
        # Encoder
        im_hid = self.encoder(xi)
        
        # RNN
        x = torch.cat((im_hid, xv), dim=1)
        new_h_fast, new_h_slow, new_u_fast, new_u_slow = self.h2h(x, context)
        
        out_vec     = self.h_fast2v(new_h_fast)
        out_im_feat = self.h_fast2d(new_h_fast)
        
        # Decoder
        out_im = self.decoder(out_im_feat)

        state = new_h_fast, new_h_slow, new_u_fast, new_u_slow

        return out_im, out_vec, state


# if __name__ == '__main__':
#     from torchinfo import summary
#     batch_size = 50

#     rnn_model = CNNMTRNN(context_size={"cf": 50, "cs": 10}, tau={"cf": 2.0, "cs": 5.0})
#     summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
