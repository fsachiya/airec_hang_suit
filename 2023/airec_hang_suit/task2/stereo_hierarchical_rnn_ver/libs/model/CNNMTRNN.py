import torch
import torch.nn as nn
# from typing import Dict

import sys
# from eipl.layer import MTRNNCell
from eipl.utils import get_activation_fn

try:
    from libs.layer import MTRNNCell
except:
    sys.path.append("./libs/")
    from layer import MTRNNCell


class CNNMTRNN(nn.Module):
    def __init__(
        self, 
        context_size:dict[str, int], 
        tau:dict[str, torch.Tensor], 
        feat_size=10,
        trainInitialValue=True
    ):
        super(CNNMTRNN, self).__init__()
        
        self.activation = nn.ReLU()

        # Encoder: Input Shape [3, 64, 64], Stereo Input Shape [6, 128, 128]
        self.conv_encoder = nn.Sequential(
            # stereo ver
            nn.Conv2d(  3,  8, 4, 2, 1), # [  8, 64, 64]
            self.activation,
            nn.Conv2d(  8, 16, 4, 2, 1), # [ 16, 32, 32]
            self.activation,
            nn.Conv2d( 16, 32, 4, 2, 1), # [ 32, 16, 16]
            self.activation,
            nn.Conv2d( 32, 64, 4, 2, 1), # [ 64,  8,  8]
            self.activation,
            nn.Conv2d( 64,128, 4, 2, 1), # [128,  4,  4]
            self.activation,
            nn.Conv2d(128,256, 4, 2, 1), # [256,  2,  2],
            self.activation,
            nn.Conv2d(256,512, 4, 2, 1), # [512,  1,  1]
            self.activation
        )
        
        self.linear_encoder = nn.Sequential(
            # nn.Linear(1*1*256, 100),
            # stereo ver
            nn.Linear(1*1*512, 100),
            self.activation,
            nn.Linear(100, 10),
            self.activation
        )
        
        # RNN
        input_size   = 10+14+5  # img_feat, arm_joint, hand_state
        out_img_size = 100
        out_vec_size = 14
        out_state_size = 5

        self.h2h = MTRNNCell(input_size, context_size, tau)
        
        self.h_fast2yi_feat = nn.Linear(context_size["cf"], out_img_size)
        self.h_fast2yv = nn.Linear(context_size["cf"], out_vec_size)
        self.h_fast2ys = nn.Linear(context_size["cf"], out_state_size)

        self.w_q = nn.Linear(input_size, input_size)
        self.w_k = nn.Linear(input_size, input_size)
        self.w_v = nn.Linear(input_size, input_size)
        
        # Decoder
        self.linear_decoder = nn.Sequential(
            # nn.Linear(100, 1*1*256),
            # stereo ver
            nn.Linear(100, 1*1*512),
            self.activation
        )
        self.conv_decoder = nn.Sequential(
            # stereo ver
            nn.ConvTranspose2d(512, 256, 4, 2, 1, 0), # [256,  2,  2]
            self.activation,
            nn.ConvTranspose2d(256, 128, 4, 2, 1, 0), # [128,  4,  4]
            self.activation,
            nn.ConvTranspose2d(128,  64, 4, 2, 1, 0), # [ 64,  8,  8]
            self.activation,
            nn.ConvTranspose2d( 64,  32, 4, 2, 1, 0), # [ 32, 16, 16]
            self.activation,
            nn.ConvTranspose2d( 32,  16, 4, 2, 1, 0), # [ 16, 32, 32]
            self.activation,
            nn.ConvTranspose2d( 16,   8, 4, 2, 1, 0), # [  8, 64, 64]
            self.activation,
            nn.ConvTranspose2d(  8,   3, 4, 2, 1, 0), # [  6,128,128]
            self.activation
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

    def forward(self, xi, xv, xs, state=None):
        # Encoder
        xi_feat = self.encoder(xi)

        # RNN
        x = torch.cat((xi_feat, xv, xs), dim=1)
        state = self.h2h(x, state)
        
        new_h_fast, new_h_slow, new_u_fast, new_u_slow = state
        yi_feat = self.h_fast2yi_feat(new_h_fast)
        yv = self.h_fast2yv(new_h_fast)
        ys = self.h_fast2ys(new_h_fast)
        
        # Decoder
        yi = self.decoder(yi_feat)

        return yi, yv, ys, state


# if __name__ == '__main__':
#     from torchinfo import summary
#     batch_size = 50

#     rnn_model = CNNMTRNN(context_size={"cf": 50, "cs": 10}, tau={"cf": 2.0, "cs": 5.0})
#     summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
