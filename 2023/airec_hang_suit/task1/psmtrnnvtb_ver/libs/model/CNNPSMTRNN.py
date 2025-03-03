import torch
import torch.nn as nn
# from typing import Dict

import sys
# from eipl.layer import MTRNNCell
from eipl.utils import get_activation_fn

try:
    from libs.layer import PSMTRNNCell
except:
    sys.path.append("./libs/")
    from layer import PSMTRNNCell


class CNNPSMTRNN(nn.Module):
    def __init__(
        self, 
        context_size:dict[str, int], 
        tau:dict[str, torch.Tensor], 
        feat_size=10,
        trainInitialValue=True
    ):
        super(CNNPSMTRNN, self).__init__()
        
        self.context_size = context_size
        self.activation = nn.ReLU()

        # Encoder: Input Shape [3, 64, 64], Stereo Input Shape [6, 128, 128]
        self.conv_encoder = nn.Sequential(
            # stereo ver
            nn.Conv2d(  6,  8, 4, 2, 1), # [  8, 64, 64]
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
        input_size   = 10+7+2 # left or right
        out_img_size = 100
        out_vec_size = 7
        out_state_size = 2

        self.h2h = PSMTRNNCell(input_size, self.context_size, tau)
        
        self.lh_fast2lyi_feat = nn.Linear(self.context_size["cf"], out_img_size)
        self.lh_fast2lyv = nn.Linear(self.context_size["cf"], out_vec_size)
        self.lh_fast2lys = nn.Linear(self.context_size["cf"], out_state_size)
        
        self.rh_fast2ryi_feat = nn.Linear(self.context_size["cf"], out_img_size)
        self.rh_fast2ryv = nn.Linear(self.context_size["cf"], out_vec_size)
        self.rh_fast2rys = nn.Linear(self.context_size["cf"], out_state_size)

        # self.w_q = nn.Linear(input_size, input_size)
        # self.w_k = nn.Linear(input_size, input_size)
        # self.w_v = nn.Linear(input_size, input_size)
        
        # Decoder
        self.linear_decoder = nn.Sequential(
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
            nn.ConvTranspose2d(  8,   6, 4, 2, 1, 0), # [  6,128,128]
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

    def forward(self, 
                xi:torch.Tensor,
                xv:dict[str, torch.Tensor], 
                xs:dict[str, torch.Tensor], 
                state:list[torch.Tensor]=None
                ):
        # Encoder
        xi_feat = self.encoder(xi)

        # RNN
        # x = torch.cat((xi_feat, xv, xs), dim=1)
        lx = torch.cat((xi_feat, xv["left"], xs["left"]), dim=1)    # , xdv["left"]
        rx = torch.cat((xi_feat, xv["right"], xs["right"]), dim=1)  # , xdv["right"]
        x = {"left": lx, "right": rx}
        
        state = self.h2h(x, state)
        
        [new_h_fast, new_h_slow, new_u_fast, new_u_slow] = state
                
        # yi_feat = self.h_fast2yi_feat(new_h_fast)
        # yv = self.h_fast2yv(new_h_fast)
        # ys = self.h_fast2ys(new_h_fast)
        new_lh_fast = new_h_fast[:,:self.context_size["cf"]]
        new_rh_fast = new_h_fast[:,:self.context_size["cf"]]
        
        lyi_feat = self.lh_fast2lyi_feat(new_lh_fast)
        ryi_feat = self.rh_fast2ryi_feat(new_rh_fast)
        lyv = self.lh_fast2lyv(new_lh_fast)
        ryv = self.rh_fast2ryv(new_rh_fast)
        lys = self.lh_fast2lys(new_lh_fast)
        rys = self.rh_fast2rys(new_rh_fast)
        
        # Decoder
        # yi = self.decoder(yi_feat)
        lyi = self.decoder(lyi_feat)
        ryi = self.decoder(ryi_feat)
        
        yi = {"left": lyi, "right": ryi}
        yv = {"left": lyv, "right": ryv}
        ys = {"left": lys, "right": rys}

        return yi, yv, ys, state


# if __name__ == '__main__':
#     from torchinfo import summary
#     batch_size = 50

#     rnn_model = CNNMTRNN(context_size={"cf": 50, "cs": 10}, tau={"cf": 2.0, "cs": 5.0})
#     summary( rnn_model, input_size=[(batch_size,3,28,28), (batch_size,2)] )
