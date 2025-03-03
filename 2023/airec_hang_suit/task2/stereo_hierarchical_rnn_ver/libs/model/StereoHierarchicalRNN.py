import torch
import torch.nn as nn
from torchinfo import summary

import sys
try:
    from libs.layer import SARNN
except:
    sys.path.append("./libs/")
    from layer import SARNN

# try:
#     # from libs.model import HierarchicalRNN
#     # from libs.utils import MultimodalDataset
#     # # from libs.utils import cos_interpolation
#     # from libs.trainer import fullBPTTtrainer
# except:
#     sys.path.append("./libs/")
#     # from model import HierarchicalRNN
#     # from utils import MultimodalDataset
#     # # from utils import cos_interpolation
#     # from trainer import fullBPTTtrainer


class StereoHierarchicalRNNCell(nn.Module):
    # :: HierachicalRNNCell
    """HierachicalRNNCell

    Arguments:
        input_dim (int): Number of input features.
        fast_dim (int): Number of fast context neurons.
        slow_dim (int): Number of slow context neurons.
        fast_tau (float): Time constant value of fast context.
        slow_tau (float): Time constant value of slow context.
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias (Boolean, optional): whether the layer uses a bias vector. The default is False.
        use_pb (Boolean, optional): whether the recurrent uses a pb vector. The default is False.
    """
    
    """
    Name Rules:
        x is a one of modalities, x = k(key_pont), v(vector/joint), c(command/hand_state)
        sensory RNN: xSRNN
            dim: srnn_input/hid/out_dim
            state( = hid + cell): 
                prev/new_xsrnn_state/hid/cell
            val: xx(=xsrnn_input), yx(=xsrnn_out)
            out_layer: xsrnn_out_layer
        union RNN: URNN
            dim: urnn_input/hid/out_dim
            state( = hid + cell): 
                prev/new_urnn_state/hid/cell
            val: urnn_input/out
            out_layer: urnn_out_layer
    """

    def __init__(self,
                 srnn_input_dims:dict[str, float],
                 srnn_hid_dim=50,
                 urnn_hid_dim=20,
                 # hid_dim,
                 # k_dim=5,
                 temperature=1e-4,
                 heatmap_size=0.1,
                 kernel_size=3,
                 activation="lrelu",
                 img_size=[128, 128],
                ):
        super(StereoHierarchicalRNNCell, self).__init__()

        self.srnn_hid_dim = srnn_hid_dim
        self.urnn_hid_dim = urnn_hid_dim
        
        # Sensory RNN
        # Joint and key point
        self.modal_num = len(srnn_input_dims) + 1   # image has two modals

        # # Key point RNN
        # self.kSRNN = nn.LSTMCell(srnn_input_dims["k"], srnn_hid_dim)
        
        # hid_dim: LSTM hidden dim
        # k_dim: LSTM input dim / 2
        k_dim = int(srnn_input_dims["k"] / 2)
        self.kSRNN = SARNN(hid_dim=self.srnn_hid_dim, k_dim=k_dim)
        
        # Joint(vector) RNN
        self.vSRNN = nn.LSTMCell(srnn_input_dims["v"], self.srnn_hid_dim)

        # Command RNN
        self.cSRNN = nn.LSTMCell(srnn_input_dims["c"], self.srnn_hid_dim)

        # Union RNN
        self.URNN = nn.LSTMCell(srnn_hid_dim * self.modal_num, self.urnn_hid_dim)
        self.urnn_out_layer = nn.Linear(
            urnn_hid_dim, srnn_hid_dim * self.modal_num, bias=True)

    def get_initial_states(self, batch_size, device):
        # return hidden state and cell state
        prev_lksrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_rksrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_vsrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_csrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_urnn_state = [ torch.zeros(batch_size, self.urnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.urnn_hid_dim).to(device)]
        states = [prev_lksrnn_state, prev_rksrnn_state, prev_vsrnn_state, prev_csrnn_state, prev_urnn_state]
        return states

    def forward(self, 
                xi_dic:dict[str, torch.Tensor], 
                xv, 
                xc, 
                states=None): # key_point, vector, command
        batch_size = xv.shape[0]
        device = xv.device
        if states is not None:
            [prev_lksrnn_state, prev_rksrnn_state, 
             prev_vsrnn_state, prev_csrnn_state, 
             prev_urnn_state] = states
        else:
            [prev_lksrnn_state, prev_rksrnn_state, 
             prev_vsrnn_state, prev_csrnn_state, 
             prev_urnn_state] = self.get_initial_states(batch_size, device)

        prev_lksrnn_state = list(prev_lksrnn_state)  # left key point state
        prev_rksrnn_state = list(prev_rksrnn_state)  # right key point state
        prev_vsrnn_state = list(prev_vsrnn_state)
        prev_csrnn_state = list(prev_csrnn_state)
        # concat hidden state of each rnn
        urnn_input = torch.cat((    prev_lksrnn_state[0],
                                    prev_rksrnn_state[0],
                                    prev_vsrnn_state[0],
                                    prev_csrnn_state[0]), axis=-1)

        new_urnn_state = self.URNN(urnn_input, prev_urnn_state)
        urnn_out = self.urnn_out_layer(new_urnn_state[0])
        prev_lksrnn_hid, prev_rksrnn_hid, prev_vsrnn_hid, prev_csrnn_hid = torch.split(
            urnn_out, self.srnn_hid_dim, dim=-1)

        # update rnn hidden state
        prev_lksrnn_state[0] = prev_lksrnn_hid
        prev_rksrnn_state[0] = prev_rksrnn_hid
        prev_vsrnn_state[0] = prev_vsrnn_hid
        prev_csrnn_state[0] = prev_csrnn_hid
        
        yli, left_enc_pts, left_dec_pts, new_lksrnn_state = self.kSRNN(xi_dic["left"], prev_lksrnn_state)
        yri, right_enc_pts, right_dec_pts, new_rksrnn_state = self.kSRNN(xi_dic["right"], prev_rksrnn_state)
        new_vsrnn_state = self.vSRNN(xv, prev_vsrnn_state)
        new_csrnn_state = self.cSRNN(xc, prev_csrnn_state)
        
        yi_dic = {"left": yli, "right": yri}
        enc_pts_dic = {"left": left_enc_pts, "right": right_enc_pts}
        dec_pts_dic = {"left": left_dec_pts, "right": right_dec_pts}
        states = [new_lksrnn_state, new_rksrnn_state, new_vsrnn_state, new_csrnn_state, new_urnn_state]

        return yi_dic, enc_pts_dic, dec_pts_dic, states


class StereoHierarchicalRNN(nn.Module):
    def __init__(self,
                 srnn_input_dims={"k":8, "v": 7, "c": 3},
                 srnn_hid_dim=50,
                 urnn_hid_dim=20
                 ):
        super(StereoHierarchicalRNN, self).__init__()

        self.hrnn = StereoHierarchicalRNNCell(
            srnn_input_dims=srnn_input_dims, 
            srnn_hid_dim=srnn_hid_dim, 
            urnn_hid_dim=urnn_hid_dim)
        self.ksrnn_out_layer = nn.Sequential(
            nn.Linear(srnn_hid_dim, srnn_input_dims["k"], bias=True), 
            torch.nn.LeakyReLU())
        self.vsrnn_out_layer = nn.Sequential(
            nn.Linear(srnn_hid_dim, srnn_input_dims["v"], bias=True), 
            torch.nn.LeakyReLU())
        self.csrnn_out_layer = nn.Sequential(
            nn.Linear(srnn_hid_dim, srnn_input_dims["c"], bias=True), 
            torch.nn.LeakyReLU())

    def forward(self, xi_dic, xv, xc, states=None):
        yi_dic, enc_pts_dic, dec_pts_dic, states = self.hrnn(xi_dic, xv, xc, states)
        ylk = self.ksrnn_out_layer(states[0][0])
        yrk = self.ksrnn_out_layer(states[1][0])
        yv = self.vsrnn_out_layer(states[2][0])
        yc = self.csrnn_out_layer(states[3][0])
        
        yk_dic = {"left": ylk, "right": yrk}

        return yi_dic, yk_dic, yv, yc, enc_pts_dic, dec_pts_dic, states


if __name__ == "__main__":
    batch_size = 7
    srnn_input_dims={"k":8, "v": 7, "c": 3}

    # test RNNCell
    model = HierarchicalRNNCell(srnn_input_dims)
    summary(
        model,
        input_size=[(batch_size, srnn_input_dims["k"]), 
                    (batch_size, srnn_input_dims["v"]),
                    (batch_size, srnn_input_dims["c"])]
    )

    # test RNNModel
    model = HierarchicalRNN(srnn_input_dims)
    summary(
        model,
        input_size=[(batch_size, srnn_input_dims["k"]), 
                    (batch_size, srnn_input_dims["v"]),
                    (batch_size, srnn_input_dims["c"])]
    )
