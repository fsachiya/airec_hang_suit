import torch
import torch.nn as nn
from torchinfo import summary
from typing import Dict, List, Tuple

import sys
try:
    from libs.layer import SARNN
except:
    sys.path.append("./libs/")
    from layer import SARNN

class HierarchicalRNNCell(nn.Module):
    # :: HierachicalRNNCell
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
                 srnn_input_dims:Dict[str, float],
                 srnn_hid_dim=50,
                 urnn_hid_dim=20,
                 
                 heatmap_size=0.1,
                 temperature=1e-4,
                 kernel_size=3,
                 activation="lrelu",
                 img_size=[128, 128],
                ):
        super(HierarchicalRNNCell, self).__init__()

        self.srnn_hid_dim = srnn_hid_dim
        self.urnn_hid_dim = urnn_hid_dim
        
        # Sensory RNN
        # Joint and key point
        self.modal_num = len(srnn_input_dims)

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

    def get_initial_states(self, x):
        batch_size = x.shape[0]
        device = x.device
        # return hidden state and cell state
        prev_ksrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_vsrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_csrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_urnn_state = [ torch.zeros(batch_size, self.urnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.urnn_hid_dim).to(device)]
        states = [prev_ksrnn_state, prev_vsrnn_state, prev_csrnn_state, prev_urnn_state]
        return states

    def forward(self, 
                xi, xv, xc, 
                states=None): # key_point, vector, command
        if states is not None:
            prev_ksrnn_state, prev_vsrnn_state, prev_csrnn_state, prev_urnn_state = states
        else:
            prev_ksrnn_state, prev_vsrnn_state, prev_csrnn_state, prev_urnn_state = self.get_initial_states(xi)

        prev_ksrnn_state = list(prev_ksrnn_state)
        prev_vsrnn_state = list(prev_vsrnn_state)
        prev_csrnn_state = list(prev_csrnn_state)
        # concat hidden state of each rnn
        urnn_input = torch.cat((    prev_ksrnn_state[0], 
                                    prev_vsrnn_state[0],
                                    prev_csrnn_state[0]), axis=-1)

        new_urnn_state = self.URNN(urnn_input, prev_urnn_state)
        urnn_out = self.urnn_out_layer(new_urnn_state[0])
        prev_ksrnn_hid, prev_vsrnn_hid, prev_csrnn_hid = torch.split(
            urnn_out, self.srnn_hid_dim, dim=-1)

        # update rnn hidden state
        prev_ksrnn_state[0] = prev_ksrnn_hid
        prev_vsrnn_state[0] = prev_vsrnn_hid
        prev_csrnn_state[0] = prev_csrnn_hid
        
        yi, enc_pts, dec_pts, new_ksrnn_state = self.kSRNN(xi, prev_ksrnn_state)
        new_vsrnn_state = self.vSRNN(xv, prev_vsrnn_state)
        new_csrnn_state = self.cSRNN(xc, prev_csrnn_state)
        
        states = [new_ksrnn_state, new_vsrnn_state, new_csrnn_state, new_urnn_state]

        return yi, enc_pts, dec_pts, states


class HierarchicalRNN(nn.Module):
    def __init__(self,
                 srnn_input_dims={"k":8, "v": 7, "c": 3},
                 srnn_hid_dim=50,
                 urnn_hid_dim=20,
                 heatmap_size=0.1,
                 temperature=1e-4,
                 ):
        super(HierarchicalRNN, self).__init__()

        self.hrnn = HierarchicalRNNCell(
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

    def forward(self, xi, xv, xc, states=None):
        yi, enc_pts, dec_pts, states = self.hrnn(xi, xv, xc, states)
        yk = self.ksrnn_out_layer(states[0][0])
        yv = self.vsrnn_out_layer(states[1][0])
        yc = self.csrnn_out_layer(states[2][0])

        return yi, yk, yv, yc, enc_pts, dec_pts, states


if __name__ == "__main__":
    batch_size = 7
    srnn_input_dims={"k":8, "v": 14, "c": 5}

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
