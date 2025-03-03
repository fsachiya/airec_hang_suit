#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import get_activation_fn


class PSMTRNNCell(nn.Module):
    #:: MTRNNCell
    """Multiple Timescale RNN.

    Implements a form of Recurrent Neural Network (RNN) that operates with multiple timescales.
    This is based on the idea of hierarchical organization in human cognitive functions.

    Arguments:
        input_dim (int): Number of input features.
        fast_dim (int): Number of fast context neurons.
        slow_dim (int): Number of slow context neurons.
        fast_tau (float): Time constant value of fast context.
        slow_tau (float): Time constant value of slow context.
        activation (string, optional): If you set `None`, no activation is applied (ie. "linear" activation: `a(x) = x`).
        use_bias (Boolean, optional): whether the layer uses a bias vector. The default is False.
        use_pb (Boolean, optional): whether the recurrent uses a pb vector. The default is False.

    Yuichi Yamashita, Jun Tani,
    "Emergence of Functional Hierarchy in a Multiple Timescale Neural Network Model: A Humanoid Robot Experiment.", NeurIPS 2018.
    https://arxiv.org/abs/1807.03247v2
    """

    def __init__(
        self,
        input_dim,
        context_size,
        tau,
        activation="tanh",
        use_bias=False,
        use_pb=False,
    ):
        super(PSMTRNNCell, self).__init__()

        self.input_dim = input_dim
        self.fast_dim = context_size["cf"]
        self.slow_dim = context_size["cs"]
        self.fast_tau = tau["cf"]
        self.slow_tau = tau["cs"]
        self.use_bias = use_bias
        self.use_pb = use_pb

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

        # Input Layers ix->f : 19->100
        self.li2lf = nn.Linear(self.input_dim, self.fast_dim, bias=use_bias)
        self.ri2rf = nn.Linear(self.input_dim, self.fast_dim, bias=use_bias)

        # Fast context layer if->s : 200->50
        self.lf2lf = nn.Linear(self.fast_dim, self.fast_dim, bias=False)
        self.rf2rf = nn.Linear(self.fast_dim, self.fast_dim, bias=False)
        self.f2s = nn.Linear(2*self.fast_dim, self.slow_dim, bias=use_bias)
        
        # Slow context layer
        self.s2s = nn.Linear(self.slow_dim, self.slow_dim, bias=False)
        self.s2f = nn.Linear(self.slow_dim, 2*self.fast_dim, bias=use_bias)


    def forward(self, 
                x:dict[str, torch.Tensor], 
                state:list[torch.Tensor], 
                pb=None):
        """Forward propagation of the MTRNN.

        Arguments:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            state (list): Previous states (h_fast, h_slow, u_fast, u_slow), each of shape (batch_size, context_dim).
                   If None, initialize states to zeros.
            pb (bool): pb vector. Used if self.use_pb is set to True.

        Returns:
            new_h_fast (torch.Tensor): Updated fast context state.
            new_h_slow (torch.Tensor): Updated slow context state.
            new_u_fast (torch.Tensor): Updated fast internal state.
            new_u_slow (torch.Tensor): Updated slow internal state.
        """
        lx, rx = x["left"], x["right"]
        batch_size = lx.shape[0]
        if state is not None:
            [prev_h_fast, prev_h_slow, prev_u_fast, prev_u_slow] = state
        else:
            device = lx.device
            # prev_lh_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            # prev_lu_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            
            # prev_rh_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            # prev_ru_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_h_fast = torch.zeros(batch_size, 2*self.fast_dim).to(device)
            prev_u_fast = torch.zeros(batch_size, 2*self.fast_dim).to(device)
            
            prev_h_slow = torch.zeros(batch_size, self.slow_dim).to(device)
            prev_u_slow = torch.zeros(batch_size, self.slow_dim).to(device)
        
        # new_u_fast = (1.0 - 1.0 / self.fast_tau) * prev_u_fast + 1.0 / self.fast_tau * (
        #     self.i2f(x) + self.f2f(prev_h_fast) + self.s2f(prev_h_slow)
        # )
        prev_lh_fast = prev_u_fast[:, :self.fast_dim]
        prev_rh_fast = prev_u_fast[:, self.fast_dim:]
        prev_lu_fast = prev_u_fast[:, :self.fast_dim]
        prev_ru_fast = prev_u_fast[:, self.fast_dim:]
        
        new_lu_fast = (1.0 - 1.0 / self.fast_tau) * prev_lu_fast + \
            1.0 / self.fast_tau * (self.li2lf(lx) + self.lf2lf(prev_lh_fast) + self.s2f(prev_h_slow)[:, :self.fast_dim])
        new_lh_fast = self.activation(new_lu_fast)
        
        new_ru_fast = (1.0 - 1.0 / self.fast_tau) * prev_ru_fast + \
            1.0 / self.fast_tau * (self.ri2rf(rx) + self.lf2lf(prev_rh_fast) + self.s2f(prev_h_slow)[:, self.fast_dim:])
        new_rh_fast = self.activation(new_ru_fast)
        
        new_u_fast = torch.concatenate([new_lh_fast, new_rh_fast], axis=1)
        prev_h_fast = torch.concatenate([prev_lh_fast, prev_rh_fast], axis=1)
        
        new_u_slow = (1.0 - 1.0 / self.slow_tau) * prev_u_slow + 1.0 / self.slow_tau * (self.f2s(prev_h_fast) + self.s2s(prev_h_slow))

        new_h_fast = self.activation(new_u_fast)
        new_h_slow = self.activation(new_u_slow)
        
        state = new_h_fast, new_h_slow, new_u_fast, new_u_slow

        return state
