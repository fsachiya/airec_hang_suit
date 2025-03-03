#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import get_activation_fn, normalization


class VMTRNNCell(nn.Module):
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
        fast_dim,
        slow_dim,
        init_fast_tau = 4,
        init_slow_tau = 16,
        fast_tau_range={"min": 2, "max": 6},
        slow_tau_range={"min": 12, "max": 20},
        activation="tanh",
        use_bias=False,
        use_pb=False,
    ):
        super(VMTRNNCell, self).__init__()

        self.input_dim = input_dim
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.fast_tau_pb = nn.Parameter(torch.normal(mean=0, std=1, size=(1, 100)), requires_grad=True)
        self.slow_tau_pb = nn.Parameter(torch.normal(mean=0, std=1, size=(1, 100)), requires_grad=True)
        self.init_fast_tau = init_fast_tau
        self.init_slow_tau = init_slow_tau
        # self.fast_tau = nn.Parameter(torch.tensor(4.0), requires_grad=True)
        # self.slow_tau = nn.Parameter(torch.tensor(16.0), requires_grad=True)
        self.fast_tau_range = fast_tau_range
        self.slow_tau_range = slow_tau_range
        # self.norm_tau_range = norm_tau_range
        self.use_bias = use_bias
        self.use_pb = use_pb
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = get_activation_fn(activation)
        else:
            self.activation = activation

        # Input Layers
        self.i2f = nn.Linear(input_dim, fast_dim, bias=use_bias)

        # Fast context layer
        self.f2f = nn.Linear(fast_dim, fast_dim, bias=False)
        self.f2s = nn.Linear(fast_dim, slow_dim, bias=use_bias)

        # Slow context layer
        self.s2s = nn.Linear(slow_dim, slow_dim, bias=False)
        self.s2f = nn.Linear(slow_dim, fast_dim, bias=use_bias)
        
        self.fast_tau_pb_layer = nn.Linear(100,1)
        self.slow_tau_pb_layer = nn.Linear(100,1)

    def forward(self, x, state=None, pb=None):
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
        batch_size = x.shape[0]
        if state is not None:
            prev_h_fast, prev_h_slow, prev_u_fast, prev_u_slow = state
        else:
            device = x.device
            prev_h_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_h_slow = torch.zeros(batch_size, self.slow_dim).to(device)
            prev_u_fast = torch.zeros(batch_size, self.fast_dim).to(device)
            prev_u_slow = torch.zeros(batch_size, self.slow_dim).to(device)
        
        # self.fast_tau.data = torch.clamp(self.fast_tau.data, self.fast_tau_range["min"], self.fast_tau_range["max"])
        # self.slow_tau.data = torch.clamp(self.slow_tau.data, self.slow_tau_range["min"], self.slow_tau_range["max"])
        # import ipdb; ipdb.set_trace()
        _fast_tau_pb = nn.Sigmoid()(self.fast_tau_pb_layer(self.fast_tau_pb))
        fast_tau = self.init_fast_tau + normalization(_fast_tau_pb, [0.0,1.0], [-2.0,2.0])
        
        _slow_tau_pb = nn.Sigmoid()(self.slow_tau_pb_layer(self.slow_tau_pb))
        slow_tau = self.init_slow_tau + normalization(_slow_tau_pb, [0.0,1.0], [-4.0,4.0])
        
        new_u_fast = (1.0 - 1.0 / fast_tau) * prev_u_fast + 1.0 / fast_tau * (self.i2f(x) + self.f2f(prev_h_fast) + self.s2f(prev_h_slow))
        _input_slow = self.f2s(prev_h_fast) + self.s2s(prev_h_slow)
        
        if pb is not None:
            _input_slow += pb

        new_u_slow = (1.0 - 1.0 / slow_tau) * prev_u_slow + 1.0 / slow_tau * _input_slow

        new_h_fast = self.activation(new_u_fast)
        new_h_slow = self.activation(new_u_slow)
        # print(f"grad: {self.fast_tau.grad}")
        # print(f"fast_tau_type: {self.fast_tau.device.type}")
        # print(f"fast_tau: {self.fast_tau.item():.8f}, slow_tau: {self.slow_tau.item():.8f}")
        # print(f"slow_tau: {self.slow_tau.item()}")
        
        return new_h_fast, new_h_slow, new_u_fast, new_u_slow, fast_tau, slow_tau
