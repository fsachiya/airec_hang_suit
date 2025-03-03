import torch
import torch.nn as nn
from torchinfo import summary


class HierachicalRNNCell(nn.Module):
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

    def __init__(self,
                 input_dim1,
                 input_dim2,
                 rnn_dim=50,
                 union_dim=20
                 ):
        super(HierachicalRNNCell, self).__init__()

        self.rnn_dim = rnn_dim
        self.union_dim = union_dim
        # Joint and key point
        self.modal_num = 2

        # Joint RNN
        self.io_rnn1 = nn.LSTMCell(input_dim1, rnn_dim)

        # Key point RNN
        self.io_rnn2 = nn.LSTMCell(input_dim2, rnn_dim)

        # Union RNN
        self.union_rnn = nn.LSTMCell(rnn_dim * self.modal_num, union_dim)
        self.union_out = nn.Linear(
            union_dim, rnn_dim * self.modal_num, bias=True)

    def get_initial_states(self, x):
        batch_size = x.shape[0]
        device = x.device
        # return hidden state and cell state
        prev_rnn1 = [torch.zeros(batch_size, self.rnn_dim).to(device),
                     torch.zeros(batch_size, self.rnn_dim).to(device)]
        prev_rnn2 = [torch.zeros(batch_size, self.rnn_dim).to(device),
                     torch.zeros(batch_size, self.rnn_dim).to(device)]
        prev_union = [torch.zeros(batch_size, self.union_dim).to(device),
                      torch.zeros(batch_size, self.union_dim).to(device)]

        return prev_rnn1, prev_rnn2, prev_union

    def forward(self, xv, xk, states=None):
        if states is not None:
            prev_rnn1, prev_rnn2, prev_union = states
        else:
            prev_rnn1, prev_rnn2, prev_union = self.get_initial_states(xv)

        # concat hidden state of each rnn
        inputs_u = torch.concat((prev_rnn1[0], prev_rnn2[0]), axis=-1)

        new_union = self.union_rnn(inputs_u, prev_union)
        _union_hidden_state = self.union_out(new_union[0])
        rnn1_state, rnn2_state = torch.split(
            _union_hidden_state, self.rnn_dim, dim=-1)

        # update rnn hidden state
        prev_rnn1[0] = rnn1_state
        prev_rnn2[0] = rnn2_state

        new_rnn1 = self.io_rnn1(xv, prev_rnn1)
        new_rnn2 = self.io_rnn2(xk, prev_rnn2)

        return new_rnn1, new_rnn2, new_union


class HierachicalSARNN(nn.Module):
    def __init__(self,
                 input_dim1=7,
                 input_dim2=13,
                 rnn_dim=50,
                 union_dim=20
                 ):
        super(HierachicalSARNN, self).__init__()

        self.hlstm = HierachicalRNNCell(
            input_dim1=input_dim1, input_dim2=input_dim2, rnn_dim=rnn_dim, union_dim=union_dim)
        self.decoder_joint = nn.Sequential(
            nn.Linear(rnn_dim, input_dim1, bias=True), torch.nn.LeakyReLU())
        self.decoder_point = nn.Sequential(
            nn.Linear(rnn_dim, input_dim2, bias=True), torch.nn.LeakyReLU())

    def forward(self, xv, xk, states=None):
        states = self.hlstm(xv, xk, states)
        y_joint = self.decoder_joint(states[0][0])
        y_point = self.decoder_point(states[1][0])

        return y_joint, y_point, states


if __name__ == "__main__":
    batch_size = 7
    input_dim1 = 7
    input_dim2 = 13

    # test RNNCell
    model = HierachicalRNNCell(input_dim1, input_dim2)
    summary(
        model,
        input_size=[(batch_size, input_dim1), (batch_size, input_dim2)]
    )

    # test RNNModel
    model = HierachicalSARNN(input_dim1, input_dim2)
    summary(
        model,
        input_size=[(batch_size, input_dim1), (batch_size, input_dim2)]
    )
