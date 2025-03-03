import torch
import torch.nn as nn


class HierachicalRNNCell(nn.Module):
    # :: HierachicalRNNCell

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


    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if "rnn" in name:
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(p.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(p.data)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                elif "bias_hh" in name:
                    p.data.fill_(0)

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
        new_prev_rnn1 = [rnn1_state, prev_rnn1[1]]
        new_prev_rnn2 = [rnn2_state, prev_rnn2[1]]
        # prev_rnn2[0] = rnn2_state

        new_rnn1 = self.io_rnn1(xv, new_prev_rnn1)
        new_rnn2 = self.io_rnn2(xk, new_prev_rnn2)

        return new_rnn1, new_rnn2, new_union
