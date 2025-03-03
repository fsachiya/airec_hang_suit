import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class HierarchicalRNNCell(nn.Module):
    # :: HierachicalRNNCell

    def __init__(self,
                #  input_dim1,
                #  input_dim2,
                #  rnn_dim=50,
                #  union_dim=20
                 srnn_input_dims:Dict[str, float],
                 srnn_hid_dim=50,
                 urnn_hid_dim=20,
                 ):
        super(HierarchicalRNNCell, self).__init__()

        # self.rnn_dim = rnn_dim
        # self.union_dim = union_dim
        self.srnn_hid_dim = srnn_hid_dim
        self.urnn_hid_dim = urnn_hid_dim
        
        # Joint and key point
        self.modal_num = len(srnn_input_dims)

        # Key point RNN
        # self.io_rnn2 = nn.LSTMCell(input_dim2, rnn_dim)
        self.kSRNN = nn.LSTMCell(srnn_input_dims["k"], self.srnn_hid_dim)

        # Joint RNN
        # self.io_rnn1 = nn.LSTMCell(input_dim1, rnn_dim)
        self.vSRNN = nn.LSTMCell(srnn_input_dims["v"], self.srnn_hid_dim)

        # Union RNN
        # self.union_rnn = nn.LSTMCell(rnn_dim * self.modal_num, union_dim)
        # self.union_out = nn.Linear(
        #     union_dim, rnn_dim * self.modal_num, bias=True)
        self.URNN = nn.LSTMCell(srnn_hid_dim * self.modal_num, self.urnn_hid_dim)
        self.urnn_out_layer = nn.Linear(
            urnn_hid_dim, srnn_hid_dim * self.modal_num, bias=True)


    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        import ipdb; ipdb.set_trace()
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
        # batch_size = x.shape[0]
        # device = x.device
        # # return hidden state and cell state
        # prev_rnn1 = [torch.zeros(batch_size, self.rnn_dim).to(device),
        #              torch.zeros(batch_size, self.rnn_dim).to(device)]
        # prev_rnn2 = [torch.zeros(batch_size, self.rnn_dim).to(device),
        #              torch.zeros(batch_size, self.rnn_dim).to(device)]
        # prev_union = [torch.zeros(batch_size, self.union_dim).to(device),
        #               torch.zeros(batch_size, self.union_dim).to(device)]

        # return prev_rnn1, prev_rnn2, prev_union
        batch_size = x.shape[0]
        device = x.device
        # return hidden state and cell state
        prev_ksrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_vsrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_urnn_state = [ torch.zeros(batch_size, self.urnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.urnn_hid_dim).to(device)]
        states = [prev_ksrnn_state, prev_vsrnn_state, prev_urnn_state]
        return states

    # modality is key_point, vector(joint + cmd)
    def forward(self, 
                xk, xv, 
                states=None):
        if states is not None:
            # prev_rnn1, prev_rnn2, prev_union = states
            prev_ksrnn_state, prev_vsrnn_state, prev_urnn_state = states
        else:
            # prev_rnn1, prev_rnn2, prev_union = self.get_initial_states(xv)
            prev_ksrnn_state, prev_vsrnn_state, prev_urnn_state = self.get_initial_states(xk)
        
        prev_ksrnn_state = list(prev_ksrnn_state)
        prev_vsrnn_state = list(prev_vsrnn_state)
        
        # concat hidden state of each rnn
        # inputs_u = torch.concat((prev_rnn1[0], prev_rnn2[0]), axis=-1)
        urnn_input = torch.cat((prev_ksrnn_state[0], 
                                prev_vsrnn_state[0]), axis=-1)
        
        # new_union = self.union_rnn(inputs_u, prev_union)
        # _union_hidden_state = self.union_out(new_union[0])
        # rnn1_state, rnn2_state = torch.split(
        #     _union_hidden_state, self.rnn_dim, dim=-1)
        
        new_urnn_state = self.URNN(urnn_input, prev_urnn_state)
        urnn_out = self.urnn_out_layer(new_urnn_state[0])
        prev_ksrnn_hid, prev_vsrnn_hid = torch.split(
            urnn_out, self.srnn_hid_dim, dim=-1)

        # update rnn hidden state
        # new_prev_rnn1 = [rnn1_state, prev_rnn1[1]]
        # new_prev_rnn2 = [rnn2_state, prev_rnn2[1]]
        prev_ksrnn_state[0] = prev_ksrnn_hid
        prev_vsrnn_state[0] = prev_vsrnn_hid
        
        # prev_rnn2[0] = rnn2_state
        # new_rnn1 = self.io_rnn1(xv, new_prev_rnn1)
        # new_rnn2 = self.io_rnn2(xk, new_prev_rnn2)
        new_ksrnn_state = self.kSRNN(xk, prev_ksrnn_state)
        new_vsrnn_state = self.vSRNN(xv, prev_vsrnn_state)
        
        states = [new_ksrnn_state, new_vsrnn_state, new_urnn_state]
        
        return states
