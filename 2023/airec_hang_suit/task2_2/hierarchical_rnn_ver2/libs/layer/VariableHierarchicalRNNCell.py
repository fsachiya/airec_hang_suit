import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class VariableHierarchicalRNNCell(nn.Module):
    # :: HierachicalRNNCell

    def __init__(
            self,
            srnn_input_dims:Dict[str, float],
            srnn_hid_dim=50,
            urnn_hid_dim=20,
        ):
        super(VariableHierarchicalRNNCell, self).__init__()

        self.srnn_hid_dim = srnn_hid_dim
        self.urnn_hid_dim = urnn_hid_dim
        
        self.modal_num = len(srnn_input_dims)
        
        # # Feat RNN
        # self.fSRNN = nn.LSTMCell(srnn_input_dims["f"], self.srnn_hid_dim)
        
        # Key point RNN
        self.kSRNN = nn.LSTMCell(srnn_input_dims["k"], self.srnn_hid_dim)
        
        # Vector RNN
        self.vSRNN = nn.LSTMCell(srnn_input_dims["v"], self.srnn_hid_dim)
        
        # Pressure RNN
        self.pSRNN = nn.LSTMCell(srnn_input_dims["p"], self.srnn_hid_dim)

        # Union RNN
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
        batch_size = x.shape[0]
        device = x.device
        # return hidden state and cell state
        # prev_fsrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
        #                     torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_ksrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_vsrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_psrnn_state = [torch.zeros(batch_size, self.srnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.srnn_hid_dim).to(device)]
        prev_urnn_state = [ torch.zeros(batch_size, self.urnn_hid_dim).to(device),
                            torch.zeros(batch_size, self.urnn_hid_dim).to(device)]
        states = [prev_ksrnn_state, prev_vsrnn_state, prev_psrnn_state, prev_urnn_state]    # prev_fsrnn_state, 
        return states

    # modality is key_point, vector(joint + cmd)
    def forward(
            self, 
            xk, xv, xp,     # xf, 
            step,
            states=None
        ):
        if states is not None:
            prev_ksrnn_state, prev_vsrnn_state, prev_psrnn_state, prev_urnn_state = states  # prev_fsrnn_state, 
        else:
            prev_ksrnn_state, prev_vsrnn_state, prev_psrnn_state, prev_urnn_state = self.get_initial_states(xk)     # prev_fsrnn_state, 
        
        # prev_fsrnn_state = list(prev_fsrnn_state)
        prev_ksrnn_state = list(prev_ksrnn_state)
        prev_vsrnn_state = list(prev_vsrnn_state)
        prev_psrnn_state = list(prev_psrnn_state)
        
        # _prev_fsrnn_hid = prev_fsrnn_state[0].detach().clone()
        _prev_ksrnn_hid = prev_ksrnn_state[0].detach().clone()
        
        alpha = 1
        # delta_alpha = 0.01
        # thresh_step = 0
        # init_alpha = 0.5
        # if step < thresh_step:
        #     alpha = 0
        # else:
        #     alpha = init_alpha + delta_alpha * (step - thresh_step)
        #     if alpha > 1:
        #         alpha = 1
        # print(alpha)
        # prev_fsrnn_state[0] = alpha * prev_fsrnn_state[0]
        prev_ksrnn_state[0] = alpha * prev_ksrnn_state[0]
        
        # concat hidden state of each rnn
        urnn_input = torch.cat((prev_ksrnn_state[0], 
                                prev_vsrnn_state[0],
                                prev_psrnn_state[0]), axis=-1)  # prev_fsrnn_state[0],
        
        new_urnn_state = self.URNN(urnn_input, prev_urnn_state)   # urnn_input
        # new_urnn_state = list(new_urnn_state)
        # new_urnn_state[0] += prev_urnn_state[0]
        urnn_out = self.urnn_out_layer(new_urnn_state[0])
        prev_ksrnn_hid, prev_vsrnn_hid, prev_psrnn_hid = torch.split(
            urnn_out, self.srnn_hid_dim, dim=-1
        )   # prev_fsrnn_hid, 

        # update rnn hidden state
        # prev_fsrnn_state[0] = prev_fsrnn_hid
        # prev_ksrnn_state[0] = prev_ksrnn_hid
        # prev_fsrnn_state[0] = alpha * prev_fsrnn_hid + (1-alpha) * _prev_fsrnn_hid
        prev_ksrnn_state[0] = alpha * prev_ksrnn_hid + (1-alpha) * _prev_ksrnn_hid
        prev_vsrnn_state[0] = prev_vsrnn_hid
        prev_psrnn_state[0] = prev_psrnn_hid

        # new_fsrnn_state = self.fSRNN(xf, prev_fsrnn_state)
        # new_fsrnn_state = list(new_fsrnn_state)
        # new_fsrnn_state[0] += prev_fsrnn_state[0]
        new_ksrnn_state = self.kSRNN(xk, prev_ksrnn_state)
        # new_ksrnn_state = list(new_ksrnn_state)
        # new_ksrnn_state[0] += prev_ksrnn_state[0]
        new_vsrnn_state = self.vSRNN(xv, prev_vsrnn_state)
        # new_vsrnn_state = list(new_vsrnn_state)
        # new_vsrnn_state[0] += prev_vsrnn_state[0]
        new_psrnn_state = self.pSRNN(xp, prev_psrnn_state)
        # new_psrnn_state = list(new_psrnn_state)
        # new_psrnn_state[0] += prev_psrnn_state[0]
        
        states = [new_ksrnn_state, new_vsrnn_state, new_psrnn_state, new_urnn_state]    # new_fsrnn_state, 
        
        return states
