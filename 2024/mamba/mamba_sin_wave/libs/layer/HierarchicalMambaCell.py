import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import sys

sys.path.append("/home/fujita/work/mamba")
from _mamba import Mamba, MambaConfig


class HierarchicalMambaCell(nn.Module):
    # :: HierachicalRNNCell

    def __init__(
            self,
            smamba_input_dims:Dict[str, float],
            smamba_hid_dim=50,
            umamba_hid_dim=20,
        ):
        super(HierarchicalMambaCell, self).__init__()

        self.smamba_hid_dim = smamba_hid_dim
        self.umamba_hid_dim = umamba_hid_dim
        
        self.modal_num = len(smamba_input_dims)
        
        # # Feat RNN
        # self.fSRNN = nn.LSTMCell(srnn_input_dims["f"], self.smamba_hid_dim)
        
        kconfig = MambaConfig(d_model=smamba_input_dims["k"], 
                            n_layers=1,
                            d_state=self.smamba_hid_dim,
                            d_conv=4,
                            expand_factor=2)
        vconfig = MambaConfig(d_model=smamba_input_dims["v"], 
                            n_layers=1,
                            d_state=self.smamba_hid_dim,
                            d_conv=4,
                            expand_factor=2)
        pconfig = MambaConfig(d_model=smamba_input_dims["p"], 
                            n_layers=1,
                            d_state=self.smamba_hid_dim,
                            d_conv=4,
                            expand_factor=2)
        uconfig = MambaConfig(d_model=smamba_hid_dim * self.modal_num, 
                            n_layers=1,
                            d_state=self.umamba_hid_dim,
                            d_conv=4,
                            expand_factor=2)
        self.kmamba = Mamba(kconfig)
        self.vmamba = Mamba(vconfig)
        self.pmamba = Mamba(pconfig)
        self.umamba = Mamba(uconfig)
        
        # num_heads = 5
        # self.umamba_attn = nn.MultiheadAttention(embed_dim=smamba_hid_dim*self.modal_num, num_heads=num_heads)
        
    
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
        # prev_fsrnn_state = [torch.zeros(batch_size, self.smamba_hid_dim).to(device),
        #                     torch.zeros(batch_size, self.smamba_hid_dim).to(device)]
        prev_kmamba_state = torch.zeros(batch_size, self.smamba_hid_dim).to(device)
        prev_vmamba_state = [torch.zeros(batch_size, self.smamba_hid_dim).to(device),
                            torch.zeros(batch_size, self.smamba_hid_dim).to(device)]
        prev_pmamba_state = [torch.zeros(batch_size, self.smamba_hid_dim).to(device),
                            torch.zeros(batch_size, self.smamba_hid_dim).to(device)]
        prev_umamba_state = [ torch.zeros(batch_size, self.umamba_mamba_dim).to(device),
                            torch.zeros(batch_size, self.umamba_mamba_dim).to(device)]
        states = [prev_kmamba_state, prev_vmamba_state, prev_pmamba_state, prev_umamba_state]    # prev_fsrnn_state, 
        return states

    # modality is key_point, vector(joint + cmd)
    def forward(
            self, 
            xf, xk, xv, xp, 
            states=None
        ):
        
        if states is not None:
            prev_kmamba_state, prev_vmamba_state, prev_pmamba_state, prev_umamba_state = states  # prev_fsrnn_state, 
        else:
            prev_kmamba_state, prev_vmamba_state, prev_pmamba_state, prev_umamba_state = self.get_initial_states(xk) # prev_fsrnn_state, 
        
        # prev_fsrnn_state = list(prev_fsrnn_state)
        
        # concat hidden state of each rnn
        umamba_input = torch.cat((prev_kmamba_state, 
                                prev_vmamba_state,
                                prev_pmamba_state), axis=-1)  # prev_fsrnn_state[0],
        
        # attn_xu, attn_wu = self.umamba_attn(umamba_input, umamba_input, umamba_input)
        
        yu_hat, new_umamba_state = self.umamba(umamba_input)   # umamba_input
        
        
        prev_kmamba_hid, prev_vmamba_hid, prev_pmamba_hid = torch.split(
            yu_hat, self.smamba_hid_dim, dim=-1 # prev_fsrnn_hid, 
        )


        # new_fsrnn_state = self.fSRNN(xf, prev_fsrnn_state)
        new_kmamba_state = self.kmamba(xk, prev_kmamba_state)
        new_vmamba_state = self.vmamba(xv, prev_vmamba_state)
        new_pmamba_state = self.pmamba(xp, prev_pmamba_state)
        
        states = [new_kmamba_state, new_vmamba_state, new_pmamba_state, new_umamba_state]    # new_fsrnn_state, 
        
        return states
