import torch
import torch.nn as nn
import ipdb

class HRNNCell(nn.Module):
    def __init__(self, img_feat_dim, state_dim, sensory_dim, union1_dim, union2_dim):
        super(HRNNCell, self).__init__()
        self.img_feat_gru = nn.GRUCell(img_feat_dim, sensory_dim)
        self.state_gru = nn.GRUCell(state_dim, sensory_dim)
        
        self.union1_gru = nn.GRUCell(sensory_dim*2, union1_dim)
        self.union2_gru = nn.GRUCell(union1_dim, union2_dim)

        self.img_feat_out = nn.Linear(sensory_dim, img_feat_dim)
        self.state_out = nn.Linear(sensory_dim, state_dim)
        
        self.union1_out = nn.Linear(union1_dim, sensory_dim*2)
        self.union2_out = nn.Linear(union2_dim, union1_dim)
        
        self.sensory_dim = sensory_dim

    def forward(
        self, 
        z_img_enc, 
        x_state, 
        hid_dict={"img_feat": None, "state": None, "union1": None, "union2": None}
        ):  

        img_feat_hid = self.img_feat_gru(z_img_enc, hid_dict["img_feat"])
        state_hid = self.state_gru(x_state, hid_dict["state"])
        sensory_hid = torch.cat([img_feat_hid, state_hid], dim=1)
        
        union1_hid = self.union1_gru(sensory_hid, hid_dict["union1"])
        union2_hid = self.union2_gru(union1_hid, hid_dict["union2"])
        
        z_img_dec = self.img_feat_out(img_feat_hid)
        y_state = self.state_out(state_hid)
        
        prev_sensory_hid = self.union1_out(union1_hid)
        prev_img_feat_hid, prev_state_hid = prev_sensory_hid[:,:self.sensory_dim], prev_sensory_hid[:,self.sensory_dim:]
        prev_union1_hid = self.union2_out(union2_hid)
        prev_union2_hid = union2_hid
        
        prev_hid_dict={"img_feat": prev_img_feat_hid, "state": prev_state_hid, 
                       "union1": prev_union1_hid, "union2": prev_union2_hid}

        return z_img_dec, y_state, prev_hid_dict