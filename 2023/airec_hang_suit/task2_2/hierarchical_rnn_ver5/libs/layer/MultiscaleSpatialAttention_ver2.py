import torch
from torch import nn
from einops import rearrange

class MSA(nn.Module):
    def __init__(self, att_num, img_h, img_w, temperature, device, type='concat'):
        super(MSA, self).__init__()
        self.att_num = att_num
        self.img_h = img_h
        self.img_w = img_w
        self.temperature = temperature
        self.device = device
        self.type = type
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, 1, padding=1), nn.BatchNorm2d(16)) # 16, 128, 128
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, padding=1), nn.BatchNorm2d(32)) # 32, 128, 128
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, padding=1), nn.BatchNorm2d(64)) # 64, 64, 64

        self.att_weight1 = nn.Sequential(nn.Conv2d(16, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))
        self.att_weight2 = nn.Sequential(nn.Conv2d(32, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))
        self.att_weight3 = nn.Sequential(nn.Conv2d(64, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))

        self.att_feat = nn.Sequential(nn.Conv2d(3, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))

        if type == 'concat':
            self.sum_att = nn.Sequential(nn.Conv2d(att_num*3, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))
        if type == 'add':
            self.sum_att = nn.Sequential(nn.Conv2d(att_num, att_num, 1, 1, bias=False), nn.BatchNorm2d(att_num))

        self.act = nn.LeakyReLU(True)
        self.sigmoid = nn.Sigmoid()

        i_linear = torch.linspace(0, 1.0, img_h, device=device)
        j_linear = torch.linspace(0, 1.0, img_w, device=device)

        i_grid, j_grid = torch.meshgrid(i_linear, j_linear, indexing='ij')
        coord = torch.stack([i_grid, j_grid], dim=-1)
        self.coord = coord.reshape(img_h * img_w, 2).to(device)

    def forward(self, img, state):
        feat, gate_weights = self.encoder(img, state)
        att_map = self.get_attention_map(feat)
        pt = self.get_pt(att_map).reshape(-1, self.att_num*2)
        return feat, pt, gate_weights
    
    def encoder(self, img):
        feat1 = self.act(self.conv1(img)) # 16
        feat2 = self.act(self.conv2(feat1)) # 32
        feat3 = self.act(self.conv3(feat2)) # 64

        att_weight_1 = self.sigmoid(self.att_weight1(feat1))
        att_weight_2 = self.sigmoid(self.att_weight2(feat2)) 
        att_weight_3 = self.sigmoid(self.att_weight3(feat3)) 

        att_feat = self.att_feat(img)

        if self.type == 'concat':
            out1 = att_feat * att_weight_1
            out2 = att_feat * att_weight_2
            out3 = att_feat * att_weight_3
            out = torch.cat([out1, out2, out3], dim=1)
            out = self.act(self.sum_att(out))
        if self.type == 'add':
            out1 = self.act(self.sum_att(att_feat * att_weight_1))
            out2 = self.act(self.sum_att(att_feat * att_weight_2))
            out3 = self.act(self.sum_att(att_feat * att_weight_3))
            out = (out1 + out2 + out3)/3.0
        return out
    
    def get_attention_map(self, img):
        batch_size, n, h, w = img.shape
        att_map = rearrange(img, "b n h w -> b n (h w)", n=self.att_num, h=h, w=w)
        att_map = torch.softmax(att_map/self.temperature, dim=-1)
        att_map = att_map.unsqueeze(-2)
        return att_map #b, n, 1, hw

    def get_pt(self, att_map):
        coord = self.coord.expand(att_map.shape[0], self.att_num, self.img_h*self.img_w, 2)
        ij_coord = torch.matmul(att_map, coord)
        return ij_coord.reshape(-1, self.att_num*2)

class Imgcropper(nn.Module):
    def __init__(self, att_num, img_h, img_w, temp, gpu):
        super(Imgcropper, self).__init__()
        self.att_num = att_num
        self.img_h = img_h
        self.img_w = img_w
        self.temp = temp

        i_linear = torch.linspace(0, 1.0, img_h, device=gpu)
        j_linear = torch.linspace(0, 1.0, img_w, device=gpu)
        i_grid, j_grid = torch.meshgrid(i_linear, j_linear, indexing='ij')
        coord = torch.stack([i_grid, j_grid], dim=-1)

        self.coord = coord.reshape(img_h * img_w, 2).to(gpu)

    def cropmap(self, att_pts):
        att_pts = att_pts.reshape(-1, self.att_num, 2).unsqueeze(-2)
        att_pts_split = torch.chunk(att_pts, self.att_num, dim=1)
        cropmap_list = []
        for att_pt in att_pts_split:
            coord = self.coord.expand(att_pt.shape[0], 3, self.img_h*self.img_w, 2)
            squared_dist = torch.sum((att_pt - coord)**2, dim=-1) ** 0.5
            cropmap = torch.exp(-squared_dist / self.temp)
            cropmap = cropmap.reshape(att_pt.shape[0], 3, self.img_h, self.img_w)
            cropmap_list.append(cropmap)
        return cropmap_list # att_num (bs, 3, h, w)

    def cropimg(self, img, cropmap_list):
        img_crop_list = []
        for cropmap in cropmap_list:
            img_crop = torch.mul(cropmap, img)
            img_crop_list.append(img_crop)
        summed_features = torch.stack(img_crop_list, dim=0).sum(dim=0)
        return summed_features # (bs, 3, h, w)
    
    def forward(self, img, pts):
        cropmap_list = self.cropmap(pts)
        cropimg = self.cropimg(img, cropmap_list)
        return cropimg # (bs, 3, h, w) 

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.act = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.conv3 = nn.ConvTranspose2d(16, 8, 3, 1, padding=1)
        self.conv4 = nn.ConvTranspose2d(8, 3, 3, 1, padding=1)

    def forward(self, x):
        y = self.act(self.conv1(x)) 
        y = self.act(self.conv2(y))
        y = self.act(self.conv3(y)) 
        y = self.act(self.conv4(y))
        return y 
    
class MTRNN(nn.Module):
    def __init__(self, in_dim, f_dim, s_dim, f_tau, s_tau, device):
        super(MTRNN, self).__init__()
        self.in_dim = in_dim
        self.f_dim = f_dim
        self.s_dim = s_dim
        self.f_tau = f_tau
        self.s_tau = s_tau
        self.device = device
        self.act = nn.LeakyReLU()

        # Layers
        self.in_to_fdim = nn.Linear(in_dim, f_dim, bias=True).to(device)

        self.fh_to_fdim = nn.Linear(f_dim, f_dim, bias=True).to(device)
        self.sh_to_fdim = nn.Linear(s_dim, f_dim, bias=False).to(device)

        self.sh_to_sdim = nn.Linear(s_dim, s_dim, bias=True).to(device)
        self.fh_to_sdim = nn.Linear(f_dim, s_dim, bias=False).to(device)

    def forward(self, x, hids):
        batch_size = x.shape[0]
        if hids is not None:
            fh, sh, fu, su = hids
        else:
            fh = torch.zeros(batch_size, self.f_dim).to(self.device)
            sh = torch.zeros(batch_size, self.s_dim).to(self.device)
            fu = torch.zeros(batch_size, self.f_dim).to(self.device)
            su = torch.zeros(batch_size, self.s_dim).to(self.device)
        hid = self.in_to_fdim(x)

        next_fu = (1-self.f_tau) * fu +\
                self.f_tau * (hid + self.fh_to_fdim(fh) + self.sh_to_fdim(sh))
        next_su = (1-self.s_tau) * su +\
                self.s_tau * (self.sh_to_sdim(sh) + self.fh_to_sdim(fh))

        next_fh = self.act(next_fu)
        next_sh = self.act(next_su)
        
        return (next_fh, next_sh, next_fu, next_su)
         
class MSARNN(nn.Module):
    def __init__(self,
                 device,
                 img_h=64,
                 img_w=128,
                 att_num=4,
                 robot_dim=7,
                 rnn_dim=50,
                 temperature=0.05,
                 heatmap_size=0.05,
                 ):
        super(MSARNN, self).__init__()
        self.device = device
        self.att_num = att_num
        self.robot_dim = robot_dim # x, y, g_of, g_obj
        self.rnn_dim = rnn_dim
        self.img_h = img_h
        self.img_w = img_w
        self.temperature = temperature
        self.heatmap_size = heatmap_size
        self.imgcroper = Imgcropper(att_num, img_h, img_w, heatmap_size, device)
        self.decoder = AutoEncoder()
        self.att = MSA(self.att_num, img_h, img_w, self.temperature, device)
        #self.rnn = nn.LSTMCell(self.att_num*4+self.robot_dim, self.att_num*4+self.robot_dim)
        self.rnn = MTRNN(self.att_num*2+self.robot_dim, 50, self.rnn_dim, 1./2., 1./12., device=device)
        self.rnn_decoder = nn.Sequential(nn.Linear(50, self.att_num*2+self.robot_dim))

        i_linear = torch.linspace(0, 1.0, img_h, device=device)
        j_linear = torch.linspace(0, 1.0, img_w, device=device)

        i_grid, j_grid = torch.meshgrid(i_linear, j_linear, indexing='ij')
        coord = torch.stack([i_grid, j_grid], dim=-1)

        self.coord = coord.reshape(img_h * img_w, 2).to(device)

    def forward(self, img, x_rb, state=None):
        img_feat, x_pt = self.att(img)
        x = torch.cat([x_rb, x_pt], -1)
        # y = self.rnn(x, state)
        # y = self.rnn_decoder(y[0])
        state = self.rnn(x, state)
        y = self.rnn_decoder(state[0])

        y_pt = y[:, self.robot_dim:] 
        y_rb = y[:, :self.robot_dim]

        cropimg_list = self.imgcroper(img, y_pt)
        y_img = self.decoder(cropimg_list)
        rec_img = self.decoder(img)

        return  [y_img, rec_img], y_rb, [x_pt, y_pt], state