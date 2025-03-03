#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#
import ipdb
import os
import sys
import glob
from natsort import natsorted
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import argparse

sys.path.append("/home/fujita/work/mamba")
from _mamba import Mamba, MambaConfig

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img, cos_interpolation

# own library
try:
    from libs.model import CNNMTRNN
    from libs.utils import get_unique_list
    from libs.utils import deprocess_img, get_batch
    from libs.trainer import fullBPTTtrainer
    from libs.utils import sinwave

except:
    sys.path.append("libs/")
    from model import CNNMTRNN
    from utils import get_unique_list
    from utils import deprocess_img, get_batch
    from trainer import fullBPTTtrainer
    from utils import sinwave

parser = argparse.ArgumentParser(
    description="mtrnn square wave prediction test"
)
parser.add_argument('state_dir')
parser.add_argument('--model', type=str, default="mtrnn")
parser.add_argument('--total_step', type=int, default=200)
parser.add_argument('--input_param', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--plt_show_flag', action='store_true')
parser.add_argument('--idx', type=int, default=1)
args = parser.parse_args()    

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# restore parameters
ckpt = sorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
latest = ckpt.pop()
# dir_name = os.path.split(args.state_dir)[0]
dir_name = args.state_dir
day = dir_name.split('/')[-1]
params = restore_args( os.path.join(dir_name, 'args.json') )
idx = args.idx

try:
    os.makedirs(f'./output/test_pos_{idx}/{params["tag"]}/')
except:
    pass

minmax = [params["vmin"], params["vmax"]]
test_sinwave = np.array(sinwave.test_list)[idx]
test_sinwave = np.expand_dims(test_sinwave, 0)
test_x = test_sinwave[:,:-1]
test_y = test_sinwave[:,1:]

# # model of neural network
# context_size = {"cf": 100, "cs": 50}
# tau = {"cf": 4.0, "cs": 20.0}

# # model of neural network
# model = CNNMTRNN(context_size, tau)
batch, length, dim = 1, test_x.shape[1], 1
# model = Mamba(
#     d_model=dim,  # モデルの次元
#     d_state=16,   # SSMの状態の拡張係数
#     d_conv=4,     # ローカルな畳み込みの幅
#     expand=2     # ブロックの拡張係数
# ).to(device)
config = MambaConfig(d_model=dim, 
                     n_layers=1,
                     d_state=16,
                     d_conv=4,
                     expand_factor=2)
model = Mamba(config).to(device)


ckpt = torch.load(latest, map_location=torch.device(device))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ipdb.set_trace()

state = None
y_hat_list = []
T = test_x.shape[1]

x = torch.tensor(test_x).unsqueeze(dim=2).float().to(device)
y = torch.tensor(test_y).unsqueeze(dim=2).float().to(device)

# for t in range(T):
#     _y, state = model(
#         x[:,t], state
#     )
#     y_hat_list.append(_y)
# y_hat = torch.stack([y_hat for y_hat in y_hat_list])

y_hat, hs = model(x)
# y_hat = y_hat.permute(1,0,2)
loss = ((y_hat - y) ** 2)


loss_list = tensor2numpy(loss.reshape(-1))
y_hat_list = tensor2numpy(y_hat.reshape(-1))
y_list = tensor2numpy(y.reshape(-1))

urnn_states = hs.reshape([hs.shape[0], hs.shape[1], -1]).detach().cpu().numpy()
# urnn_states = hs[:,:,0].detach().cpu().numpy()

# ######
# urnn_states = arm_joint_data[:,1:]

# urnn_states = urnn_states[:,10:]
# ######

# Reshape the state from [N,T,D] to [-1,D] for PCA of RNN.
# N is the number of datasets
# T is the sequence length
# D is the dimension of the hidden state
N, T, D = urnn_states.shape
urnn_states = urnn_states.reshape(-1, D)
# PCA
loop_ct = float(360) / T
pca_dim = 3
pca = PCA(n_components=pca_dim).fit(urnn_states)
pca_val = pca.transform(urnn_states)
# Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
# visualize each state as a 3D scatter.
pca_val = pca_val.reshape(N, T, pca_dim)

# plot images
fig = plt.figure()  # dpi=60
ax = fig.add_subplot(projection="3d")
# ax.view_init(30, 90)

# c_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
# for n, color in enumerate(c_list):
#     ax.scatter(
#         pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=color, s=3.0
#     )

# ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
# pca_ratio = pca.explained_variance_ratio_ * 100
# ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
# ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
# ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))
# plt.savefig(f'./fig/_pca_3d')

def anim_update(i):
    ax.cla()
    angle = int(loop_ct * i)
    ax.view_init(30, angle)

    # c_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
    # for n, color in enumerate(c_list):
    #     ax.scatter(
    #         pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=color, s=3.0
    #     )
    
    # for n in range(right_img_data.shape[0]):
    #     ax.scatter(
    #         pca_val[n, 1:tg[n], 0], pca_val[n, 1:tg[n], 1], pca_val[n, 1:tg[n], 2], color="C0", s=3.0
    #     )
    #     ax.scatter(
    #         pca_val[n, tg[n]:to[n], 0], pca_val[n, tg[n]:to[n], 1], pca_val[n, tg[n]:to[n], 2], color="C1", s=3.0
    #     )
    #     ax.scatter(
    #         pca_val[n, to[n]:, 0], pca_val[n, to[n]:, 1], pca_val[n, to[n]:, 2], color="C2", s=3.0
    #     )
    
    for n in range(hs.shape[0]):
        c = "C1"
        ax.scatter(
            pca_val[n, 0:, 0], pca_val[n, 0:, 1], pca_val[n, 0:, 2], color=c, s=3.0, label=f"elem_{n}"
        )
        c = "C2"
        ax.plot(
            pca_val[n, 0:, 0], pca_val[n, 0:, 1], pca_val[n, 0:, 2], color=c, label=f"elem_{n}"
        )

    ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
    pca_ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
    ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
    ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]))
ani.save(f'./output/test_pos_{idx}/{params["tag"]}/pca.gif')

# f'./output/test_pos_{idx}/{params["tag"]}/sin_wave.png'
