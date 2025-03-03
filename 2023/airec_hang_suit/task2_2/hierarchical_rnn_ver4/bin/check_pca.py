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

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img, cos_interpolation

try:
    # from libs.model import HSARNN
    from libs.model import AttnHSARNN
    from libs.utils import moving_average

except:
    sys.path.append("./libs/")
    # from model import HSARNN
    from model import AttnHSARNN
    from utils import moving_average



# smoothing
def smoothing(data):
    N, _, vec = data.shape
    smoothed_hand = []
    for i in range(N):
        _tmp = []
        for j in range(vec):
            _tmp.append(cos_interpolation(data[i, :, j])[:, 0])
        smoothed_hand.append(np.array(_tmp).T)

    return np.array(smoothed_hand)


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("state_dir", type=str, default=None)
parser.add_argument("--idx", type=int, default=-1)
parser.add_argument("--state", type=str, choices=["k","v","p","u","i", "kc", "vc", "pc", "uc"], default="u")
args = parser.parse_args()

# restore parameters
# ckpt = sorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
ckpt = natsorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
# latest = ckpt.pop()
latest = ckpt[args.idx]
# dir_name = os.path.split(args.state_dir)[0]
dir_name = args.state_dir
day = dir_name.split('/')[-1]
params = restore_args( os.path.join(dir_name, 'args.json') )
# idx = args.idx

try:
    os.makedirs(f'./output/test_idx_{idx}/{params["tag"]}/')    # fast_tau_{int(params["fast_tau"])}/
except:
    pass

# load dataset
minmax = [params["vmin"], params["vmax"]]

data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2_x3"
# test data
# # img
# raw_left_img_data = np.load(f"{data_dir_path}/test/left_img.npy")
# # _left_img_data = np.expand_dims(raw_left_img_data[idx], 0)
# plt_left_img_data = _left_img_data = resize_img(raw_left_img_data, (params["img_size"], params["img_size"]))
# _left_img_data = normalization(_left_img_data.astype(np.float32), (0.0, 255.0), minmax)
# left_img_data = np.transpose(_left_img_data, (0, 1, 4, 2, 3))

raw_right_img_data = np.load(f"{data_dir_path}/test/right_img.npy")
# _right_img_data = np.expand_dims(raw_right_img_data[idx], 0)
plt_right_img_data = _right_img_data = resize_img(raw_right_img_data, (params["img_size"], params["img_size"]))
_right_img_data = normalization(_right_img_data.astype(np.float32), (0.0, 255.0), minmax)
right_img_data = np.transpose(_right_img_data, (0, 1, 4, 2, 3))


# joint bounds
arm_joint_bounds = np.load(f"{data_dir_path}/param/arm_joint_bounds.npy")
thresh = 0.02
for i in range(arm_joint_bounds.shape[1]):
    if arm_joint_bounds[1,i] - arm_joint_bounds[0,i] < thresh:
        arm_joint_bounds[0,i] = arm_joint_bounds[0].min()
        arm_joint_bounds[1,i] = arm_joint_bounds[1].max()
        
# pressure bounds
press_bounds = np.load(f"{data_dir_path}/param/pressure_bounds.npy")
thresh = 100
for i in range(press_bounds.shape[1]):
    if press_bounds[1,i] - press_bounds[0,i] < thresh:
        press_bounds[0,i] = press_bounds[0].min()
        press_bounds[1,i] = press_bounds[1].max()

# joint
raw_arm_joint_data = np.load(f"{data_dir_path}/test/joint_state.npy")
plt_arm_joint_data = _arm_joint_data = raw_arm_joint_data[:,:,7:]
# plt_arm_joint_data = _arm_joint_data = np.expand_dims(raw_arm_joint_data[idx], 0)[:,:,7:]
arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds[:,7:], minmax)

# cmd
raw_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
# _hand_cmd_data = np.expand_dims(raw_hand_cmd_data[idx], 0)
tg = []
to = []
for i in range(raw_hand_cmd_data.shape[0]):
    for j in range(raw_hand_cmd_data.shape[2]):
        if j == 0:
            pass
        elif j == 1:
            tg.append(raw_hand_cmd_data[i,:,j].tolist().index(1))
        elif j == 2:
            to.append(raw_hand_cmd_data[i,:,j].tolist().index(1))

plt_hand_cmd_data = _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax)

# vector
vec_data = np.concatenate((arm_joint_data, hand_cmd_data), axis=-1)

# pressure
raw_press_data = np.load(f"{data_dir_path}/test/pressure.npy")
_press_data = raw_press_data[:,:,19+3:19+6+1]
# _press_data = np.expand_dims(raw_press_data[idx], 0)[:,:,19+3:19+6+1]
plt_press_data = _press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)
press_data = normalization(_press_data, press_bounds[:,19+3:19+6+1], minmax)
# _press_data = _press_data[:,:,19+3:19+6+1]


print("vector: ", vec_data.min(), vec_data.max())
# print("image: ", left_img_data.min(), left_img_data.max())
print("image: ", right_img_data.min(), right_img_data.max())

vec_dim = vec_data.shape[-1]
press_dim = press_data.shape[-1]

# define model
model = AttnHSARNN( # HSARNN
    srnn_hid_dim = params["srnn_hid_dim"],
    urnn_hid_dim = params["urnn_hid_dim"],
    k_dim=params["k_dim"],
    vec_dim=vec_dim,
    press_dim=press_dim,
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    img_size=[params["img_size"], params["img_size"]]
    )


if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(latest, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
loss_w_dic = {"i": params["img_loss"],
              "k": params["pt_loss"], 
              "v": params["vec_loss"],
              "p": params["press_loss"]}

# image: numpy to tensor
# x_left_img = y_left_img = torch.from_numpy(left_img_data).float()
x_right_img = y_right_img = torch.from_numpy(right_img_data).float()
# joint: numpy to tensor
x_vec = y_vec = torch.from_numpy(vec_data).float()
x_press = y_press = torch.from_numpy(press_data).float()

states = None
right_ksrnn_state_list, vsrnn_state_list, psrnn_state_list, urnn_state_list = [], [], [], []
xi_feat_list = []
right_ksrnn_cell_list, vsrnn_cell_list, psrnn_cell_list, urnn_cell_list = [], [], [], []
img_size = params["img_size"]
T = x_right_img.shape[1]

for t in range(T-1):
    # predict rnn
    # y_image, y_joint, ect_pts, dec_pts, states = model(img_t, joint_t, state)
    [yri_hat, yv_hat, yp_hat, right_enc_pts, right_dec_pts, states, xi_feat] = model(
        x_right_img[:, t], x_vec[:, t], x_press[:, t], states=states    # step=500, 
    )
    [right_ksrnn_state, vsrnn_state, psrnn_state, urnn_state] = states # psrnn_state, 
    right_ksrnn_state_list.append(right_ksrnn_state[0])
    vsrnn_state_list.append(vsrnn_state[0])
    psrnn_state_list.append(psrnn_state[0])
    urnn_state_list.append(urnn_state[0])
    xi_feat_list.append(xi_feat)
    right_ksrnn_cell_list.append(right_ksrnn_state[1])
    vsrnn_cell_list.append(vsrnn_state[1])
    psrnn_cell_list.append(psrnn_state[1])
    urnn_cell_list.append(urnn_state[1])

rksrnn_states = torch.permute(torch.stack(right_ksrnn_state_list), (1, 0, 2))    # 207*7*20
rksrnn_states = tensor2numpy(rksrnn_states)
vsrnn_states = torch.permute(torch.stack(vsrnn_state_list), (1, 0, 2))    # 207*7*20
vsrnn_states = tensor2numpy(vsrnn_states)
psrnn_states = torch.permute(torch.stack(psrnn_state_list), (1, 0, 2))    # 207*7*20
psrnn_states = tensor2numpy(psrnn_states)
urnn_states = torch.permute(torch.stack(urnn_state_list), (1, 0, 2))    # 207*7*20
urnn_states = tensor2numpy(urnn_states)

xi_feats = torch.permute(torch.stack(xi_feat_list), (1, 0, 2, 3, 4))    # 207*7*8*58*58
xi_feats = xi_feats.reshape(xi_feats.shape[0], xi_feats.shape[1], -1)
xi_feats = tensor2numpy(xi_feats)

rksrnn_cells = torch.permute(torch.stack(right_ksrnn_cell_list), (1, 0, 2))    # 207*7*20
rksrnn_cells = tensor2numpy(rksrnn_cells)
vsrnn_cells = torch.permute(torch.stack(vsrnn_cell_list), (1, 0, 2))    # 207*7*20
vsrnn_cells = tensor2numpy(vsrnn_cells)
psrnn_cells = torch.permute(torch.stack(psrnn_cell_list), (1, 0, 2))    # 207*7*20
psrnn_cells = tensor2numpy(psrnn_cells)
urnn_cells = torch.permute(torch.stack(urnn_cell_list), (1, 0, 2))    # 207*7*20
urnn_cells = tensor2numpy(urnn_cells)

# import ipdb; ipdb.set_trace()


if args.state == "u":
    pass
elif args.state == "k":
    urnn_states = rksrnn_states
elif args.state == "v":
    urnn_states = vsrnn_states
elif args.state == "p":
    urnn_states = psrnn_states
elif args.state == "i":
    urnn_states = xi_feats
elif args.state == "kc":
    urnn_states = rksrnn_cells
elif args.state == "vc":
    urnn_states = vsrnn_cells
elif args.state == "pc":
    urnn_states = psrnn_cells
elif args.state == "uc":
    urnn_states = urnn_cells

# import ipdb; ipdb.set_trace()

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

# plt.figure()
# for i in range(0,20):
#     if i in [6,7,8,9,10,11,18,19]:
#         c = "C1"
#     else:
#         c = "C2"
#     plt.scatter(pca_val[i,:,0], pca_val[i,:,1], color=c, label=f"elem_{i}")
#     plt.scatter(pca_val[i,0,0], pca_val[i,0,1], color="k")
# # plt.legend()
# plt.savefig("./fig/test_v_pca_2d.png")

# import ipdb; ipdb.set_trace()


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
    
    for n in range(right_img_data.shape[0]):
        if n in [6,7,8,9,10,11,18,19,20]:
            c = "C1"
        else:
            c = "C2"
        
        ax.scatter(
            pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=c, s=3.0, label=f"elem_{n}"
        )

    ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
    pca_ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
    ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
    ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))
    # ax.legend(loc=2)

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]))
ani.save(f'./output/_{args.state}_epc{args.idx}_PCA_HSARNN_{params["tag"]}.gif')
# ani.save(f'./output/_train_vec_pca.gif')




# True,  True,  True,  True,  True,  True,  True,  True,  True, True,  
# True,  True, False, False, False, False, False, False, False, False, 
# False, False, False, False, False, False, False, False, False, False, 
# False, False, False, False, False, False, False, False, False, False, 
# False, False,  True,  True,  True, False, False, False,  True,  True,  
# True,  True,  True,  True, False, False, False, False, False, False, 
# False, False, False, False, False, False, False, False, False, False, 
# False, False, False, False, False, False, False, False, False, False, 
# False, True,  True,  True