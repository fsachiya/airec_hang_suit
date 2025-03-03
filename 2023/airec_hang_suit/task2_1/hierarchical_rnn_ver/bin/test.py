#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import glob
import sys
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img
from eipl.utils import normalization, resize_img, cos_interpolation

try:
    from libs.model import HierarchicalRNN
except:
    sys.path.append("./libs/")
    from model import HierarchicalRNN

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("state_dir", type=str, default=None)
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

# restore parameters
ckpt = sorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
latest = ckpt.pop()
# dir_name = os.path.split(args.state_dir)[0]
dir_name = args.state_dir
day = dir_name.split('/')[-1]
params = restore_args( os.path.join(dir_name, 'args.json') )
idx = args.idx

try:
    os.makedirs(f'./output/test_idx_{idx}/{params["tag"]}/')    # fast_tau_{int(params["fast_tau"])}/
except:
    pass

# load dataset
minmax = [params["vmin"], params["vmax"]]

data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_1"
# img
_img_data = np.load(f"{data_dir_path}/test/left_img.npy")
_img_data = resize_img(_img_data, (128, 128))
_img_data = np.transpose(_img_data, (0,1,4,2,3))
_img_data = normalization(_img_data, (0,255), minmax)[idx]
img_data = np.expand_dims(_img_data, 0)

# joint
arm_joint_bounds = np.load(f"{data_dir_path}/param/arm_joint_bounds.npy")
# arm_joint
_arm_joint_data = np.load(f"{data_dir_path}/test/joint_state.npy")
_arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds, minmax)[idx]
arm_joint_data = np.expand_dims(_arm_joint_data, 0)[:,:,7:]

# hand_cmds
# _left_hand_cmd_data = np.load(f"{data_dir_path}/test/left_hand.npy")
# _right_hand_cmd_data = np.load(f"{data_dir_path}/test/right_hand.npy")
# _hand_cmd_data = np.concatenate([_left_hand_cmd_data,
#                                      _right_hand_cmd_data], axis=-1)
# _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, 
#                                         _hand_cmd_data, step=10)[:,:,[11,12,13]]
# _hand_cmd_data = _hand_cmd_data[idx]
# hand_cmd_data = np.expand_dims(_hand_cmd_data, 0)
_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
_hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, 
                                        _hand_cmd_data, step=10)
_hand_cmd_data = _hand_cmd_data[idx]
hand_cmd_data = np.expand_dims(_hand_cmd_data, 0)

print("images shape:{}, min={}, max={}".format(img_data.shape, img_data.min(), img_data.max()))
print("joints shape:{}, min={}, max={}".format(arm_joint_data.shape, arm_joint_data.min(), arm_joint_data.max()))

model = HierarchicalRNN(srnn_input_dims={"k": params["key_point_num"] * 2, 
                                         "v": arm_joint_data.shape[-1], 
                                         "c": hand_cmd_data.shape[-1]},
                        srnn_hid_dim=params["srnn_hid_dim"],
                        urnn_hid_dim=params["urnn_hid_dim"])

# load weight
ckpt = torch.load(latest, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# Inference
loss_w_dic = {"i": params["img_loss"],
              "k": params["key_point_loss"], 
              "v": params["joint_loss"], 
              "c": params["cmd_loss"]}

# joint: numpy to tensor
xv_data = yv_data = torch.from_numpy(arm_joint_data).float()
# command: numpy to tensor
xc_data = yc_data = torch.from_numpy(hand_cmd_data).float()
# image: numpy to tensor
xi_data = yi_data = torch.from_numpy(img_data).float()

states = None
yi_hat_list, yk_hat_list, yv_hat_list, yc_hat_list = [], [], [], []
enc_pts_list, dec_pts_list = [], []
T = xi_data.shape[1]
for t in range(T-1):
    xi = xi_data[:, t]
    xv = xv_data[:, t]
    xc = xc_data[:, t]
        
    yi_hat, yk_hat, yv_hat, yc_hat, enc_pts, dec_pts, states = model(
        xi, xv, xc, states
    )
    yi_hat_list.append(yi_hat)
    yk_hat_list.append(yk_hat)
    yv_hat_list.append(yv_hat)
    yc_hat_list.append(yc_hat)  
    enc_pts_list.append(enc_pts)
    dec_pts_list.append(dec_pts)

# list2tensor
yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
yk_hat_data = torch.permute(torch.stack(yk_hat_list), (1, 0, 2))
yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
yc_hat_data = torch.permute(torch.stack(yc_hat_list), (1, 0, 2))
enc_pts_data = torch.permute(torch.stack(enc_pts_list), (1, 0, 2))
dec_pts_data = torch.permute(torch.stack(dec_pts_list), (1, 0, 2))

# calc loss
img_loss = (yi_hat_data - yi_data[:, 1:])**2 * loss_w_dic["i"]
key_point_loss = (enc_pts_data - dec_pts_data)**2 * loss_w_dic["k"]
joint_loss = (yv_hat_data - yv_data[:, 1:])**2 * loss_w_dic["v"]
cmd_loss = (yc_hat_data - yc_data[:, 1:])**2 * loss_w_dic["c"]

img_loss = img_loss.mean(dim=(2,3,4))
key_point_loss = key_point_loss.mean(dim=-1)
joint_loss = joint_loss.mean(dim=-1)
cmd_loss = cmd_loss.mean(dim=-1)
loss = img_loss + key_point_loss + joint_loss + cmd_loss

# tensor2numpy
# img
_pred_img_data = yi_hat_data.detach().numpy()
_pred_img_data = np.transpose(_pred_img_data, (0,1,3,4,2))
# pred_img_data = normalization(_pred_img_data, minmax, (0,255))
pred_img_data = deprocess_img(_pred_img_data, vmin=minmax[0], vmax=minmax[1])

# joint
_pred_arm_joint_data = yv_hat_data.detach().numpy()
pred_arm_joint_data = normalization(_pred_arm_joint_data, minmax, arm_joint_bounds[:, 7:])
# hand_cmd
pred_hand_cmd_data = yc_hat_data.detach().numpy()
# key_point
enc_pts_data = enc_pts_data.detach().numpy()
dec_pts_data = dec_pts_data.detach().numpy()

# arange data for plot
# img
_img_data = np.transpose(img_data, (0,1,3,4,2))
_img_data = normalization(_img_data, minmax, (0,255))
_img_data = _img_data[:,1:].astype(int)
_img_data = _img_data[0]

_pred_img_data = pred_img_data.astype(int)
_pred_img_data = _pred_img_data[0]

# joint
_arm_joint_data = normalization(arm_joint_data, minmax, arm_joint_bounds[:,7:])
_arm_joint_data = _arm_joint_data[0]
_pred_arm_joint_data = pred_arm_joint_data[0]

# command
_hand_cmd_data = hand_cmd_data[0]
_pred_hand_cmd_data = pred_hand_cmd_data[0]

# attention points
_enc_pts_data = enc_pts_data[0].reshape([-1, params["key_point_num"], 2])
_dec_pts_data = dec_pts_data[0].reshape([-1, params["key_point_num"], 2])
_enc_pts_data[:,:,0] *= _img_data.shape[2]
_enc_pts_data[:,:,1] *= _img_data.shape[1]
_dec_pts_data[:,:,0] *= _img_data.shape[2]
_dec_pts_data[:,:,1] *= _img_data.shape[1]


T = len(img_data[0]-1) - 1
fig, ax = plt.subplots(2, 2, figsize=(16, 8), dpi=60)
def anim_update(i):
    print(i)
    for j in range(2):
        for k in range(2):
            ax[k][j].cla()
        
    # plot cam image
    ax[0][0].imshow(_img_data[i, :, :, ::-1])
    for j in range(params["key_point_num"]):
        ax[0][0].plot(int(_enc_pts_data[i, j, 0]), 
                      int(_enc_pts_data[i, j, 1]), "bo", markersize=6)  # encoder
        ax[0][0].plot(int(_dec_pts_data[i, j, 0]), 
                      int(_dec_pts_data[i, j, 1]), "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0][0].axis("off")
    ax[0][0].set_title("cam img")
    
    # plot pred img
    ax[1][0].imshow(_pred_img_data[i, :, :, ::-1]) # [i, :, :, ::-1]
    ax[1][0].axis("off")
    ax[1][0].set_title("pred img")
    
    # # plot left joint angle
    # ax[0][1].set_ylim(-2.0, 2.5)
    # ax[0][1].set_xlim(0, T)
    # ax[0][1].plot(_arm_joint_data[1:, :7], linestyle="dashed", c="k")
    # # left arm has 7 joints, not 8
    # for joint_idx in range(14)[:7]:
    #     ax[0][1].plot(np.arange(i + 1), _pred_arm_joint_data[: i + 1, joint_idx])
    # # ax[0][1].set_xlabel("Step")
    # ax[0][1].set_title("left arm joint angles")
    # plot true/pred joint angle
    ax[0][1].set_ylim(-2.0, 2.5)
    ax[0][1].set_xlim(0, T)
    ax[0][1].plot(_arm_joint_data[1:], linestyle="dashed", c="k")
    # right arm has 7 joints, not 8
    for joint_idx in range(7):
        ax[0][1].plot(np.arange(i + 1), _pred_arm_joint_data[: i + 1, joint_idx])
    # ax[0][1].set_xlabel("Step")
    ax[0][1].set_title("right arm joint angles")

    # # plot right joint angle
    # ax[1][1].set_ylim(-2.0, 2.5)
    # ax[1][1].set_xlim(0, T)
    # ax[1][1].plot(_arm_joint_data[1:, 7:], linestyle="dashed", c="k")
    # # right arm has 7 joints, not 8
    # for joint_idx in range(14)[7:]:
    #     ax[1][1].plot(np.arange(i + 1), _pred_arm_joint_data[: i + 1, joint_idx])
    # # ax[1][1].set_xlabel("Step")
    # ax[1][1].set_title("right arm joint angles")
    
    # # plot left command
    # ax[1][1].set_ylim(-1.0, 2.0)
    # ax[1][1].set_xlim(0, T)
    # ax[1][1].plot(_hand_cmd_data[1:, :3], linestyle="dashed", c="k")
    # # command has 3
    # for cmd_idx in range(3):
    #     ax[1][1].plot(np.arange(i + 1), _pred_hand_cmd_data[: i + 1, cmd_idx])
    # # ax[0][2].set_xlabel("Step")
    # ax[1][1].set_title("right hand command")
    
    # # plot right command
    # ax[1][2].set_ylim(-1.0, 2.0)
    # ax[1][2].set_xlim(0, T)
    # ax[1][2].plot(_hand_cmd_data[1:, 3:], linestyle="dashed", c="k")
    # # command has 5
    # for cmd_idx in range(5)[2:]:
    #     ax[1][2].plot(np.arange(i + 1), _pred_hand_cmd_data[: i + 1, cmd_idx])
    # ax[1][2].set_xlabel("Step")
    # ax[1][2].set_title("right hand command")
    
    # plot true/pred command
    ax[1][1].set_ylim(-1.0, 2.0)
    ax[1][1].set_xlim(0, T)
    ax[1][1].plot(_hand_cmd_data[1:], linestyle="dashed", c="k")
    # right command has 3
    for cmd_idx in range(3):
        ax[1][1].plot(np.arange(i + 1), _pred_hand_cmd_data[: i + 1, cmd_idx])
    # ax[0][2].set_xlabel("Step")
    ax[1][1].set_title("right hand command")

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save(f'./output/test_idx_{idx}/{params["tag"]}/HierarchicalRNN_{params["tag"]}.gif')

loss = loss[0].detach().numpy()
fig = plt.figure()
plt.plot(range(len(loss)), loss, linestyle='solid', c='k', label="online")
plt.plot(range(len(loss)), loss.mean()*np.ones_like(loss), linestyle='dashed', c='r', label="average")
plt.grid()
# plt.ylim(0, 0.05)
plt.xlabel("step")
plt.ylabel("tesst_loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_idx_{idx}/{params["tag"]}/test_loss_trend.png')
plt.clf()
plt.close()

result = {
    "hid_size": {"srnn": params["srnn_hid_dim"], 
                 "urnn": params["urnn_hid_dim"]},
    "loss": {"test": float(loss.mean())}
}
with open(f'./output/test_idx_{idx}/{params["tag"]}/result.json', 'w') as f:
    json.dump(result, f, indent=2)





# loss_weights = [params["img_loss"], params["joint_loss"], params["pt_loss"]]
# img_loss = ((images - pred_image)**2).mean() * loss_weights[0]
# joint_loss = ((joints - pred_joint)**2).mean() * loss_weights[1]
# loss = img_loss + joint_loss

# import ipdb; ipdb.set_trace()