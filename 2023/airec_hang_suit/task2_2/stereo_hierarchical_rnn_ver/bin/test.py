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
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import japanize_matplotlib
import cv2

import torch
import torch.nn as nn
import argparse

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img, cos_interpolation

try:
    from libs.model import StereoHSARNN
    from libs.utils import moving_average
except:
    sys.path.append("./libs/")
    from model import StereoHSARNN
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

data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2"
# test data
# img
raw_left_img_data = np.load(f"{data_dir_path}/test/left_img.npy")
_left_img_data = np.expand_dims(raw_left_img_data[idx], 0)
plt_left_img_data = _left_img_data = resize_img(_left_img_data, (params["img_size"], params["img_size"]))
_left_img_data = normalization(_left_img_data.astype(np.float32), (0.0, 255.0), minmax)
left_img_data = np.transpose(_left_img_data, (0, 1, 4, 2, 3))

raw_right_img_data = np.load(f"{data_dir_path}/test/right_img.npy")
_right_img_data = np.expand_dims(raw_right_img_data[idx], 0)
plt_right_img_data = _right_img_data = resize_img(_right_img_data, (params["img_size"], params["img_size"]))
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
plt_arm_joint_data = _arm_joint_data = np.expand_dims(raw_arm_joint_data[idx], 0)[:,:,7:]
arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds[:,7:], minmax)

# cmd
raw_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
_hand_cmd_data = np.expand_dims(raw_hand_cmd_data[idx], 0)
plt_hand_cmd_data = _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, _hand_cmd_data, step=10)
hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax)

# vector
vec_data = np.concatenate((arm_joint_data, hand_cmd_data), axis=-1)

# pressure
raw_press_data = np.load(f"{data_dir_path}/test/pressure.npy")
_press_data = np.expand_dims(raw_press_data[idx], 0)[:,:,19+3:19+6+1]
plt_press_data = _press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)
press_data = normalization(_press_data, press_bounds[:,19+3:19+6+1], minmax)
# _press_data = _press_data[:,:,19+3:19+6+1]


print("vector: ", vec_data.min(), vec_data.max())
print("image: ", left_img_data.min(), left_img_data.max())
print("image: ", right_img_data.min(), right_img_data.max())

vec_dim = vec_data.shape[-1]
press_dim = press_data.shape[-1]

# define model
model = StereoHSARNN(
    # rnn_dim=args.rnn_dim,
    # union_dim=args.union_dim,
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
x_left_img = y_left_img = torch.from_numpy(left_img_data).float()
x_right_img = y_right_img = torch.from_numpy(right_img_data).float()
# joint: numpy to tensor
x_vec = y_vec = torch.from_numpy(vec_data).float()
x_press = y_press = torch.from_numpy(press_data).float()

states = None
img_size = params["img_size"]
yli_hat_list, yri_hat_list, yv_hat_list, yp_hat_list = [], [], [], []
left_dec_pts_list, left_enc_pts_list, right_dec_pts_list, right_enc_pts_list = [], [], [], []
T = x_left_img.shape[1]
for t in range(T-1):
    # load data and normalization
    # img_t = images[loop_ct].transpose(2, 0, 1)
    # img_t = torch.Tensor(np.expand_dims(img_t, 0))
    # img_t = normalization(img_t, (0, 255), minmax)
    # joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    
    # predict rnn
    # y_image, y_joint, ect_pts, dec_pts, states = model(img_t, joint_t, state)
    [yli_hat, yri_hat, yv_hat, yp_hat, 
     left_enc_pts, left_dec_pts, right_enc_pts, right_dec_pts, states] = model(
        x_left_img[:, t], x_right_img[:, t], x_vec[:, t], x_press[:, t], states
    )

    # # denormalization
    # pred_image = tensor2numpy(y_image[0])
    # pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    # pred_image = pred_image.transpose(1, 2, 0)
    # pred_joint = tensor2numpy(y_joint[0])
    # # pred_joint = normalization(pred_joint, minmax, bounds)

    # append data
    # img_list.append(pred_image)
    # vec_list.append(pred_joint)
    # ect_pts_list.append(tensor2numpy(ect_pts[0]))
    # dec_pts_list.append(tensor2numpy(dec_pts[0]))
    yli_hat_list.append(yli_hat)
    yri_hat_list.append(yri_hat)
    yv_hat_list.append(yv_hat)
    yp_hat_list.append(yp_hat)
    left_enc_pts_list.append(left_enc_pts)
    left_dec_pts_list.append(left_dec_pts)
    right_enc_pts_list.append(right_enc_pts)
    right_dec_pts_list.append(right_dec_pts)
    
    print("step:{}, vec:{}".format(t, yv_hat))

# pred_image = np.array(img_list)
# pred_joint = np.array(vec_list)
yli_hat_data = torch.permute(torch.stack(yli_hat_list), (1, 0, 2, 3, 4))
yri_hat_data = torch.permute(torch.stack(yri_hat_list), (1, 0, 2, 3, 4))
yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
yp_hat_data = torch.permute(torch.stack(yp_hat_list), (1, 0, 2))
left_enc_pts_data = torch.permute(torch.stack(left_enc_pts_list), (1, 0, 2))    # [:-1]
left_dec_pts_data = torch.permute(torch.stack(left_dec_pts_list), (1, 0, 2))    # [1:]
right_enc_pts_data = torch.permute(torch.stack(right_enc_pts_list), (1, 0, 2))  # [:-1]
right_dec_pts_data = torch.permute(torch.stack(right_dec_pts_list), (1, 0, 2))  # [1:]

# calc loss
left_img_loss = (yli_hat_data - y_left_img[:, 1:])**2 * loss_w_dic["i"]
right_img_loss = (yri_hat_data - y_right_img[:, 1:])**2 * loss_w_dic["i"]
vec_loss = (yv_hat_data - y_vec[:, 1:])**2 * loss_w_dic["v"]
press_loss = (yp_hat_data - y_press[:, 1:])**2 * loss_w_dic["p"]

left_pt_loss = (left_enc_pts_data - left_dec_pts_data)**2 * loss_w_dic["k"] # * self.scheduler(self.loss_weights[2])
right_pt_loss = (right_enc_pts_data - right_dec_pts_data)**2 * loss_w_dic["k"] # * self.scheduler(self.loss_weights[2])
pt_loss = left_pt_loss + right_pt_loss

_left_enc_pts_data = left_enc_pts_data.reshape(left_enc_pts_data.shape[0],left_enc_pts_data.shape[1],-1,2)
_right_enc_pts_data = right_enc_pts_data.reshape(right_enc_pts_data.shape[0],right_enc_pts_data.shape[1],-1,2)
_left_dec_pts_data = left_dec_pts_data.reshape(left_dec_pts_data.shape[0],left_dec_pts_data.shape[1],-1,2)
_right_dec_pts_data = right_dec_pts_data.reshape(right_dec_pts_data.shape[0],right_dec_pts_data.shape[1],-1,2)

stereo_enc_pts_x_loss = (_left_enc_pts_data[:,:,:,0] - _right_enc_pts_data[:,:,:,0])**2 * (loss_w_dic["k"] * 0.1)
stereo_dec_pts_x_loss = (_left_dec_pts_data[:,:,:,0] - _right_dec_pts_data[:,:,:,0])**2 * (loss_w_dic["k"] * 0.1)
stereo_enc_pts_y_loss = (_left_enc_pts_data[:,:,:,1] - _right_enc_pts_data[:,:,:,1])**2 * (loss_w_dic["k"])
stereo_dec_pts_y_loss = (_left_dec_pts_data[:,:,:,1] - _right_dec_pts_data[:,:,:,1])**2 * (loss_w_dic["k"])
stereo_xy_loss = stereo_enc_pts_x_loss + stereo_dec_pts_x_loss + stereo_enc_pts_y_loss + stereo_dec_pts_y_loss

# loss = left_img_loss + right_img_loss + pt_loss + stereo_xy_loss + vec_loss + press_loss

left_img_loss = left_img_loss.mean(dim=(2,3,4))
right_img_loss = right_img_loss.mean(dim=(2,3,4))
vec_loss = vec_loss.mean(dim=-1)
press_loss = press_loss.mean(dim=-1)

left_pt_loss = left_pt_loss.mean(dim=-1)
right_pt_loss = right_pt_loss.mean(dim=-1)
pt_loss = left_pt_loss + right_pt_loss

stereo_enc_pts_x_loss = stereo_enc_pts_x_loss.mean(dim=-1)
stereo_dec_pts_x_loss = stereo_dec_pts_x_loss.mean(dim=-1)
stereo_enc_pts_y_loss = stereo_enc_pts_y_loss.mean(dim=-1)
stereo_dec_pts_y_loss = stereo_dec_pts_y_loss.mean(dim=-1)
stereo_xy_loss = stereo_enc_pts_x_loss + stereo_dec_pts_x_loss + stereo_enc_pts_y_loss + stereo_dec_pts_y_loss

loss = left_img_loss + right_img_loss + vec_loss + press_loss + pt_loss + stereo_xy_loss
 
# # split key points
# ect_pts = np.array(ect_pts_list)
# dec_pts = np.array(dec_pts_list)
# ect_pts = ect_pts.reshape(-1, params["k_dim"], 2) * img_size
# dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
# enc_pts = np.clip(ect_pts, 0, img_size)
# dec_pts = np.clip(dec_pts, 0, img_size)

# tensor2numpy
# img
_pred_left_img_data = yli_hat_data.detach().numpy()
_pred_left_img_data = np.transpose(_pred_left_img_data, (0,1,3,4,2))
pred_left_img_data = deprocess_img(_pred_left_img_data, vmin=minmax[0], vmax=minmax[1])

_pred_right_img_data = yri_hat_data.detach().numpy()
_pred_right_img_data = np.transpose(_pred_right_img_data, (0,1,3,4,2))
pred_right_img_data = deprocess_img(_pred_right_img_data, vmin=minmax[0], vmax=minmax[1])


# vec/joint+cmd
_pred_vec_data = yv_hat_data.detach().numpy()
_pred_arm_joint_data = _pred_vec_data[:,:,:7]
_pred_hand_cmd_data = _pred_vec_data[:,:,7:]
pred_arm_joint_data = normalization(_pred_arm_joint_data, minmax, arm_joint_bounds[:, 7:])
_pred_hand_cmd_data = normalization(_pred_hand_cmd_data, minmax, (0.0, 1.0))
pred_hand_cmd_data = np.clip(_pred_hand_cmd_data, 0.0, 1.0)


# press
_pred_press_data = yp_hat_data.detach().numpy()
pred_press_data = normalization(_pred_press_data, minmax, press_bounds[:,19+3:19+6+1])


# key_point
left_enc_pts_data = left_enc_pts_data.detach().numpy()
left_dec_pts_data = left_dec_pts_data.detach().numpy()
right_enc_pts_data = right_enc_pts_data.detach().numpy()
right_dec_pts_data = right_dec_pts_data.detach().numpy()


# arange data for plot
# img
plt_left_img_data = plt_left_img_data[0]
_pred_left_img_data = pred_left_img_data.astype(int)
plt_pred_left_img_data = _pred_left_img_data[0]

plt_right_img_data = plt_right_img_data[0]
_pred_right_img_data = pred_right_img_data.astype(int)
plt_pred_right_img_data = _pred_right_img_data[0]

# joint
plt_arm_joint_data = plt_arm_joint_data[0]
plt_pred_arm_joint_data = pred_arm_joint_data[0]

# command
plt_hand_cmd_data = plt_hand_cmd_data[0]
plt_pred_hand_cmd_data = pred_hand_cmd_data[0]

# #################################################################
# T = len(plt_left_img_data) - 1 
# fig, ax = plt.subplots(1,2, figsize=(8, 5))
# fig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.4, hspace=0.6) #, hspace=0.6

# ax[0].set_ylim(-2.0, 2.5)
# ax[0].set_xlim(0, T)
# ax[0].plot(plt_arm_joint_data[1:], linestyle="dashed", c="k")
# # right arm has 7 joints, not 8
# for joint_idx in range(7):
#     ax[0].plot(np.arange(T), plt_pred_arm_joint_data[:T, joint_idx])
# ax[0].set_xlabel("ステップ数")
# ax[0].set_ylabel("関節角度[rad]")
# ax[0].set_title("右腕の関節角度の予測値", y=-0.25)

# # plot hand_command
# ax[1].set_ylim(-0.25, 1.25)
# ax[1].set_xlim(0, T)
# ax[1].plot(plt_hand_cmd_data[1:], linestyle="dashed", c="k")
# # right command has 3
# for cmd_idx in range(3):
#     ax[1].plot(np.arange(T), plt_pred_hand_cmd_data[:T, cmd_idx])
# ax[1].set_xlabel("ステップ数")
# ax[1].set_ylabel("コマンドの値")
# ax[1].set_title("右手の形状コマンドの予測値", y=-0.25)   # , y=-0.25

# plt.savefig(f"./fig/joint_and_cmd_{idx}.png")
# plt.close()
# #################################################################

# import ipdb; ipdb.set_trace()


# pressure
plt_press_data = plt_press_data[0]
plt_pred_press_data = np.clip(pred_press_data[0], 0, 4095)

# #################################################################
# T = len(plt_left_img_data) - 1 
# plt.figure()
# plt.plot(plt_pred_press_data[1:], linestyle="dashed", c="k")
# # right arm has 7 joints, not 8
# for press_idx in range(4):
#     plt.plot(np.arange(T), plt_pred_press_data[:T, press_idx])
# plt.xlabel("ステップ数")
# plt.ylabel("圧力値(スケール値)")
# plt.savefig(f"./fig/pressure_{idx}.png")
# plt.close()
# # plt.set_title("右手の形状コマンドの予測値", y=-0.25)   # , y=-0.25
# #################################################################

# import ipdb; ipdb.set_trace()

# attention points
_left_enc_pts_data = left_enc_pts_data[0].reshape([-1, params["k_dim"], 2])
_left_dec_pts_data = left_dec_pts_data[0].reshape([-1, params["k_dim"], 2])
_right_enc_pts_data = right_enc_pts_data[0].reshape([-1, params["k_dim"], 2])
_right_dec_pts_data = right_dec_pts_data[0].reshape([-1, params["k_dim"], 2])
# _enc_pts_data[:,:,0] *= _img_data.shape[2]
# _enc_pts_data[:,:,1] *= _img_data.shape[1]
# _dec_pts_data[:,:,0] *= _img_data.shape[2]
# _dec_pts_data[:,:,1] *= _img_data.shape[1]
_left_enc_pts_data *= img_size
_left_dec_pts_data *= img_size
plt_left_enc_pts_data = np.clip(_left_enc_pts_data, 0, img_size)
plt_left_dec_pts_data = np.clip(_left_dec_pts_data, 0, img_size)

_right_enc_pts_data *= img_size
_right_dec_pts_data *= img_size
plt_right_enc_pts_data = np.clip(_right_enc_pts_data, 0, img_size)
plt_right_dec_pts_data = np.clip(_right_dec_pts_data, 0, img_size)

# plot images
# T = len(images)
T = len(plt_left_img_data) - 1 
# fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)
fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=60)
def anim_update(i):
    print(i)
    
    for j in range(2):
        for k in range(4):
            ax[j][k].cla()
    
    # plot left_cam image
    ax[0][0].imshow(plt_left_img_data[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0][0].plot(plt_left_enc_pts_data[i, j, 0], 
                      plt_left_enc_pts_data[i, j, 1], "bo", markersize=6)  # encoder
        ax[0][0].plot(plt_left_dec_pts_data[i, j, 0], 
                      plt_left_dec_pts_data[i, j, 1], "rx", markersize=6, markeredgewidth=2)  # decoder
    ax[0][0].axis("off")
    ax[0][0].set_title("left cam img")
    
    # plot right_cam image
    ax[0][1].imshow(plt_right_img_data[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0][1].plot(plt_right_enc_pts_data[i, j, 0], 
                      plt_right_enc_pts_data[i, j, 1], "bo", markersize=6)  # encoder
        ax[0][1].plot(plt_right_dec_pts_data[i, j, 0], 
                      plt_right_dec_pts_data[i, j, 1], "rx", markersize=6, markeredgewidth=2)  # decoder
    ax[0][1].axis("off")
    ax[0][1].set_title("right cam img")
    
    # plot left_pred image
    ax[1][0].imshow(plt_pred_left_img_data[i, :, :, ::-1]) # [i, :, :, ::-1]
    ax[1][0].axis("off")
    ax[1][0].set_title("left pred img")
    
    # plot right_pred image
    ax[1][1].imshow(plt_pred_right_img_data[i, :, :, ::-1]) # [i, :, :, ::-1]
    ax[1][1].axis("off")
    ax[1][1].set_title("right pred img")
    
    # plot arm_joint
    ax[0][2].set_ylim(-2.0, 2.5)
    ax[0][2].set_xlim(0, T)
    ax[0][2].plot(plt_arm_joint_data[1:], linestyle="dashed", c="k")
    # right arm has 7 joints, not 8
    for joint_idx in range(7):
        ax[0][2].plot(np.arange(i + 1), plt_pred_arm_joint_data[: i + 1, joint_idx])
    # ax[0][1].set_xlabel("Step")
    ax[0][2].set_title("right arm joint angles")
    
    # plot hand_command
    ax[1][2].set_ylim(-1.0, 2.0)
    ax[1][2].set_xlim(0, T)
    ax[1][2].plot(plt_hand_cmd_data[1:], linestyle="dashed", c="k")
    # right command has 3
    for cmd_idx in range(3):
        ax[1][2].plot(np.arange(i + 1), plt_pred_hand_cmd_data[: i + 1, cmd_idx])
    # ax[0][2].set_xlabel("Step")
    ax[1][2].set_title("right hand command")
    
    # plot pressure
    ax[0][3].set_ylim(-1.0, 4096)
    ax[0][3].set_xlim(0, T)
    ax[0][3].plot(plt_press_data[1:], linestyle="dashed", c="k")
    # right command has 3
    for press_idx in range(4):
        ax[0][3].plot(np.arange(i + 1), plt_pred_press_data[: i + 1, press_idx])
    # ax[0][2].set_xlabel("Step")
    ax[0][3].set_title("pressure")

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save(f'./output/test_idx_{idx}/{params["tag"]}/StereoHierarchicalRNN_{params["tag"]}.gif')

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")


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