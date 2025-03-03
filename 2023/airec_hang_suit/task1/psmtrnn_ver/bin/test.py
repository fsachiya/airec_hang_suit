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
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

try:
    from libs.model import CNNMTRNN
    from libs.utils import cos_interpolation
except:
    sys.path.append("./libs/")
    from model import CNNMTRNN
    from utils import cos_interpolation


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
data_dir_path = "/home/fujita/job/2023/airec_hang_apron/rosbag/HangApron2/data"
# img
img_data = np.load(f"{data_dir_path}/test/imgs.npy")[idx]
left_img_data = img_data[:,:,:,[0,2,4]]
right_img_data = img_data[:,:,:,[1,3,5]]

# joint
_img_data = np.load(f"{data_dir_path}/test/imgs.npy")[idx]
_img_data = np.expand_dims(_img_data, 0)
_img_data = np.transpose(_img_data, (0,1,4,2,3))
img_data = normalization(_img_data, (0,255), minmax)
left_img_data = img_data[:,:,[0,2,4]]
right_img_data = img_data[:,:,[1,3,5]]

# bounds
joint_bounds = np.load(f"{data_dir_path}/joint_bounds.npz")
arm_joint_bounds = joint_bounds["arm_joint_bounds"]

# arm_joint
_arm_joint_data = np.load(f"{data_dir_path}/test/arm_joints.npy")[idx]
_arm_joint_data = np.expand_dims(_arm_joint_data, 0)
arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds, minmax)

# hand_states
_hand_state_data = np.load(f"{data_dir_path}/test/hand_states.npy")[:, :, [0,2,16,17]]
_hand_state_data = cos_interpolation(_hand_state_data)
hand_state_data = np.expand_dims(_hand_state_data[idx], 0)

print("images shape:{}, min={}, max={}".format(img_data.shape, img_data.min(), img_data.max()))
print("joints shape:{}, min={}, max={}".format(arm_joint_data.shape, arm_joint_data.min(), arm_joint_data.max()))







context_size = {"cf": params["fast_context"], 
                "cs": params["slow_context"]}
tau = {"cf": params["fast_tau"], 
       "cs": params["slow_tau"]}

model = CNNMTRNN(context_size, tau)

# load weight
ckpt = torch.load(latest, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# Inference
loss_weights = [params["img_loss"], 
                params["joint_loss"], 
                params["state_loss"]
                ]


# joint: numpy to tensor
xv_data = yv_data = torch.from_numpy(arm_joint_data).float()
# state: numpy to tensor
xs_data = ys_data = torch.from_numpy(hand_state_data).float()
# image: numpy to tensor
left_img_data = torch.from_numpy(left_img_data).float()
right_img_data = torch.from_numpy(right_img_data).float()
xi_data = yi_data = torch.stack([ left_img_data[:,:,0],
                                right_img_data[:,:,0],
                                left_img_data[:,:,1],
                                right_img_data[:,:,1],
                                left_img_data[:,:,2],
                                right_img_data[:,:,2]], axis=2)

state = None
yi_hat_list, yv_hat_list, ys_hat_list = [], [], []
T = xi_data.shape[1]
for t in range(T-1):
    # # load data and normalization
    # # left_img_data = img_data[:,:,[0,2,4]]
    # # right_img_data = img_data[:,:,[1,3,5]]
    # x_img = img_data[loop_ct].transpose(2, 0, 1)
    # x_img = torch.Tensor(np.expand_dims(x_img, 0))
    # x_img = normalization(x_img, (0, 255), minmax)
    # x_joint = torch.Tensor(np.expand_dims(arm_joint_data[loop_ct], 0))
    # x_joint = normalization(x_joint, arm_joint_bounds, minmax)
    # x_state= torch.Tensor(np.expand_dims(hand_state_data[loop_ct], 0))

    # if not (loop_ct == nloop-1):
    #     t_img = img_data[loop_ct+1].transpose(2, 0, 1)
    #     t_img = torch.Tensor(np.expand_dims(t_img, 0))
    #     t_img = normalization(t_img, (0, 255), minmax)
    #     t_joint = torch.Tensor(np.expand_dims(arm_joint_data[loop_ct], 0))
    #     t_joint = normalization(t_joint, arm_joint_bounds, minmax)
    #     t_state= torch.Tensor(np.expand_dims(hand_state_data[loop_ct], 0))

    # # predict rnn
    # y_img, y_joint, y_state, state = model(x_img, x_joint, x_state, state)
    
    # # yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
    # # yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))
    
    # img_loss = nn.MSELoss()(y_img, t_img) * loss_weights[0]
    # joint_loss = nn.MSELoss()(y_joint, t_joint) * loss_weights[1]
    # state_loss = nn.MSELoss()(y_state, t_state) * loss_weights[1]
    # loss = img_loss + joint_loss + state_loss
    # loss_list.append(tensor2numpy(loss))

    # # denormalization
    # pred_img = tensor2numpy(y_img[0])
    # pred_img = deprocess_img(pred_img, params["vmin"], params["vmax"])
    # pred_img = pred_img.transpose(1, 2, 0)
    # pred_joint = tensor2numpy(y_joint[0])
    # pred_joint = normalization(pred_joint, minmax, arm_joint_bounds)
    # pred_state = tensor2numpy(y_state[0])
    
    # # append data
    # img_list.append(pred_img)
    # joint_list.append(pred_joint)
    # state_list.append(pred_state)

    # print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))
    yi_hat, yv_hat, ys_hat, state = model(
        xi_data[:, t], xv_data[:, t], xs_data[:, t], state
    )
    yi_hat_list.append(yi_hat)
    yv_hat_list.append(yv_hat)
    ys_hat_list.append(ys_hat)  

# pred_img_data = np.array(img_list)
# pred_left_img_data = pred_img_data[:,:,:,[0,2,4]]
# pred_right_img_data = pred_img_data[:,:,:,[1,3,5]]
# pred_arm_joint_data = np.array(joint_list)
# pred_hand_state_data = np.array(state_list)
# loss_list = np.array(loss_list)

yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
yv_hat_data = torch.permute(torch.stack(yv_hat_list), (1, 0, 2))
ys_hat_data = torch.permute(torch.stack(ys_hat_list), (1, 0, 2))

img_loss = ((yi_hat_data - yi_data[:, 1:])**2*loss_weights[0]).mean(dim=[2,3,4])
joint_loss = ((yv_hat_data - yv_data[:, 1:])**2*loss_weights[1]).mean(dim=2)
state_loss = ((ys_hat_data - ys_data[:, 1:])**2*loss_weights[2]).mean(dim=2)

loss = img_loss + joint_loss + state_loss 
import ipdb; ipdb.set_trace()

_pred_img_data = yi_hat_data.detach().numpy()
_pred_img_data = np.transpose(_pred_img_data, (0,1,3,4,2))
pred_img_data = normalization(_pred_img_data, minmax, (0,255))
pred_left_img_data = pred_img_data[:,:,:,:,[0,2,4]]
pred_right_img_data = pred_img_data[:,:,:,:,[1,3,5]]

_pred_arm_joint_data = yv_hat_data.detach().numpy()
pred_arm_joint_data = normalization(_pred_arm_joint_data, minmax, arm_joint_bounds)

pred_hand_state_data = ys_hat_data.detach().numpy()

# img
_left_img_data = tensor2numpy(left_img_data.permute(0,1,3,4,2))
_left_img_data = normalization(_left_img_data, minmax, (0,255))
_left_img_data = _left_img_data[:,1:].astype(int)
_left_img_data = _left_img_data[0]

_right_img_data = tensor2numpy(right_img_data.permute(0,1,3,4,2))
_right_img_data = normalization(_right_img_data, minmax, (0,255))
_right_img_data = _right_img_data[:,1:].astype(int)
_right_img_data = _right_img_data[0]

_pred_left_img_data = pred_left_img_data.astype(int)
_pred_left_img_data = _pred_left_img_data[0]

_pred_right_img_data = pred_right_img_data.astype(int)
_pred_right_img_data = _pred_right_img_data[0]

# joint
_arm_joint_data = normalization(arm_joint_data, minmax, arm_joint_bounds)
_arm_joint_data = _arm_joint_data[0]
_pred_arm_joint_data = pred_arm_joint_data[0]

# state
_hand_state_data = hand_state_data[0]
_pred_hand_state_data = pred_hand_state_data[0]


T = len(img_data[0]-1) - 1
fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=60)
def anim_update(i):
    print(i)
    for j in range(4):
        for k in range(2):
            ax[k][j].cla()
        
    # plot left/right cam image
    ax[0][0].imshow(_left_img_data[i, :, :, ::-1])
    ax[0][0].axis("off")
    ax[0][0].set_title("left cam img")
    ax[1][0].imshow(_right_img_data[i, :, :, ::-1])
    ax[1][0].axis("off")
    ax[1][0].set_title("right cam img")
    
    # plot left/right pred img
    ax[0][1].imshow(_pred_left_img_data[i, :, :, ::-1])
    ax[0][1].axis("off")
    ax[0][1].set_title("left pred img")
    ax[1][1].imshow(_pred_right_img_data[i, :, :, ::-1])
    ax[1][1].axis("off")
    ax[1][1].set_title("right pred img")
    
    # plot left joint angle
    ax[0][2].set_ylim(-2.0, 2.5)
    ax[0][2].set_xlim(0, T)
    ax[0][2].plot(_arm_joint_data[1:, :7], linestyle="dashed", c="k")
    # left arm has 7 joints, not 8
    for joint_idx in range(14)[:7]:
        ax[0][2].plot(np.arange(i + 1), _pred_arm_joint_data[: i + 1, joint_idx])
    ax[0][2].set_xlabel("Step")
    ax[0][2].set_title("left arm joint angles")

     # plot right joint angle
    ax[1][2].set_ylim(-2.0, 2.5)
    ax[1][2].set_xlim(0, T)
    ax[1][2].plot(_arm_joint_data[1:, 7:], linestyle="dashed", c="k")
    # right arm has 7 joints, not 8
    for joint_idx in range(14)[7:]:
        ax[1][2].plot(np.arange(i + 1), _pred_arm_joint_data[: i + 1, joint_idx])
    ax[1][2].set_xlabel("Step")
    ax[1][2].set_title("right arm joint angles")
    
    # plot left state
    ax[0][3].set_ylim(-1.0, 2.0)
    ax[0][3].set_xlim(0, T)
    ax[0][3].plot(_hand_state_data[1:, :2], linestyle="dashed", c="k")
    # state has 7
    for state_idx in range(4)[:2]:
        ax[0][3].plot(np.arange(i + 1), _pred_hand_state_data[: i + 1, state_idx])
    ax[0][3].set_xlabel("Step")
    ax[0][3].set_title("left hand state")
    
    # plot right state
    ax[1][3].set_ylim(-1.0, 2.0)
    ax[1][3].set_xlim(0, T)
    ax[1][3].plot(_hand_state_data[1:, 2:], linestyle="dashed", c="k")
    # state has 7
    for state_idx in range(4)[2:]:
        ax[1][3].plot(np.arange(i + 1), _pred_hand_state_data[: i + 1, state_idx])
    ax[1][3].set_xlabel("Step")
    ax[1][3].set_title("right hand state")

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save(f'./output/test_idx_{idx}/{params["tag"]}/MTRNN_{params["tag"]}.gif')

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
    "context_size": {"cf": params["fast_context"], 
                     "cs": params["slow_context"]},
    "tau": {"cf": params["fast_tau"], 
            "cs": params["slow_tau"]},
    "loss": {"test": float(loss.mean())}
}
with open(f'./output/test_idx_{idx}/{params["tag"]}/result.json', 'w') as f:
    json.dump(result, f, indent=2)





# loss_weights = [params["img_loss"], params["joint_loss"], params["pt_loss"]]
# img_loss = ((images - pred_image)**2).mean() * loss_weights[0]
# joint_loss = ((joints - pred_joint)**2).mean() * loss_weights[1]
# loss = img_loss + joint_loss

# import ipdb; ipdb.set_trace()