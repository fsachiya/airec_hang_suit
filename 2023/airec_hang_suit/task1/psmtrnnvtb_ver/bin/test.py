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
except:
    sys.path.append("./libs/")
    from model import CNNMTRNN


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
data_dir_path = "/home/fujita/job/2023/airec_hang_apron/rosbag/HangApron/data"
# joint_bounds = np.load("../data/joint_bounds.npy")
# joints = np.load("../data/test/joints.npy")[idx]
# images = np.load("../data/test/images.npy")[idx]
# load test data
# img
img_data = np.load(f"{data_dir_path}/test/imgs.npy")[idx]
left_img_data = img_data[:,:,:,[0,2,4]]
right_img_data = img_data[:,:,:,[1,3,5]]

# joint
joint_bounds = np.load(f"{data_dir_path}/joint_bounds.npz")
arm_joint_bounds = joint_bounds["arm_joint_bounds"]
# arm_joint
arm_joint_data = np.load(f"{data_dir_path}/test/arm_joints.npy")[idx]
# hand_states
hand_state_data = np.load(f"{data_dir_path}/test/hand_states.npy")[idx]
print("images shape:{}, min={}, max={}".format(img_data.shape, img_data.min(), img_data.max()))
print("joints shape:{}, min={}, max={}".format(arm_joint_data.shape, arm_joint_data.min(), arm_joint_data.max()))

context_size = {"cf": params["fast_context"], "cs": params["slow_context"]}
tau = {"cf": params["fast_tau"], "cs": params["slow_tau"]}
model = CNNMTRNN(context_size, tau)

# load weight
ckpt = torch.load(latest, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


# Inference
img_size = 64
loss_weights = [params["img_loss"], 
                params["joint_loss"], 
                params["pt_loss"]]
img_list, joint_list, state_list = [], [], []
loss_list = []
state = None
nloop = len(img_data)
for loop_ct in range(nloop):
    # load data and normalization
    # left_img_data = img_data[:,:,[0,2,4]]
    # right_img_data = img_data[:,:,[1,3,5]]
    x_img = img_data[loop_ct].transpose(2, 0, 1)
    x_img = torch.Tensor(np.expand_dims(x_img, 0))
    x_img = normalization(x_img, (0, 255), minmax)
    x_joint = torch.Tensor(np.expand_dims(arm_joint_data[loop_ct], 0))
    x_joint = normalization(x_joint, arm_joint_bounds, minmax)
    x_state= torch.Tensor(np.expand_dims(hand_state_data[loop_ct], 0))

    
    if not (loop_ct == nloop-1):
        t_img = img_data[loop_ct+1].transpose(2, 0, 1)
        t_img = torch.Tensor(np.expand_dims(t_img, 0))
        t_img = normalization(t_img, (0, 255), minmax)
        t_joint = torch.Tensor(np.expand_dims(arm_joint_data[loop_ct], 0))
        t_joint = normalization(t_joint, arm_joint_bounds, minmax)
        t_state= torch.Tensor(np.expand_dims(hand_state_data[loop_ct], 0))

    # predict rnn
    y_img, y_joint, y_state, state = model(x_img, x_joint, x_state, state)
    
    # yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
    # yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))
    
    img_loss = nn.MSELoss()(y_img, t_img) * loss_weights[0]
    joint_loss = nn.MSELoss()(y_joint, t_joint) * loss_weights[1]
    state_loss = nn.MSELoss()(y_state, t_state) * loss_weights[1]
    loss = img_loss + joint_loss + state_loss
    loss_list.append(tensor2numpy(loss))

    # denormalization
    pred_img = tensor2numpy(y_img[0])
    pred_img = deprocess_img(pred_img, params["vmin"], params["vmax"])
    pred_img = pred_img.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, arm_joint_bounds)
    pred_state = tensor2numpy(y_state[0])
    
    # append data
    img_list.append(pred_img)
    joint_list.append(pred_joint)
    state_list.append(pred_state)

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_img_data = np.array(img_list)
pred_left_img_data = pred_img_data[:,:,:,[0,2,4]]
pred_right_img_data = pred_img_data[:,:,:,[1,3,5]]
pred_arm_joint_data = np.array(joint_list)
pred_hand_state_data = np.array(state_list)
loss_list = np.array(loss_list)

# plot img
# left_img, pred_left_img, left_arm_joint, left_hand_state
# right_img, pred_right_img, right_arm_joint, right_hand_state

T = len(img_data)
fig, ax = plt.subplots(2, 4, figsize=(16, 8), dpi=60)
def anim_update(i):
    for j in range(4):
        for k in range(2):
            ax[k][j].cla()
        
    # plot left camera image
    ax[0][0].imshow(left_img_data[i, :, :, ::-1])
    ax[0][0].axis("off")
    ax[0][0].set_title("left camera image")
    
    ax[1][0].imshow(right_img_data[i, :, :, ::-1])
    ax[1][0].axis("off")
    ax[1][0].set_title("right camera image")

    # plot predicted image
    ax[0][1].imshow(pred_left_img_data[i, :, :, ::-1])
    ax[0][1].axis("off")
    ax[0][1].set_title("left pred image")
    
    ax[1][1].imshow(pred_right_img_data[i, :, :, ::-1])
    ax[1][1].axis("off")
    ax[1][1].set_title("right pred image")

    # plot left joint angle
    ax[0][2].set_ylim(-2.0, 2.5)
    ax[0][2].set_xlim(0, T)
    ax[0][2].plot(arm_joint_data[1:, :7], linestyle="dashed", c="k")
    # left arm has 7 joints, not 8
    for joint_idx in range(14)[:7]:
        ax[0][2].plot(np.arange(i + 1), pred_arm_joint_data[: i + 1, joint_idx])
    ax[0][2].set_xlabel("Step")
    ax[0][2].set_title("left arm joint angles")

    # plot right joint angle
    ax[1][2].set_ylim(-2.0, 2.5)
    ax[1][2].set_xlim(0, T)
    ax[1][2].plot(arm_joint_data[1:, 7:], linestyle="dashed", c="k")
    # right arm has 7 joints, not 8
    for joint_idx in range(14)[7:]:
        ax[1][2].plot(np.arange(i + 1), pred_arm_joint_data[: i + 1, joint_idx])
    ax[1][2].set_xlabel("Step")
    ax[1][2].set_title("right arm joint angles")
    
    # plot left state
    ax[0][3].set_ylim(-1.0, 2.0)
    ax[0][3].set_xlim(0, T)
    ax[0][3].plot(hand_state_data[1:, :2], linestyle="dashed", c="k")
    # state has 7
    for state_idx in range(4)[:2]:
        ax[0][3].plot(np.arange(i + 1), pred_hand_state_data[: i + 1, state_idx])
    ax[0][3].set_xlabel("Step")
    ax[0][3].set_title("left hand state")
    
    # plot right state
    ax[1][3].set_ylim(-1.0, 2.0)
    ax[1][3].set_xlim(0, T)
    ax[1][3].plot(hand_state_data[1:, 2:], linestyle="dashed", c="k")
    # state has 7
    for state_idx in range(4)[2:]:
        ax[1][3].plot(np.arange(i + 1), pred_hand_state_data[: i + 1, state_idx])
    ax[1][3].set_xlabel("Step")
    ax[1][3].set_title("right hand state")

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save(f'./output/test_idx_{idx}/MTRNN_{params["tag"]}.gif')

fig = plt.figure()
plt.plot(range(len(loss_list)), loss_list, linestyle='solid', c='k', label="online")
plt.plot(range(len(loss_list)), loss_list.mean()*np.ones_like(loss_list), linestyle='dashed', c='r', label="average")
plt.grid()
# plt.ylim(0, 0.05)
plt.xlabel("step")
plt.ylabel("tesst_loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_idx_{idx}/{params["tag"]}/test_loss_trend.png')
plt.clf()
plt.close()

result = {
    "context_size": {"cf": params["fast_context"], "cs": params["slow_context"]},
    "tau": {"cf": params["fast_tau"], "cs": params["slow_tau"]},
    "loss": {"test": float(loss_list.mean())}
}
with open(f'./output/test_idx_{idx}/{params["tag"]}/result.json', 'w') as f:
    json.dump(result, f, indent=2)





# loss_weights = [params["img_loss"], params["joint_loss"], params["pt_loss"]]
# img_loss = ((images - pred_image)**2).mean() * loss_weights[0]
# joint_loss = ((joints - pred_joint)**2).mean() * loss_weights[1]
# loss = img_loss + joint_loss

# import ipdb; ipdb.set_trace()