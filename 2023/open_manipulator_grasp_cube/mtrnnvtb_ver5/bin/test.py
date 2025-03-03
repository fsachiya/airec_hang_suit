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
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

try:
    from libs.model import CNNMTRNNVTB
except:
    sys.path.append("./libs/")
    from model import CNNMTRNNVTB


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
    os.makedirs(f'./output/test_pos_{idx}/{params["tag"]}/')
except:
    pass

# load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bounds = np.load("../data/joint_bounds.npy")
joints = np.load("../data/test/joints.npy")[idx]
step0_joint = np.expand_dims(joints[0], 0)
joints_shape = list(joints.shape)
joints_shape[0] += params["expand_step"]
expand_joints = np.zeros(joints_shape)
expand_joints[:params["expand_step"]] = step0_joint
expand_joints[params["expand_step"]:] = joints

images = np.load("../data/test/images.npy")[idx]
step0_images = np.expand_dims(images[0], 0)
images_shape = list(images.shape)
images_shape[0] += params["expand_step"]
expand_images = np.zeros(images_shape)
expand_images[:params["expand_step"]] = step0_images
expand_images[params["expand_step"]:] = images
print("images shape:{}, min={}, max={}".format(images.shape, images.min(), images.max()))
print("joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max()))

context_size = {"cf": params["fast_context"], "cs": params["slow_context"]}
fast_tau_range = {"min": params["fast_tau_min"], "max": params["fast_tau_max"]}
slow_tau = params["slow_tau"]
model = CNNMTRNNVTB(context_size, fast_tau_range, slow_tau)

# load weight
ckpt = torch.load(latest, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
img_size = 64
loss_weights = [params["img_loss"], params["joint_loss"], params["pt_loss"]]
expand_image_list, expand_joint_list = [], []
expand_loss_list, expand_img_loss_list, expand_joint_loss_list = [], [], []
expand_variable_fast_tau_list = []
state = None
nloop = len(expand_images)
for loop_ct in range(nloop):
    # load data and normalization
    x_img = expand_images[loop_ct].transpose(2, 0, 1)
    x_img = torch.Tensor(np.expand_dims(x_img, 0))
    x_img = normalization(x_img, (0, 255), minmax)
    x_joint = torch.Tensor(np.expand_dims(expand_joints[loop_ct], 0))
    x_joint = normalization(x_joint, joint_bounds, minmax)
    
    if not (loop_ct == nloop-1):
        t_img = expand_images[loop_ct+1].transpose(2, 0, 1)
        t_img = torch.Tensor(np.expand_dims(t_img, 0))
        t_img = normalization(t_img, (0, 255), minmax)
        t_joint = torch.Tensor(np.expand_dims(expand_joints[loop_ct], 0))
        t_joint = normalization(t_joint, joint_bounds, minmax)

    # predict rnn
    y_image, y_joint, state, variable_fast_tau = model(x_img, x_joint, state)
    
    img_loss = nn.MSELoss()(y_image, t_img) * loss_weights[0]
    joint_loss = nn.MSELoss()(y_joint, t_joint) * loss_weights[1]
    loss = img_loss + joint_loss
    expand_loss_list.append(tensor2numpy(loss))
    expand_img_loss_list.append(tensor2numpy(img_loss))
    expand_joint_loss_list.append(tensor2numpy(joint_loss))
    variable_fast_tau = torch.mean(variable_fast_tau, dim=1)
    expand_variable_fast_tau_list.append(tensor2numpy(variable_fast_tau[0]))

    # denormalization
    expand_pred_image = tensor2numpy(y_image[0])
    expand_pred_image = deprocess_img(expand_pred_image, params["vmin"], params["vmax"])
    expand_pred_image = expand_pred_image.transpose(1, 2, 0)
    expand_pred_joint = tensor2numpy(y_joint[0])
    expand_pred_joint = normalization(expand_pred_joint, minmax, joint_bounds)

    # append data
    expand_image_list.append(expand_pred_image)
    expand_joint_list.append(expand_pred_joint)

    print("loop_ct:{}, joint:{}".format(loop_ct, expand_joint_list))

pred_image = np.array(expand_image_list)[10:]
pred_joint = np.array(expand_joint_list)[10:]
loss_list = np.array(expand_loss_list)[10:]
img_loss_list = np.array(expand_img_loss_list)[10:]
joint_loss_list = np.array(expand_joint_loss_list)[10:]
variable_fast_tau_list = np.array(expand_variable_fast_tau_list)[10:]

# plot images
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)
def anim_update(i):
    for j in range(3):
        ax[j].cla()
        
    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    ax[0].axis("off")
    ax[0].set_title("camera image")

    # plot predicted image
    ax[1].imshow(pred_image[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    ax[2].set_ylim(-1.0, 2.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    # om has 5 joints, not 8
    for joint_idx in range(5):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")

ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save(f'./output/test_pos_{idx}/{params["tag"]}/MTRNNVTB_{params["tag"]}.gif')

fig = plt.figure()
plt.plot(range(len(loss_list)), loss_list, linestyle='solid', label="loss_online")
plt.plot(range(len(loss_list)), loss_list.mean()*np.ones_like(loss_list), linestyle='dashed', label="loss_average")
plt.plot(range(len(img_loss_list)), img_loss_list, linestyle='solid', label="img_loss_online")
plt.plot(range(len(img_loss_list)), img_loss_list.mean()*np.ones_like(loss_list), linestyle='dashed', label="img_loss_average")
plt.plot(range(len(joint_loss_list)), joint_loss_list, linestyle='solid', label="joint_loss_online")
plt.plot(range(len(joint_loss_list)), joint_loss_list.mean()*np.ones_like(loss_list), linestyle='dashed', label="joint_loss_average")
plt.grid()
# plt.ylim(0, 0.05)
plt.xlabel("step")
plt.ylabel("tesst_loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_pos_{idx}/{params["tag"]}/test_loss_trend.png')
plt.clf()
plt.close()

fig = plt.figure()
plt.plot(range(len(variable_fast_tau_list)), variable_fast_tau_list, linestyle='solid', c='k', label="online")
plt.plot(range(len(variable_fast_tau_list)), variable_fast_tau_list.mean()*np.ones_like(variable_fast_tau_list), linestyle='dashed', c='r', label="average")
plt.grid()
plt.ylim(0, 8)
plt.xlabel("step")
plt.ylabel("variable_fast_tau")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_pos_{idx}/{params["tag"]}/variable_fast_tau_trend.png')
plt.clf()
plt.close()

result = {
    "context_size": {"cf": params["fast_context"], "cs": params["slow_context"]},
    "variable_fast_tau": [float(variable_fast_tau) for variable_fast_tau in variable_fast_tau_list],  # float(variable_fast_tau_list[-1])
    "loss": {"test": float(loss_list.mean())}
}
with open(f'./output/test_pos_{idx}/{params["tag"]}/result.json', 'w') as f:
    json.dump(result, f, indent=2)

