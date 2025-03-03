#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#
import sys
sys.path.append('./libs/')
from model import HSARNN

from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img, cos_interpolation
import matplotlib.animation as anim
import matplotlib.pylab as plt
import numpy as np
import argparse
import torch
import glob
import os


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
parser.add_argument("filename", type=str, default=None)
parser.add_argument("idx", type=int, default=0)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
params = restore_args(os.path.join(dir_name, "args.json"))
idx = args.idx

# load dataset
minmax = [params["vmin"], params["vmax"]]
bounds = np.load('../data/task2/param/ja_param.npy')
arm_angles = np.load('../data/task2/test/arm_angles.npy')
norm_arm_angles = normalization(
    arm_angles - bounds[0], bounds[1:], minmax)[:, :, 7:]
right_hand = smoothing(np.load('../data/task2/test/right_hand.npy')[:, :, 4:])
norm_right_hand = normalization(right_hand, (0.0, 1.0), minmax)
joints = np.concatenate((norm_arm_angles, norm_right_hand), axis=-1)
_img = np.load('../data/task2/test/images_right_adjust.npy')
test_img_raw = resize_img(_img, (params['im_size'], params['im_size']))
images = test_img_raw[idx]
joints = joints[idx]
joint_dim = joints.shape[-1]
print("joints: ", joints.min(), joints.max())
print("images: ", images.min(), images.max())


# define model
model = HSARNN(
    rnn_dim=params['rnn_dim'],
    union_dim=params['union_dim'],
    joint_dim=joint_dim,
    k_dim=params['k_dim'],
    heatmap_size=params['heatmap_size'],
    temperature=params['temperature'],
    im_size=[64, 64],
)

if params["compile"]:
    model = torch.compile(model)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
img_size = 64
image_list, joint_list = [], []
ect_pts_list, dec_pts_list = [], []
state = None
nloop = len(images)
for loop_ct in range(nloop):
    # load data and normalization
    img_t = images[loop_ct].transpose(2, 0, 1)
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    img_t = normalization(img_t, (0, 255), minmax)
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))

    # predict rnn
    y_image, y_joint, ect_pts, dec_pts, state = model(img_t, joint_t, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    # pred_joint = normalization(pred_joint, minmax, bounds)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    ect_pts_list.append(tensor2numpy(ect_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# split key points
ect_pts = np.array(ect_pts_list)
dec_pts = np.array(dec_pts_list)
ect_pts = ect_pts.reshape(-1, params["k_dim"], 2) * img_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
enc_pts = np.clip(ect_pts, 0, img_size)
dec_pts = np.clip(dec_pts, 0, img_size)


# plot images
T = len(images)
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)


def anim_update(i):
    for j in range(3):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1],
                   "bo", markersize=6)  # encoder
        ax[0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot predicted image
    ax[1].imshow(pred_image[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    ax[2].set_ylim(0.0, 1.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    # om has 5 joints, not 8
    for joint_idx in range(joint_dim):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")


ani = anim.FuncAnimation(
    fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/SARNN_{}_{}.gif".format(params["tag"], idx))

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")
