#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
import cv2
import ipdb
import GPUtil
import time
import json
from natsort import natsorted

from sklearn.decomposition import PCA
import torch

# eipl
sys.path.append("/home/fujita/work/eipl")
from eipl.data import SampleDownloader, WeightDownloader
# from eipl.model import SARNN
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img, resize_img

# local
sys.path.append("../")
from util import ImgStateDataset, Visualize
from libs import fullBPTTtrainer4SARNN, fullBPTTtrainer4StackRNN, fullBPTTtrainer4HRNN, fullBPTTtrainer4FasterHRNN
from model import SARNN, StackRNN, HRNN, FasterHRNN
from inf import Inf4HRNN, Inf4StackRNN, Inf4FasterHRNN

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir_name", type=str, default="20241210_1427_23")
# parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--open_ratio", type=float, default=1.0)
parser.add_argument("--pretrained", action="store_true")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--show_inf", type=bool, default=True)
parser.add_argument("--plot_pca", type=bool, default=False)
args = parser.parse_args()

# check args
assert args.log_dir_name or args.pretrained, "Please set log_dir_name or pretrained"

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# restore parameters
# dir_name = os.path.split(args.log_dir_name)[0]
log_dir_path = f"../log/{args.log_dir_name}"
params = restore_args(os.path.join(log_dir_path, "args.json"))
# idx = int(args.idx)

test_imgs = np.load("../data/test/images.npy")
test_states = np.load("../data/test/joints.npy")
test_poses = np.load("../data/test/poses.npy")

seq_len = test_imgs.shape[1]
test_num = test_imgs.shape[0]

stdev = params["data"]["stdev"] * (params["data"]["vmax"] - params["data"]["vmin"])
minmax = [params["data"]["vmin"], params["data"]["vmax"]]
img_bounds = params["data"]["img_bounds"]
state_bounds = params["data"]["state_bounds"]

_test_imgs = normalization(test_imgs, img_bounds, minmax)
_test_imgs = resize_img(_test_imgs, (64, 64))
_test_imgs = _test_imgs.transpose(0, 1, 4, 2, 3)
_test_imgs = torch.tensor(_test_imgs).to(torch.float32)

_test_states = normalization(test_states, state_bounds, minmax)
_test_states = torch.tensor(_test_states).to(torch.float32)

model_name = params["model"]["model_name"]

# define model
if model_name in ["sarnn",]:
    model = SARNN(
        union_dim=params["model"]["hid_dim"],
        state_dim=8,
        key_dim=params["key_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
    ).to(device)
elif model_name in ["stackrnn"]:
    model = StackRNN(
        img_feat_dim=8,
        state_dim=8,
        union1_dim=params["model"]["hid_dim"],
        union2_dim=params["model"]["hid_dim"],
        union3_dim=params["model"]["hid_dim"],
    ).to(device)
elif model_name in ["hrnn"]:
    model = HRNN(
        img_feat_dim=8,
        state_dim=8,
        sensory_dim=params["model"]["hid_dim"],
        union1_dim=params["model"]["hid_dim"],
        union2_dim=params["model"]["hid_dim"],
    ).to(device)
elif model_name in ["fasterhrnn"]:
    model = FasterHRNN(
        img_feat_dim=8,
        state_dim=8,
        sensory_dim=int(params["model"]["hid_dim"]/2),
        union1_dim=params["model"]["hid_dim"],
        union2_dim=params["model"]["hid_dim"],
    ).to(device)
else:
    print(f"{model_name} is invalid model")
    exit()

# load weight
weight_pathes = natsorted(glob.glob(f"{log_dir_path}/*.pth"))
ckpt = torch.load(weight_pathes[-1], map_location=torch.device("cuda:0"), weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

if model_name in ["sarnn",]:
    loss_weight_dict = {"img": params["loss"]["img_loss"], "state": params["loss"]["state_loss"], "key": params["loss"]["key_loss"]}
elif model_name in ["stackrnn","hrnn","fasterhrnn",]:
    loss_weight_dict = {"img": params["loss"]["img_loss"], "state": params["loss"]["state_loss"]}

if model_name in ["hrnn"]:
    inf = Inf4HRNN(model, open_ratio=args.open_ratio)
elif model_name in ["stackrnn"]:
    inf = Inf4StackRNN(model, open_ratio=args.open_ratio)
elif model_name in ["fasterhrnn",]:
    inf = Inf4FasterHRNN(model, open_ratio=args.open_ratio)

x_imgs = _test_imgs[:,:-1].to(device)
y_imgs = _test_imgs[:,1:].to(device)
x_states = _test_states[:,:-1].to(device)
y_states = _test_states[:,1:].to(device)


y_imgs_hat, y_states_hat, hids_dict = inf.inf(x_imgs, x_states)

# pred plot
_y_imgs_hat = normalization(y_imgs_hat, minmax, img_bounds)
_y_imgs_hat = _y_imgs_hat.permute(0,1,3,4,2)
_y_imgs_hat = _y_imgs_hat[[0,2,4,6,8,10,12,14,16]]

_y_states_hat = normalization(y_states_hat, minmax, state_bounds)
_y_states_hat = _y_states_hat[[0,2,4,6,8,10,12,14,16]]

pred_imgs = np.uint8(_y_imgs_hat.detach().clone().cpu().numpy())
pred_states = _y_states_hat.detach().clone().cpu().numpy()

# input plot
_y_imgs = normalization(y_imgs, minmax, img_bounds)
_y_imgs = _y_imgs.permute(0,1,3,4,2)
_y_imgs = _y_imgs[[0,2,4,6,8,10,12,14,16]]

_y_states = normalization(y_states, minmax, state_bounds)
_y_states = _y_states[[0,2,4,6,8,10,12,14,16]]

y_imgs = np.uint8(_y_imgs.detach().clone().cpu().numpy())
y_states = _y_states.detach().clone().cpu().numpy()


save_log_dir = f"../output/{args.log_dir_name}"
try:
    os.makedirs(save_log_dir)
except FileExistsError as e:
    pass

# plot images
T = x_imgs.shape[1]
fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)

if args.show_inf:
    for idx in range(y_imgs.shape[0]):
        
        def anim_update(i):
            print(i)
            for j in range(3):
                ax[j].cla()

            # plot camera image
            ax[0].imshow(y_imgs[idx, i, :, :, ::-1])
            # for j in range(params["k_dim"]):
            #     ax[0].plot(enc_pts[i, j, 0], enc_pts[i, j, 1], "bo", markersize=6)  # encoder
            #     ax[0].plot(
            #         dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
            #     )  # decoder
            ax[0].axis("off")
            ax[0].set_title("Input image")

            # plot predicted image
            ax[1].imshow(pred_imgs[idx, i, :, :, ::-1])
            ax[1].axis("off")
            ax[1].set_title("Predicted image")

            # plot joint angle
            # ax[2].set_ylim(-1.0, 2.0)
            ax[2].set_xlim(0, T)
            ax[2].plot(y_states[idx], linestyle="dashed", c="k")
            for joint_idx in range(8):
                ax[2].plot(np.arange(i + 1), pred_states[idx, :i+1, joint_idx])
            ax[2].set_xlabel("Step")
            ax[2].set_title("Joint angles")
        
        ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
        ani.save(f"{save_log_dir}/{model_name}_{params['general']['tag']}_{idx}_{args.open_ratio}.gif")

        # ipdb.set_trace()

ipdb.set_trace()

if args.plot_pca:
    for key, hid_list in zip(hids_dict.keys(), hids_dict.values()):
        hids = torch.cat(hid_list).permute(1,0,2) 
        hids = tensor2numpy(hids)
        N, T, D = hids.shape
        hids = hids.reshape(-1, D)

        # plot pca
        loop_ct = float(360) / T
        pca_dim = 3
        pca = PCA(n_components=pca_dim).fit(hids)
        pca_val = pca.transform(hids)
        # Reshape the hids from [-1, pca_dim] to [N,T,pca_dim] to
        # visualize each state as a 3D scatter.
        pca_val = pca_val.reshape(N, T, pca_dim)

        fig = plt.figure(dpi=60)
        ax = fig.add_subplot(projection="3d")
        
        def anim_update(i):
            print(i)
            ax.cla()
            angle = int(loop_ct * i)
            ax.view_init(30, angle)

            c_list = ["C0",]    #  "C1", "C2", "C3", "C4"
            for n, color in enumerate(c_list):
                ax.scatter(
                    pca_val[n, 1:, 0], pca_val[n, 1:, 1], pca_val[n, 1:, 2], color=color, s=3.0
                )

            ax.scatter(pca_val[n, 0, 0], pca_val[n, 0, 1], pca_val[n, 0, 2], color="k", s=30.0)
            pca_ratio = pca.explained_variance_ratio_ * 100
            ax.set_xlabel("PC1 ({:.1f}%)".format(pca_ratio[0]))
            ax.set_ylabel("PC2 ({:.1f}%)".format(pca_ratio[1]))
            ax.set_zlabel("PC3 ({:.1f}%)".format(pca_ratio[2]))
        
        ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
        ani.save(f"{save_log_dir}/{model_name}_PCA_{params['general']['tag']}_{idx}.gif")




# If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
# ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]), writer="imagemagick")
# ani.save("./output/PCA_SARNN_{}.mp4".format(params["tag"]), writer="ffmpeg")
