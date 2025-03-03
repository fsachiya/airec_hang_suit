#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import ipdb
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/fujita/work/eipl")
from eipl.data import  MultiEpochsDataLoader
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization
from eipl.utils import resize_img, cos_interpolation

try:
    from libs.model import StereoHSARNN
    from libs.utils import MultimodalDataset
    from libs.utils import moving_average
    from libs.trainer import fullBPTTtrainer
except:
    sys.path.append("./libs/")
    from model import StereoHSARNN
    from utils import MultimodalDataset
    from utils import moving_average
    from trainer import fullBPTTtrainer


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
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="hierachical_rnn")
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=5)

# parser.add_argument("--rnn_dim", type=int, default=50)
# parser.add_argument("--union_dim", type=int, default=20)
parser.add_argument("--srnn_hid_dim", type=int, default=50)
parser.add_argument("--urnn_hid_dim", type=int, default=20)
parser.add_argument("--k_dim", type=int, default=8)

parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--vec_loss", type=float, default=1.0)
# parser.add_argument("--press_loss", type=float, default=1.0)
# parser.add_argument("--joint_loss", type=float, default=1.0)

parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)

parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, default="adam")

parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.1)
parser.add_argument("--vmax", type=float, default=0.9)
parser.add_argument("--device", type=int, default=0)

# parser.add_argument("--im_size", type=int, default=64)
parser.add_argument("--img_size", type=int, default=64)
parser.add_argument("--n_worker", type=int, default=8)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
minmax = [args.vmin, args.vmax]

data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2"

# train data
# img
raw_left_img_data = np.load(f"{data_dir_path}/train/left_img.npy")
_left_img_data = resize_img(raw_left_img_data, (args.img_size, args.img_size))
_left_img_data = normalization(_left_img_data.astype(np.float32), (0.0, 255.0), minmax)
left_img_data = np.transpose(_left_img_data, (0, 1, 4, 2, 3))

raw_right_img_data = np.load(f"{data_dir_path}/train/right_img.npy")
_right_img_data = resize_img(raw_right_img_data, (args.img_size, args.img_size))
_right_img_data = normalization(_right_img_data.astype(np.float32), (0.0, 255.0), minmax)
right_img_data = np.transpose(_right_img_data, (0, 1, 4, 2, 3))

# approximate parallax
pix_mse_list = []
for i  in range(int(args.img_size/2)):
    pix_loss = (left_img_data[:,:,:,:,i:] - right_img_data[:,:,:,:,:args.img_size-i])**2
    pix_mse = pix_loss.mean()
    pix_mse_list.append(pix_mse)
approx_parallax = pix_mse_list.index(min(pix_mse_list))
approx_parallax_ratio = approx_parallax / args.img_size


# joint bounds
arm_joint_bounds = np.load(f"{data_dir_path}/param/arm_joint_bounds.npy")
thresh = 0.02
for i in range(arm_joint_bounds.shape[1]):
    if arm_joint_bounds[1,i] - arm_joint_bounds[0,i] < thresh:
        arm_joint_bounds[0,i] = arm_joint_bounds[0].min()
        arm_joint_bounds[1,i] = arm_joint_bounds[1].max()

# # pressure bounds
# press_bounds = np.load(f"{data_dir_path}/param/pressure_bounds.npy")
# thresh = 100
# for i in range(press_bounds.shape[1]):
#     if press_bounds[1,i] - press_bounds[0,i] < thresh:
#         press_bounds[0,i] = press_bounds[0].min()
#         press_bounds[1,i] = press_bounds[1].max()


# delta = 0.05
# _minmax = [minmax[0]+delta, minmax[1]-delta]
# joint
raw_arm_joint_data = np.load(f"{data_dir_path}/train/joint_state.npy")
_arm_joint_data = normalization(raw_arm_joint_data, arm_joint_bounds, minmax)   # _minmax
arm_joint_data = _arm_joint_data[:,:,7:]

# cmd
raw_hand_cmd_data = np.load(f"{data_dir_path}/train/hand_cmd.npy")
_hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax) # _minmax

# vector
vec_data = np.concatenate((arm_joint_data, hand_cmd_data), axis=-1)
# for i in range(vec_data.shape[0]):
#     for j in range(vec_data.shape[2]):
#         # vec_data[i,:,j] += np.linspace(-delta, delta, vec_data.shape[1])
#         x = np.arange(vec_data.shape[1])
#         vec_data[i,:,j] += (delta * np.sin(2.0*np.pi/vec_data.shape[1] * x))        

# plt.figure(); plt.plot(vec_data[1]); plt.savefig("./fig/samplevec_data.png")

# # pressure
# raw_press_data = np.load(f"{data_dir_path}/train/pressure.npy")
# _press_data = normalization(raw_press_data, press_bounds, minmax)
# _press_data = _press_data[:,:,19+3:19+6+1]

# press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)

# for i in range(len(press_data)):
#     plt.figure()
#     plt.plot(press_data[i])
#     plt.savefig(f"./fig/sample_smooth_press_{i}.png")
#     plt.close()
# 3,4,5,6

train_dataset = MultimodalDataset(left_img_data, 
                                  right_img_data,
                                  vec_data, 
                                #   press_data,
                                  device=device, stdev=stdev)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory=False)

# test data
# img
raw_left_img_data = np.load(f"{data_dir_path}/test/left_img.npy")
_left_img_data = resize_img(raw_left_img_data, (args.img_size, args.img_size))
_left_img_data = normalization(_left_img_data.astype(np.float32), (0.0, 255.0), minmax)
left_img_data = np.transpose(_left_img_data, (0, 1, 4, 2, 3))

raw_right_img_data = np.load(f"{data_dir_path}/test/right_img.npy")
_right_img_data = resize_img(raw_right_img_data, (args.img_size, args.img_size))
_right_img_data = normalization(_right_img_data.astype(np.float32), (0.0, 255.0), minmax)
right_img_data = np.transpose(_right_img_data, (0, 1, 4, 2, 3))

# joint
raw_arm_joint_data = np.load(f"{data_dir_path}/test/joint_state.npy")
_arm_joint_data = normalization(raw_arm_joint_data, arm_joint_bounds, minmax)   # _minmax
arm_joint_data = _arm_joint_data[:,:,7:]

# cmd
raw_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
_hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax) # _minmax

# vector
vec_data = np.concatenate((arm_joint_data, hand_cmd_data), axis=-1)
# for i in range(vec_data.shape[0]):
#     for j in range(vec_data.shape[2]):
#         # vec_data[i,:,j] += np.linspace(-delta, delta, vec_data.shape[1])
#         x = np.arange(vec_data.shape[1])
#         vec_data[i,:,j] += (delta * np.sin(2.0*np.pi/vec_data.shape[1] * x))    

# # pressure
# raw_press_data = np.load(f"{data_dir_path}/test/pressure.npy")
# _press_data = normalization(raw_press_data, press_bounds, minmax)
# _press_data = _press_data[:,:,19+3:19+6+1]

# press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)

# for i in range(len(press_data)):
#     plt.figure()
#     plt.plot(press_data[i])
#     plt.savefig(f"./fig/sample_smooth_press_{i}.png")
#     plt.close()
# 3,4,5,6

test_dataset = MultimodalDataset(   left_img_data, 
                                    right_img_data,
                                    vec_data, 
                                    # press_data,
                                    device=device, stdev=stdev)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           pin_memory=False)

# joint_dim = vec_data.shape[-1]
vec_dim = vec_data.shape[-1]
# press_dim = press_data.shape[-1]

# define model
model = StereoHSARNN(
    # rnn_dim=args.rnn_dim,
    # union_dim=args.union_dim,
    srnn_hid_dim = args.srnn_hid_dim,
    urnn_hid_dim = args.urnn_hid_dim,
    k_dim=args.k_dim,
    vec_dim=vec_dim,
    # press_dim=press_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
    img_size=[args.img_size, args.img_size]
    )

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-07)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(
        args.optimizer
    )

# load trainer/tester class
loss_w_dic = {"i": args.img_loss,
              "k": args.pt_loss, 
              "v": args.vec_loss,
            #   "p": args.press_loss
              }
trainer = fullBPTTtrainer(model, 
                          optimizer, 
                          loss_w_dic=loss_w_dic, 
                          approx_parallax_ratio=approx_parallax_ratio,
                          device=device)
# scheduler = ReduceLROnPlateau(optimizer, 
#                               mode='min', 
#                               factor=0.5, 
#                               patience=10, 
#                               verbose=True)

# training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "HierarchicalRNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

train_loss_list, test_loss_list = [], []
with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # early stop
        save_ckpt, _ = early_stop(test_loss)
        
        # scheduler.step(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))
        pbar_epoch.update()

plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, label="train")
plt.plot(range(len(test_loss_list)), test_loss_list, label="test")
plt.grid()
plt.legend()
plt.savefig(f'{log_dir_path}/loss_trend.png')
plt.close()

result = {
    "hid_size": {"srnn": args.srnn_hid_dim, "urnn": args.urnn_hid_dim},
    "loss": {"train": train_loss_list[-1], "test": test_loss_list[-1]}
}
with open(f'{log_dir_path}/result.json', 'w') as f:
    json.dump(result, f, indent=2)