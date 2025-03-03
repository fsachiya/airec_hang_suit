#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import ipdb
import os
import sys
import numpy as np
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from eipl.data import  MultiEpochsDataLoader
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization
from eipl.utils import resize_img, cos_interpolation

try:
    from libs.fullBPTT import fullBPTTtrainer
    from libs.model import HSARNN
    from libs.dataset import MultimodalDataset
except:
    sys.path.append("./libs/")
    from fullBPTT import fullBPTTtrainer
    from model import HSARNN
    from dataset import MultimodalDataset


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
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--rnn_dim", type=int, default=50)
parser.add_argument("--union_dim", type=int, default=20)
parser.add_argument("--k_dim", type=int, default=8)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--im_size", type=int, default=64)
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
bounds = np.load('../data/task2/param/ja_param.npy')
arm_angles = np.load('../data/task2/train/arm_angles.npy')
norm_arm_angles = normalization(arm_angles - bounds[0], bounds[1:], minmax)[:, :, 7:]
right_hand = smoothing(np.load('../data/task2/train/right_hand.npy')[:, :, 4:])
norm_right_hand = normalization(right_hand, (0.0, 1.0), minmax)
joints = np.concatenate((norm_arm_angles, norm_right_hand), axis=-1)
_img = np.load('../data/task2/train/images_right.npy')
train_img_raw = resize_img(_img, (args.im_size, args.im_size))
images = normalization(train_img_raw.astype(np.float32), (0.0, 255.0), (0.0, 1.0))
images = np.transpose(images, (0, 1, 4, 2, 3))
train_dataset = MultimodalDataset(images, joints, device=device, stdev=stdev)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=False)

# test data
arm_angles = np.load('../data/task2/test/arm_angles.npy')
norm_arm_angles = normalization(arm_angles - bounds[0], bounds[1:], minmax)[:, :, 7:]
right_hand = smoothing(np.load('../data/task2/test/right_hand.npy')[:, :, 4:])
norm_right_hand = normalization(right_hand, (0.0, 1.0), minmax)
joints = np.concatenate((norm_arm_angles, norm_right_hand), axis=-1)
_img = np.load('../data/task2/test/images_right.npy')
test_img_raw = resize_img(_img, (args.im_size, args.im_size))
images = normalization(test_img_raw.astype(np.float32), (0.0, 255.0), (0.0, 1.0))
images = np.transpose(images, (0, 1, 4, 2, 3))
test_dataset = MultimodalDataset(images, joints, device=device, stdev=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=False)
joint_dim = joints.shape[-1]

# define model
model = HSARNN(
    rnn_dim=args.rnn_dim,
    union_dim=args.union_dim,
    joint_dim=joint_dim,
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
    im_size=[64, 64],
)

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(
        args.optimizer
    )

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)

# training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "SARNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss, test_loss=test_loss))
        pbar_epoch.update()
