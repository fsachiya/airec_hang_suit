#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

try:
    from libs.model import CNNMTRNN
    from libs.utils import MultimodalDataset
    from libs.trainer import fullBPTTtrainer
except:
    sys.path.append("./libs/")
    from model import CNNMTRNN
    from utils import MultimodalDataset
    from trainer import fullBPTTtrainer


# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="mtrnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=5)
# parser.add_argument("--rec_dim", type=int, default=50)
# parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--fast_context", type=int, default=100)
parser.add_argument("--slow_context", type=int, default=50)
parser.add_argument("--fast_tau", type=float, default=4.0)
parser.add_argument("--slow_tau", type=float, default=20.0)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--optimizer", type=str, default="radam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
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

# load train data
minmax = [args.vmin, args.vmax]
_joints = np.load("../data/train/joints.npy")
joint_bounds = np.load("../data/joint_bounds.npy")
joints = normalization(_joints, joint_bounds, (args.vmin, args.vmax))
_images = np.load("../data/train/images.npy")
_images = np.transpose(_images, (0,1,4,2,3))
images = normalization(_images, (0,255), (args.vmin, args.vmax))
train_dataset = MultimodalDataset(images, joints, stdev=stdev, training=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)
print("---train---")
print("images: ", images.shape, images.min(), images.max())
print("joints: ", joints.shape, joints.min(), joints.max())


# load test data
_joints = np.load("../data/test/joints.npy")
joints = normalization(_joints, joint_bounds, (args.vmin, args.vmax))
_images = np.load("../data/test/images.npy")
_images = np.transpose(_images, (0,1,4,2,3))
images = normalization(_images, (0,255), (args.vmin, args.vmax))
test_dataset = MultimodalDataset(images, joints, stdev=0.0, training=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)
print("---test---")
print("images: ", images.shape, images.min(), images.max())
print("joints: ", joints.shape, joints.min(), joints.max())

context_size = {"cf": args.fast_context, "cs": args.slow_context}
tau = {"cf": args.fast_tau, "cs": args.slow_tau}

model = CNNMTRNN(context_size, tau)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(args.optimizer)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.pt_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)


### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "MTRNN.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)
train_loss_list, test_loss_list = [], []

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss = trainer.process_epoch(train_loader)
        test_loss = trainer.process_epoch(test_loader, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        # early stop
        save_ckpt, _ = early_stop(test_loss)

        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict( train_loss=train_loss,
                                            test_loss=test_loss))

fig = plt.figure()
plt.plot(range(args.epoch), train_loss_list, linestyle='dashed', c='r', label="train loss")
plt.plot(range(args.epoch), test_loss_list, linestyle='dashed', c='k', label="test loss")
plt.grid()
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'{log_dir_path}/loss_trend.png')
plt.clf()
plt.close()

result = {
    "context_size": {"cf": args.fast_context, "cs": args.slow_context},
    "tau": {"cf": args.fast_tau, "cs": args.slow_tau},
    "loss": {"train": train_loss_list[-1], "test": test_loss_list[-1]}
}
with open(f'{log_dir_path}/result.json', 'w') as f:
    json.dump(result, f, indent=2)
