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
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
sys.path.append("/home/fujita/work/eipl")
from eipl.utils import EarlyStopping, check_args, set_logdir
from eipl.utils import normalization, resize_img, cos_interpolation

try:
    from libs.model import CNNMTRNN
    from libs.utils import MultimodalDataset
    # from libs.utils import cos_interpolation
    from libs.trainer import fullBPTTtrainer
except:
    sys.path.append("./libs/")
    from model import CNNMTRNN
    from utils import MultimodalDataset
    # from utils import cos_interpolation
    from trainer import fullBPTTtrainer


# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="mtrnn")
parser.add_argument("--epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=3)

parser.add_argument("--fast_context", type=int, default=100)
parser.add_argument("--slow_context", type=int, default=50)

parser.add_argument("--fast_tau", type=float, default=4.0)
parser.add_argument("--slow_tau", type=float, default=20.0)

parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--state_loss", type=float, default=1.0)

parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=0.1)
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

data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task1"
# img
_img_data = np.load(f"{data_dir_path}/train/images_left.npy")
_img_data = resize_img(_img_data, (128, 128))
_img_data = np.transpose(_img_data, (0,1,4,2,3))
img_data = normalization(_img_data, (0,255), (args.vmin, args.vmax))

# joint
arm_joint_bounds = np.load(f"{data_dir_path}/param/arm_bounds.npy")
# arm_joint
_arm_joint_data = np.load(f"{data_dir_path}/train/arm_angles.npy")
arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds, (args.vmin, args.vmax))

# hand_states
_left_hand_state_data = np.load(f"{data_dir_path}/train/left_hand.npy")
_right_hand_state_data = np.load(f"{data_dir_path}/train/right_hand.npy")
_hand_state_data = np.concatenate([_left_hand_state_data,
                             _right_hand_state_data], axis=-1)
hand_state_data = np.apply_along_axis(cos_interpolation, 1, _hand_state_data, step=10)[:,:,[1,2,3,9,10]]

# plt.figure()
# for i in range(hand_state_data.shape[-1]):
#     if not np.all(hand_state_data[0,:,i] == 0):
#         plt.plot(hand_state_data[0,:,i], label=f"idx_{i}")
# plt.legend()
# plt.savefig("./fig/prosessed_hand_state_data.png")


train_dataset = MultimodalDataset(img_data, 
                                  arm_joint_data, 
                                  hand_state_data, 
                                  stdev=stdev, training=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)

print("---train---")
print("img_data: ", img_data.shape, img_data.min(), img_data.max())
print("arm_joint_data: ", arm_joint_data.shape, arm_joint_data.min(), arm_joint_data.max())
print("hand_state_data:", hand_state_data.shape, hand_state_data.min(), hand_state_data.max())



# load test data
# img
_img_data = np.load(f"{data_dir_path}/test/images_left.npy")
_img_data = resize_img(_img_data, (128, 128))
_img_data = np.transpose(_img_data, (0,1,4,2,3))
img_data = normalization(_img_data, (0,255), (args.vmin, args.vmax))

# joint
# arm_joint
_arm_joint_data = np.load(f"{data_dir_path}/test/arm_angles.npy")
arm_joint_data = normalization(_arm_joint_data, arm_joint_bounds, (args.vmin, args.vmax))

# hand_states
_left_hand_state_data = np.load(f"{data_dir_path}/test/left_hand.npy")
_right_hand_state_data = np.load(f"{data_dir_path}/test/right_hand.npy")
_hand_state_data = np.concatenate([_left_hand_state_data,
                                   _right_hand_state_data], axis=-1)
hand_state_data = np.apply_along_axis(cos_interpolation, 1, _hand_state_data, step=10)[:,:,[1,2,3,9,10]]


test_dataset = MultimodalDataset(img_data,
                                 arm_joint_data, 
                                 hand_state_data, 
                                 stdev=stdev, training=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)

print("---test---")
print("img_data: ", img_data.shape, img_data.min(), img_data.max())
print("arm_joint_data: ", arm_joint_data.shape, arm_joint_data.min(), arm_joint_data.max())
print("hand_state_data:", hand_state_data.shape, hand_state_data.min(), hand_state_data.max())


context_size = {"cf": args.fast_context, 
                "cs": args.slow_context}
tau = {"cf": args.fast_tau, 
       "cs": args.slow_tau}

model = CNNMTRNN(context_size, tau)

# set optimizer
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(args.optimizer)

# load trainer/tester class
loss_weights = [args.img_loss, args.joint_loss, args.state_loss]
trainer = fullBPTTtrainer(model, optimizer, loss_weights=loss_weights, device=device)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


### training main
log_dir_path = set_logdir("./" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, "MTRNN.pth")
# writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)
# train_loss_list, test_loss_list = [], []

train_loss_list, train_img_loss_list, train_joint_loss_list, train_state_loss_list = [], [], [], []
test_loss_list, test_img_loss_list, test_joint_loss_list, test_state_loss_list = [], [], [], []

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss_dir = trainer.process_epoch(train_loader)
        test_loss_dir = trainer.process_epoch(test_loader, training=False)
        # writer.add_scalar("Loss/train_loss", train_loss, epoch)
        # writer.add_scalar("Loss/test_loss", test_loss, epoch)

        # train_loss_list.append(train_loss)
        # test_loss_list.append(test_loss)
        train_loss_list.append(train_loss_dir["total_loss"])
        train_img_loss_list.append(train_loss_dir["total_img_loss"])
        train_joint_loss_list.append(train_loss_dir["total_joint_loss"])
        train_state_loss_list.append(train_loss_dir["total_state_loss"])
        
        test_loss_list.append(test_loss_dir["total_loss"])
        test_img_loss_list.append(test_loss_dir["total_img_loss"])
        test_joint_loss_list.append(test_loss_dir["total_joint_loss"])
        test_state_loss_list.append(test_loss_dir["total_state_loss"])

        # early stop
        save_ckpt, _ = early_stop(test_loss_dir["total_loss"])

        scheduler.step(test_loss_dir["total_loss"])
        
        if save_ckpt:
            trainer.save(epoch, [train_loss_dir["total_loss"], test_loss_dir["total_loss"]], save_name)

        # print process bar
        pbar_epoch.set_postfix(OrderedDict( train_loss=train_loss_dir["total_loss"],
                                            test_loss=test_loss_dir["total_loss"]))

# fig = plt.figure()
# plt.plot(range(args.epoch), train_loss_list, linestyle='dashed', c='r', label="train loss")
# plt.plot(range(args.epoch), test_loss_list, linestyle='dashed', c='k', label="test loss")
# plt.grid()
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
# plt.savefig(f'{log_dir_path}/loss_trend.png')
# plt.clf()
# plt.close()
fig, ax = plt.subplots(1, 3, figsize=(20, 8), dpi=60)
for i in range(3):
    ax[i].cla()

ax[0].plot(range(len(train_loss_list)), train_loss_list, label="train")
ax[0].plot(range(len(test_loss_list)), test_loss_list, label="test")
ax[0].grid()
ax[0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
ax[0].set_title("total loss")

ax[1].plot(range(len(train_img_loss_list)), train_img_loss_list, label="img")
ax[1].plot(range(len(train_joint_loss_list)), train_joint_loss_list, label="joint")
ax[1].plot(range(len(train_state_loss_list)), train_state_loss_list, label="state")
ax[1].grid()
ax[1].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
ax[1].set_title("train data loss")

ax[2].plot(range(len(test_img_loss_list)), test_img_loss_list, label="img")
ax[2].plot(range(len(test_joint_loss_list)), test_joint_loss_list, label="joint")
ax[2].plot(range(len(test_state_loss_list)), test_state_loss_list, label="state")
ax[2].grid()
ax[2].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
ax[2].set_title("test data loss")
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
