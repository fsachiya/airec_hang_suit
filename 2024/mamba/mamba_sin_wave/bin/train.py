import os
import glob
import sys
import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import argparse
from collections import OrderedDict
from tqdm import tqdm
import datetime
import json

# from mamba_ssm import Mamba
sys.path.append("/home/fujita/work/mamba")
from mamba_seq import Mamba, MambaConfig

import ipdb
from einops import rearrange

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

try:
    # from libs.model import CNNMTRNN
    from libs.utils import DiscreteOptim
    from libs.utils import get_unique_list
    from libs.utils import deprocess_img, get_batch, tensor2numpy
    from libs.utils import SingleDataset
    from libs.trainer import fullBPTTtrainer
    from libs.utils import sinwave
except:
    sys.path.append("libs/")
    # from model import CNNMTRNN
    from utils import DiscreteOptim
    from utils import get_unique_list
    from utils import deprocess_img, get_batch, tensor2numpy
    from utils import SingleDataset
    from trainer import fullBPTTtrainer
    from utils import sinwave

parser = argparse.ArgumentParser(
    description="mtrnn square wave prediction train"
)
# parser.add_argument('sin_type', choices=['normal', 'ampl', 'wavelen', 'both'])
parser.add_argument('--state_path', default=None)
parser.add_argument('--model', type=str, default="mtrnn")
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="radam")
parser.add_argument('--log_dir', default="log/")
parser.add_argument('--total_step', type=int, default=200)
parser.add_argument('--vmin', type=float, default=0.0)
parser.add_argument('--vmax', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--plt_show_flag', action='store_true')
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

args = check_args(args)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# load dataset
minmax = [args.vmin, args.vmax]
batch_size = 5

train_sinwave = np.array(sinwave.train_list)
train_x = train_sinwave[:,:-1]
train_y = train_sinwave[:,1:]
train_data = SingleDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True
)


test_sinwave = np.array(sinwave.test_list)
test_x = test_sinwave[:,:-1]
test_y = test_sinwave[:,1:]
test_data = SingleDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True
)

# parent_dir = "/home/fujita/job/2024/sin_wave_prediction/mtrnn_ver"
try: 
    os.makedirs(f'./sample/{args.tag}/train/')   # {parent_dir}
    os.makedirs(f'./sample/{args.tag}/test/')    # {parent_dir}
except FileExistsError as e:
    pass

fig = plt.figure()
for i in range(train_sinwave.shape[0]):
    plt.plot(train_sinwave[i], linestyle='dashed', label="True value")
plt.grid()
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./sample/{args.tag}/train/sin_wave.png')   # {parent_dir}
plt.clf()
plt.close()

fig = plt.figure()
for i in range(test_sinwave.shape[0]):
    plt.plot(test_sinwave[i], linestyle='dashed', label="True value")
plt.grid()
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./sample/{args.tag}/test/sin_wave.png')    # {parent_dir}
plt.clf()
plt.close()



batch, length, dim = batch_size, train_x.shape[1], 1
# model = Mamba(
#     d_model=dim,  # モデルの次元
#     d_state=16,   # SSMの状態の拡張係数
#     d_conv=4,     # ローカルな畳み込みの幅
#     expand=2     # ブロックの拡張係数
# )   #.to("cuda")
config = MambaConfig(d_model=dim, 
                     n_layers=1,
                     d_state=16,
                     d_conv=4,
                     expand_factor=2)
model = Mamba(config)


# context_size = {"cf": 100, "cs": 50}
# tau = {"cf": 4.0, "cs": 20.0}

# model = CNNMTRNN(context_size, tau)
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: grad")
#     else:
#         print(f"{name}: no_grad")


# set optimizer
##########################
# params_to_optimize_radam = [] 
# params_to_optimize_discrete = []

# tau_pb_params = []
# others_params = []

# for name, param in model.named_parameters():
#     if name == 'h2h.fast_tau_pb' or name == 'h2h.slow_tau_pb':
#         tau_pb_params.append(param)
#     else:
#         others_params.append(param)

# optimizer_tau_pb = optim.RAdam(tau_pb_params, lr=args.lr)
# optimizer_others = optim.RAdam(others_params, lr=args.lr)
# optimizers = [optimizer_tau_pb, optimizer_others]
# ##########################


# load trainer/tester class
if args.optimizer.casefold() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "radam":
    optimizer = optim.RAdam(model.parameters(), lr=args.lr)
elif args.optimizer.casefold() == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
else:
    assert False, "Unknown optimizer name {}. please set Adam or RAdam.".format(args.optimizer)
trainer = fullBPTTtrainer(model, optimizer, device=device)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


### training main
log_dir_path = set_logdir(f"./{args.log_dir}", args.tag)
save_name = os.path.join(log_dir_path, "MAMBA.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=100000)
train_loss_list, test_loss_list = [], []
ext_hs_list = [None,] * 1

# fast_tau_list = []

with tqdm(range(args.epoch)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        train_loss, ext_hs_list = trainer.process_epoch(train_loader, ext_hs_list)
        test_loss, _ = trainer.process_epoch(test_loader, ext_hs_list, training=False)
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)
        
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        # fast_tau_list.append(fast_tau)
        # early stop
        save_ckpt, _ = early_stop(test_loss)
        
        # scheduler.step(test_loss)

        if epoch == args.epoch-1:   # save_ckpt
            trainer.save(epoch, [train_loss,test_loss], save_name)

        trainer.save(epoch, [train_loss,test_loss], save_name)
        
        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss,
                                           test_loss=test_loss))
        
        fig = plt.figure()
        plt.plot(range(len(train_loss_list)), train_loss_list, linestyle='dashed', c='r', label="train loss")
        plt.plot(range(len(train_loss_list)), test_loss_list, linestyle='dashed', c='k', label="test loss")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
        plt.savefig(f'./log/{args.tag}/loss_trend.png')  # {parent_dir}
        plt.clf()
        plt.close()

fig = plt.figure()
plt.plot(range(args.epoch), train_loss_list, linestyle='dashed', c='r', label="train loss")
plt.plot(range(args.epoch), test_loss_list, linestyle='dashed', c='k', label="test loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./log/{args.tag}/loss_trend.png')  # {parent_dir}
plt.clf()
plt.close()

# fig = plt.figure()
# plt.plot(range(args.epoch), fast_tau_list, linestyle='dashed', c='k', label="fast tau")
# plt.ylim((fast_tau_range["min"], fast_tau_range["max"]))
# # plt.plot(range(args.epoch), slow_tau_list, linestyle='dashed', c='k', label="slow tau")
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
# plt.savefig(f'{parent_dir}/log/{args.tag}/tau_trend.png')
# plt.clf()
# plt.close()