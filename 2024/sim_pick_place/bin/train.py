#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from collections import OrderedDict
import ipdb
import GPUtil
import time
import json

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

# eipl
# from eipl.model import SARNN
# from eipl.data import MultimodalDataset, SampleDownloader
sys.path.append("/home/fujita/work/eipl")
from eipl.utils import EarlyStopping, check_args, set_logdir, resize_img, normalization

# local
sys.path.append("../")
from util import ImgStateDataset, Visualize, Selector

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model_name", type=str, default="hrnn")
# data
parser.add_argument("--data_type", type=str, default="hang_right")
parser.add_argument("--offset", type=int, default=1)   
parser.add_argument("--batch_size", type=int, default=5)

parser.add_argument("--symbol", type=str, default="HRNN")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--scheduler", type=bool, default=False)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--optim", type=str, default="adam")

parser.add_argument("--hid_dim", type=int, default=64)
parser.add_argument("--key_dim", type=int, default=8)
parser.add_argument("--img_feat_dim", type=int, default=8)

parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--state_loss", type=float, default=1.0)
parser.add_argument("--key_loss", type=float, default=0.1)

parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)

parser.add_argument("--stdev", type=float, default=0.1)
parser.add_argument("--log_dir", default="log/")

parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)

parser.add_argument("--device", type=int, default=0)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# calculate the noise level (variance) from the normalized range
stdev = args.stdev * (args.vmax - args.vmin)
minmax = [args.vmin, args.vmax]

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

train_imgs = np.load(f"../data/train/images.npy")
test_imgs = np.load(f"../data/test/images.npy")

train_states = np.load(f"../data/train/joints.npy")
test_states = np.load(f"../data/test/joints.npy")


seq_len = train_imgs.shape[1]
train_num = train_imgs.shape[0]
test_num = test_imgs.shape[0]

img_bounds = [np.min(train_imgs), np.max(train_imgs)]
_train_imgs = normalization(train_imgs, img_bounds, minmax)
_train_imgs = resize_img(_train_imgs, (64, 64))
_train_imgs = _train_imgs.transpose(0, 1, 4, 2, 3)
_test_imgs = normalization(test_imgs, img_bounds, minmax)
_test_imgs = resize_img(_test_imgs, (64, 64))
_test_imgs = _test_imgs.transpose(0, 1, 4, 2, 3)

state_bounds = [np.min(train_states), np.max(train_states)]
_train_states = normalization(train_states, state_bounds, minmax)    
_test_states = normalization(test_states, state_bounds, minmax)    


train_dataset = ImgStateDataset(
    _train_imgs, 
    _train_states, 
    device=device, 
    stdev=stdev)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    drop_last=False
)

test_dataset = ImgStateDataset(
    _test_imgs, 
    _test_states, 
    device=device, 
    stdev=None)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=False,
)

selector = Selector(args)
model = selector.select_model()

# torch.compile makes PyTorch code run faster
if args.compile:
    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)

# set optimizer
if args.optim == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-07)
else:
    print(f"{args.optim} is invalid model")
    exit()

# set scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

loss_weight_dict = selector.select_loss_weight()

trainer = selector.select_trainer(optimizer)

### training main
log_dir_path = set_logdir("../" + args.log_dir, args.tag)
save_name = os.path.join(log_dir_path, f"{args.symbol}.pth")
writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
early_stop = EarlyStopping(patience=1000)
epochs = args.epochs

train_epoch_loss_list, valid_epoch_loss_list = [], []
train_epoch_time_list = []
train_epoch_gpu_usage_list, train_epoch_mem_usage_list, train_epoch_mem_used_list = [], [], []
train_start_time = time.time()
with tqdm(range(epochs)) as pbar_epoch:
    for epoch in pbar_epoch:
        # train and test
        start_epoch_time = time.time()
        train_loss = trainer.process_epoch(train_loader)
        fin_epoch_time = time.time()
        epoch_time = fin_epoch_time - start_epoch_time
        train_epoch_loss_list.append(train_loss.item())
        
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)
            valid_epoch_loss_list.append(train_loss.item())
        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/test_loss", test_loss, epoch)
        
        # early stop
        save_ckpt, _ = early_stop(test_loss)
        
        if args.scheduler:
            scheduler.step(test_loss)
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch + 1}, Validation Loss: {valid_loss:.4f}, Current Learning Rate: {current_lr:.6f}")
        
        if epoch % 10 == 0 or epoch == epochs-1:
            train_epoch_time_list.append(epoch_time)

        if epoch % 10 == 0 or epoch == epochs-1:
            gpu = GPUtil.getGPUs()[0]
            gpu_usage = gpu.load
            mem_usage = gpu.memoryUtil
            mem_used = gpu.memoryUsed
            train_epoch_gpu_usage_list.append(gpu_usage)
            train_epoch_mem_usage_list.append(mem_usage)
            train_epoch_mem_used_list.append(mem_used)
        
        curr_lr = optimizer.param_groups[0]['lr']
        
        if epoch%100==0 or epoch==epochs-1:    # save_ckpt or 
            save_name = os.path.join(log_dir_path, f"{args.symbol}_{epoch}.pth")
            trainer.save(epoch, [train_loss, test_loss], save_name)
        
        # print process bar
        pbar_epoch.set_postfix(OrderedDict(train=train_loss, loss=test_loss, lr=curr_lr))

train_fin_time = time.time()
train_total_time = train_fin_time - train_start_time
train_ave_epoch_time = train_total_time/epochs
print(f"train total time: {train_total_time} sec")
print(f"train average time: {train_ave_epoch_time} sec")
train_epoch_losses = np.stack(train_epoch_loss_list)
valid_epoch_losses = np.stack(valid_epoch_loss_list)
train_epoch_times = np.stack(train_epoch_time_list)
train_epoch_gpu_usages = np.stack(train_epoch_gpu_usage_list)
train_epoch_mem_usages = np.stack(train_epoch_mem_usage_list)
train_epoch_mem_usedes = np.stack(train_epoch_mem_used_list)
gpu = GPUtil.getGPUs()[0]

json_data = {
    "general": {
        "tag": args.tag, 
        "symbol": args.symbol
    },
    "gpu": {
        "name": gpu.name,
        "mem_total": gpu.memoryTotal,
    },
    "data": {
        "data_type": args.data_type,
        "seq_len": seq_len,
        "offset": args.offset,
        "batch_size": args.batch_size,
        "train_num": train_num,
        "test_num": test_num,
        "img_bounds": [float(x) for x in img_bounds],
        "state_bounds": [float(x) for x in state_bounds],
        "stdev": args.stdev,
        "vmin": args.vmin,
        "vmax": args.vmax
    },
    "model": {
        "model_name": args.model_name,
        "hid_dim": args.hid_dim,
    },
    "train": {
        "lr": args.lr,
        "epochs": epochs,
        "optimizer": "adam",
        "criterion": "mse",
        "total_time": train_total_time,
        "ave_epoch_time": train_ave_epoch_time,
        "valid_epoch_losses": valid_epoch_losses.tolist(),
        "epoch_times": train_epoch_times.tolist(),
        "gpu_usages": train_epoch_gpu_usages.tolist(),
        "mem_usages": train_epoch_mem_usages.tolist(),
        "mem_usedes": train_epoch_mem_usedes.tolist(),
    },
    "loss": {
        "img_loss": args.img_loss,
        "state_loss": args.state_loss,
        "key_loss": args.key_loss
    },
    "inf": {
        "compile": args.compile,
        # "inf_method": inf_method,
        # "total_time": inf_total_time,
        # "ave_seq_time": inf_ave_seq_time,
        # "seq_times": inf_seq_times.tolist(),
        # "seq_losses": test_seq_losses.tolist(),
        # "seq_gpu_usages": inf_seq_gpu_usages.tolist(),
        # "seq_mem_usages": inf_seq_mem_usages.tolist(),
        # "seq_mem_usedes": inf_seq_mem_usedes.tolist(),
    }
}

save_name = os.path.join(log_dir_path, "args.json")
with open(save_name, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
    
viz = Visualize(
    log_dir_path=log_dir_path,
    model_name=args.model_name,
    epochs=epochs
)

viz.viz_train_gpu(
    train_epoch_gpu_usages=train_epoch_gpu_usages,
    train_epoch_mem_usages=train_epoch_mem_usages,
)

viz.viz_train_loss(
    train_epoch_losses=train_epoch_losses,
    valid_epoch_losses=valid_epoch_losses,
    offset=10
)

viz.viz_train_time(
    train_epoch_times=train_epoch_times
)



