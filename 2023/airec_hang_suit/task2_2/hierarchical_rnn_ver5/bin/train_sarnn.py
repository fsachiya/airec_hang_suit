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
import gc
import cv2

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict

sys.path.append("/home/fujita/work/eipl")
from eipl.data import  MultiEpochsDataLoader
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization
from eipl.utils import resize_img, cos_interpolation

try:
    from libs.model import SARNN
    from libs.utils import MultimodalDataset
    from libs.utils import moving_average
    from libs.trainer import simple_fullBPTTtrainer
except:
    sys.path.append("./libs/")
    from model import SARNN
    from utils import MultimodalDataset
    from utils import moving_average
    from trainer import simple_fullBPTTtrainer


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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser(
        description="Learning spatial autoencoder with recurrent neural network"
    )
    parser.add_argument("--model", type=str, default="hierachical_rnn")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=4)
    
    parser.add_argument("--srnn_hid_dim", type=int, default=50)
    parser.add_argument("--urnn_hid_dim", type=int, default=20)
    parser.add_argument("--key_dim", type=int, default=16)

    parser.add_argument("--img_loss", type=float, default=0.1)
    parser.add_argument("--key_loss", type=float, default=1.0)
    parser.add_argument("--vec_loss", type=float, default=1.0)
    parser.add_argument("--press_loss", type=float, default=1.0)
    # parser.add_argument("--joint_loss", type=float, default=1.0)

    parser.add_argument("--heatmap_size", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1e-4)

    parser.add_argument("--stdev", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.005)   # 1e-3
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--log_dir", default="log/")
    parser.add_argument("--vmin", type=float, default=0.1)
    parser.add_argument("--vmax", type=float, default=0.9)
    parser.add_argument("--device", type=int, default=0)
    # parser.add_argument("--multi_gpu", type=bool, default=True)

    # parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--n_worker", type=int, default=os.cpu_count())
    parser.add_argument("--compile", action="store_true")    
    parser.add_argument("--tag", help="Tag name for snap/log sub directory")
    args = parser.parse_args()

    # check args
    args = check_args(args)

    # calculate the noise level (variance) from the normalized range
    stdev = args.stdev * (args.vmax - args.vmin)
    minmax = [args.vmin, args.vmax]

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    
    # load dataset
    data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2"  # HangSuit_task2_2_x3

    #==============================================#
    # state bounds
    arm_state_bounds = np.load(f"{data_dir_path}/param/arm_joint_bounds.npy")
    thresh = 0.02
    for i in range(arm_state_bounds.shape[1]):
        if arm_state_bounds[1,i] - arm_state_bounds[0,i] < thresh:
            arm_state_bounds[0,i] = arm_state_bounds[0].min()
            arm_state_bounds[1,i] = arm_state_bounds[1].max()

    # pressure bounds
    press_bounds = np.load(f"{data_dir_path}/param/pressure_bounds.npy")
    thresh = 100
    for i in range(press_bounds.shape[1]):
        if press_bounds[1,i] - press_bounds[0,i] < thresh:
            press_bounds[0,i] = press_bounds[0].min()
            press_bounds[1,i] = press_bounds[1].max()
    #==============================================#
    
    
    ################################################
    # train data
    #==============================================#
    # img
    raw_img_data = np.load(f"{data_dir_path}/train/right_img.npy")
    _img_data = raw_img_data.astype(np.float32)
    _img_data = resize_img(_img_data, (args.img_size, args.img_size))   # args.img_size, args.img_size
    _img_data = normalization(_img_data, (0.0, 255.0), minmax)
    img_data = np.transpose(_img_data, (0, 1, 4, 2, 3))
    # img_data = img_data[:,:,:,128:,128-32:256-32]
    #==============================================#
    
    #==============================================#
    # arm state
    raw_arm_state_data = np.load(f"{data_dir_path}/train/joint_state.npy")
    _arm_state_data = normalization(raw_arm_state_data, arm_state_bounds, minmax)
    arm_state_data = _arm_state_data[:,:,7:]

    # hand command
    raw_hand_cmd_data = np.load(f"{data_dir_path}/train/hand_cmd.npy")
    _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
    hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax)
    # vector
    vec_data = np.concatenate((arm_state_data, hand_cmd_data), axis=-1)
    
    # pressure
    raw_press_data = np.load(f"{data_dir_path}/train/pressure.npy")
    _press_data = normalization(raw_press_data, press_bounds, minmax)
    press_data = _press_data[:,:,19+3:19+6+1]
    # press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)
    #==============================================#
    #==============================================#
    # make dataset
    train_dataset = MultimodalDataset(img_data, 
                                    vec_data, 
                                    press_data,
                                    device=device, 
                                    dataset_device=device,
                                    stdev=stdev)    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.train_batch_size,
                                            shuffle=True,
                                            # num_workers=args.n_worker,
                                            # pin_memory=True,
                                            drop_last=False,)
    del train_dataset, img_data
    gc.collect()
    #==============================================#
    ################################################
    
    ################################################
    # test data
    #==============================================#
    # img
    raw_img_data = np.load(f"{data_dir_path}/test/right_img.npy")
    _img_data = raw_img_data.astype(np.float32)
    _img_data = resize_img(_img_data, (args.img_size, args.img_size))   # args.img_size, args.img_size
    _img_data = normalization(_img_data, (0.0, 255.0), minmax)
    img_data = np.transpose(_img_data, (0, 1, 4, 2, 3))
    # img_data = img_data[:,:,:,128:,128-32:256-32]
    #==============================================#
    
    #==============================================#
    # arm state
    raw_arm_state_data = np.load(f"{data_dir_path}/test/joint_state.npy")
    _arm_state_data = normalization(raw_arm_state_data, arm_state_bounds, minmax)
    arm_state_data = _arm_state_data[:,:,7:]

    # hand command
    raw_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
    _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
    hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax)
    # vector
    vec_data = np.concatenate((arm_state_data, hand_cmd_data), axis=-1)
    
    # pressure
    raw_press_data = np.load(f"{data_dir_path}/test/pressure.npy")
    _press_data = normalization(raw_press_data, press_bounds, minmax)
    _press_data = _press_data[:,:,19+3:19+6+1]
    press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)
    #==============================================#
    
    #==============================================#
    test_dataset = MultimodalDataset(img_data, 
                                    vec_data, 
                                    press_data,
                                    device=device, 
                                    dataset_device=device,
                                    stdev=stdev)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=args.test_batch_size,
                                            shuffle=True,
                                            # num_workers=args.n_worker,
                                            # pin_memory=True,
                                            drop_last=False,)
    del test_dataset, img_data
    gc.collect()
    #==============================================#
    ################################################
        
    key_dim = args.key_dim
    vec_dim = vec_data.shape[-1]
    press_dim = press_data.shape[-1]

    # define model
    model = SARNN(
        rec_dim=args.urnn_hid_dim,
        key_dim=key_dim,
        vec_dim=vec_dim,
        press_dim=press_dim,
        temperature=1e-4,
        heatmap_size=0.1,
        kernel_size=3,
        activation="lrelu",
        img_size=[128, 128],
        device=device
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
    scaler = torch.cuda.amp.GradScaler(init_scale=4096)

    # load trainer/tester class
    loss_w_dic = {
        "i": args.img_loss,
        "k": args.key_loss, 
        "v": args.vec_loss,
        "p": args.press_loss
    }
    trainer = simple_fullBPTTtrainer(
        model, 
        optimizer, 
        scaler,
        loss_w_dic=loss_w_dic, 
        device=device
    )

    # training main
    log_dir_path = set_logdir("./" + args.log_dir, args.tag)
    # save_name = os.path.join(log_dir_path, "HierarchicalRNN.pth")
    writer = SummaryWriter(log_dir=log_dir_path, flush_secs=30)
    early_stop = EarlyStopping(patience=1000)
    
    train_loss_list, test_loss_list = [], []
    with tqdm(range(args.epoch)) as pbar_epoch:
        for epoch in pbar_epoch:
            train_loss = trainer.process_epoch(train_loader, step=epoch)
            with torch.no_grad():
                test_loss = trainer.process_epoch(test_loader, step=epoch, training=False)
                
            writer.add_scalar("Loss/train_loss", train_loss, epoch)
            writer.add_scalar("Loss/test_loss", test_loss, epoch)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

            # early stop
            save_ckpt, _ = early_stop(test_loss)
            # scheduler.step(test_loss)

            # if save_ckpt:
            #     save_name = os.path.join(log_dir_path, f"HierarchicalRNN_{epoch}.pth")
            #     trainer.save(epoch, [train_loss, test_loss], save_name)
            # if (epoch % 100 == 0) or (epoch == 10):
            if (epoch % 10 == 0) or (epoch == args.epoch-1):
                save_name = os.path.join(log_dir_path, f"HierarchicalRNN_{epoch}.pth")
                trainer.save(epoch, [train_loss, test_loss], save_name)

            plt.figure()
            plt.plot(range(len(train_loss_list)), train_loss_list, label="train")
            plt.plot(range(len(test_loss_list)), test_loss_list, label="test")
            plt.grid()
            plt.legend()
            plt.savefig(f'{log_dir_path}/loss_trend.png')
            plt.cla()
            plt.clf()
            plt.close()
            
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
    
    ipdb.set_trace()
    
    

#  batch size = 2
# 128*128
"""
checkpoint enc conv
206
10.83 GB
10.88 GB
"""

"""
checkpoint enc
206
5.01 GB
5.02 GB
"""

"""
checkpoint enc, hrnn
206                                                                                                                                         
5.00 GB                                                                                                                                     
5.02 GB
"""

"""
checkpoint enc, dec
206
2.52 GB
2.52 GB
"""

"""
no checkpoint
206
12.16 GB
12.22 GB
"""


