#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#
import ipdb
import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

import torch
import torch.nn as nn
import argparse

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img, cos_interpolation

try:
    from libs.model import StereoHSARNN
    from libs.utils import moving_average
except:
    sys.path.append("./libs/")
    from model import StereoHSARNN
    from utils import moving_average

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

vec_dim = 10
press_dim = 4

model = StereoHSARNN(
    # rnn_dim=args.rnn_dim,
    # union_dim=args.union_dim,
    srnn_hid_dim = params["srnn_hid_dim"],
    urnn_hid_dim = params["urnn_hid_dim"],
    k_dim=params["k_dim"],
    vec_dim=vec_dim,
    press_dim=press_dim,
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    img_size=[params["img_size"], params["img_size"]]
    )

def measure_number_dead_neurons(net):
    # For each neuron, we create a boolean variable initially set to 1. If it has an activation unequals 0 at any time,
    # we set this variable to 0. After running through the whole training set, only dead neurons will have a 1.
    neurons_dead = [
        torch.ones(layer.weight.shape[0], device=device) for layer in net.layers[:-1] if isinstance(layer, nn.Linear)
    ] # Same shapes as hidden size in BaseNetwork
    net.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(train_loader, leave=False): # Run through whole training set
            layer_index = 0
            imgs = imgs.to(device)
            imgs = imgs.view(imgs.size(0), -1)
            for layer in net.layers[:-1]:
                imgs = layer(imgs)
                if isinstance(layer, BaseActivationFunc):
                    # Are all activations == 0 in the batch, and we did not record the opposite in the last batches?
                    neurons_dead[layer_index] = torch.logical_and(neurons_dead[layer_index], (imgs == 0).all(dim=0))
                    layer_index += 1
    number_neurons_dead = [t.sum().item() for t in neurons_dead]
    print("Number of dead neurons:", number_neurons_dead)
    print("In percentage:", ", ".join([f"{(100.0 * num_dead / tens.shape[0]):4.2f}%" for tens, num_dead in zip(neurons_dead, number_neurons_dead)]))
    
    
measure_number_dead_neurons(model)