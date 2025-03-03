import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

sys.path.append("/home/shigeki/Documents/sachiya/work/eipl")
from eipl.model import SARNN
from eipl.data import MultimodalDataset, SampleDownloader, CustomDownloader
from eipl.utils import EarlyStopping, check_args, set_logdir

try:
    from libs.fullBPTT import fullBPTTtrainer
except:
    sys.path.append("/home/shigeki/Documents/sachiya/work/eipl/eipl/works/open_manipulator_grasp_cube/sarnn_ver")
    from libs.fullBPTT import fullBPTTtrainer

# argument parser
parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="sarnn")
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--rec_dim", type=int, default=50)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--img_loss", type=float, default=0.1)
parser.add_argument("--joint_loss", type=float, default=1.0)
parser.add_argument("--pt_loss", type=float, default=0.1)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=1e-4)
parser.add_argument("--stdev", type=float, default=0.02)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--tag", help="Tag name for snap/log sub directory")
args = parser.parse_args()

# check args
args = check_args(args)

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"
    
minmax = [args.vmin, args.vmax]

grasp_data = CustomDownloader("om", "grasp_cube", img_format="CHW")

print("train data")
images, joints = grasp_data.load_norm_data("train")
train_dataset = MultimodalDataset(images, joints)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)
for image in images:
    plt.imshow(image[0,::-1].transpose(1,2,0))
    plt.pause(1.0)


print("test data")
images, joints = grasp_data.load_norm_data("test", vmin=args.vmin, vmax=args.vmax)
test_dataset = MultimodalDataset(images, joints)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True
)
for image in images:
    plt.imshow(image[0,::-1].transpose(1,2,0))
    plt.pause(1.0)




