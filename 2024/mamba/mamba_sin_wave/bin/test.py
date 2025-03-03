import os
import sys
import cv2
import shutil
import time
import numpy as np
import json
import glob
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import argparse

# from mamba_ssm import Mamba
sys.path.append("/home/fujita/work/mamba")
from _mamba import Mamba, MambaConfig

import ipdb
from einops import rearrange

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy
from eipl.utils import EarlyStopping, check_args, set_logdir, normalization

# own library
try:
    from libs.model import CNNMTRNN
    from libs.utils import get_unique_list
    from libs.utils import deprocess_img, get_batch
    from libs.trainer import fullBPTTtrainer
    from libs.utils import sinwave

except:
    sys.path.append("libs/")
    from model import CNNMTRNN
    from utils import get_unique_list
    from utils import deprocess_img, get_batch
    from trainer import fullBPTTtrainer
    from utils import sinwave


parser = argparse.ArgumentParser(
    description="mtrnn square wave prediction test"
)
parser.add_argument('state_dir')
parser.add_argument('--model', type=str, default="mtrnn")
parser.add_argument('--total_step', type=int, default=200)
parser.add_argument('--input_param', type=float, default=1.0)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--plt_show_flag', action='store_true')
parser.add_argument('--idx', type=int, default=1)
args = parser.parse_args()    

# set device id
if args.device >= 0:
    device = "cuda:{}".format(args.device)
else:
    device = "cpu"

# restore parameters
ckpt = sorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
latest = ckpt.pop()
# dir_name = os.path.split(args.state_dir)[0]
dir_name = args.state_dir
day = dir_name.split('/')[-1]
params = restore_args( os.path.join(dir_name, 'args.json') )
idx = args.idx

try:
    os.makedirs(f'./output/test_pos_{idx}/{params["tag"]}/')
except:
    pass

minmax = [params["vmin"], params["vmax"]]
test_sinwave = np.array(sinwave.test_list)[idx]
test_sinwave = np.expand_dims(test_sinwave, 0)
test_x = test_sinwave[:,:-1]
test_y = test_sinwave[:,1:]

# # model of neural network
# context_size = {"cf": 100, "cs": 50}
# tau = {"cf": 4.0, "cs": 20.0}

# # model of neural network
# model = CNNMTRNN(context_size, tau)
batch, length, dim = 1, test_x.shape[1], 1
# model = Mamba(
#     d_model=dim,  # モデルの次元
#     d_state=16,   # SSMの状態の拡張係数
#     d_conv=4,     # ローカルな畳み込みの幅
#     expand=2     # ブロックの拡張係数
# ).to(device)
config = MambaConfig(d_model=dim, 
                     n_layers=1,
                     d_state=16,
                     d_conv=4,
                     expand_factor=2)
model = Mamba(config).to(device)


ckpt = torch.load(latest, map_location=torch.device(device))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ipdb.set_trace()

ext_hs_list = [None,]
y_hat_list = []
T = test_x.shape[1]

x = torch.tensor(test_x).unsqueeze(dim=2).float().to(device)
y = torch.tensor(test_y).unsqueeze(dim=2).float().to(device)

# for t in range(T):
#     _y, state = model(
#         x[:,t], state
#     )
#     y_hat_list.append(_y)
# y_hat = torch.stack([y_hat for y_hat in y_hat_list])

y_hat, hs = model(x, ext_hs_list=ext_hs_list)
# y_hat = y_hat.permute(1,0,2)
loss = ((y_hat - y) ** 2)

ipdb.set_trace()


loss_list = tensor2numpy(loss.reshape(-1))
y_hat_list = tensor2numpy(y_hat.reshape(-1))
y_list = tensor2numpy(y.reshape(-1))

fig = plt.figure()
plt.plot(range(len(loss_list)), loss_list, linestyle='solid', label="loss_online")
plt.plot(range(len(loss_list)), loss_list.mean()*np.ones_like(loss_list), linestyle='dashed', label="loss_average")
plt.grid()
plt.xlabel("step")
plt.ylabel("tesst_loss")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_pos_{idx}/{params["tag"]}/test_loss_trend.png')
plt.clf()
plt.close()


fig = plt.figure()
plt.plot(range(len(y_list)), y_list, linestyle="dashed", label="true val")
plt.plot(range(len(y_hat_list)), y_hat_list, linestyle="solid", label="pred val")
plt.grid()
plt.xlabel("x")
plt.ylabel("y")
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
plt.savefig(f'./output/test_pos_{idx}/{params["tag"]}/sin_wave.png')
plt.clf()
plt.close()


# result = {
#     "context_size": {"cf": context_size["cf"], "cs": context_size["cs"]},
#     "loss": {"test": float(loss_list.mean())}
# }
# with open(f'./output/test_pos_{idx}/{params["tag"]}/result.json', 'w') as f:
#     json.dump(result, f, indent=2)