import sys
import os
import numpy as np
import argparse
from tqdm import tqdm
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

import torch
import torch.optim as optim
from collections import OrderedDict

sys.path.append("/home/shigeki/Documents/sachiya/work/eipl")
from eipl.model import SARNN
from eipl.utils import restore_args, normalization, tensor2numpy, deprocess_img

# argument parser
parser = argparse.ArgumentParser(
    description="Operate Open Manipulation"
)
parser.add_argument("--model_pth", type=str, default=None)
parser.add_argument("--input_param", type=float, default=0.8)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.model_pth)[0]
params = restore_args(os.path.join(dir_name, "args.json"))

# load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bounds = np.load(os.path.join(os.path.expanduser("~"), ".eipl/om/grasp_cube/joint_bounds.npy"))

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=5,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
)

# load weight
ckpt = torch.load(args.model_pth, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

rospy.init_node('test')




# Inference
# Set the inference frequency; for a 10-Hz in ROS system, set as follows.
freq = 10  # 10Hz
rate = rospy.Rate(freq)
image_list, joint_list = [], []
state = None
nloop = 200  # freq * 20 sec
for loop_ct in range(nloop):
    start_time = time.time()

    # load data and normalization
    raw_images, raw_joint = robot.get_sensor_data()
    x_img = raw_images[loop_ct].transpose(2, 0, 1)
    x_img = torch.Tensor(np.expand_dims(x_img, 0))
    x_img = normalization(x_img, (0, 255), minmax)
    x_joint = torch.Tensor(np.expand_dims(raw_joint, 0))
    x_joint = normalization(x_joint, joint_bounds, minmax)

    # closed loop
    if loop_ct > 0:
        x_img = args.input_param * x_img + (1.0 - args.input_param) * y_image
        x_joint = args.input_param * x_joint + (1.0 - args.input_param) * y_joint

    # predict rnn
    y_image, y_joint, state = rnn_model(x_img, x_joint, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, cae_params["vmin"], cae_params["vmax"])
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # send pred_joint to robot
    # send_command(pred_joint)
    pub.publish(pred_joint)

    # Sleep to infer at set frequency.
    # ROS system
    rate.sleep()



def call_back(ros_data):
    np_arr = np.fromstring(ros_data.data, np.uint8)
    print(np_arr)