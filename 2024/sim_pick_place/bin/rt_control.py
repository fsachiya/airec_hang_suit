#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import cv2
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from natsort import natsorted
import glob
import time
import ipdb

sys.path.append("/home/fujita/work/robosuite")
import robosuite as suite
from robosuite import load_controller_config

# from eipl.model import SARNN
from eipl.utils import tensor2numpy, normalization, deprocess_img, restore_args

sys.path.append("/home/fujita/work/eipl/eipl/tutorials/robosuite/simulator/libs")
from environment import cube
from samplers import BiasedRandomSampler
from rt_control_wrapper import RTControlWrapper

# local
sys.path.append("../")
from util import ImgStateDataset, Visualize
from libs import fullBPTTtrainer4SARNN, fullBPTTtrainer4StackRNN, fullBPTTtrainer4HRNN, fullBPTTtrainer4FasterHRNN
from model import SARNN, StackRNN, HRNN, FasterHRNN
from inf import Inf4HRNN, Inf4StackRNN, Inf4FasterHRNN


# # load param
# joint_bounds = np.load("./data/joint_bounds.npy")


def rt_control(env, nloop, rate, open_ratio, params):   # open_ratio
    state_bounds = params["data"]["state_bounds"]
    img_bounds = params["data"]["img_bounds"]
    minmax = [params["data"]["vmin"], params["data"]["vmax"]]
    model_name = params["model"]["model_name"]

    # Set user parameter
    loop_ct = 0

    # Reset the environment and Set environment
    obs = env.reset()
    initial_q = np.deg2rad([0.0, 12.0, 0.0, -150.0, 0.0, 168.0, 45])
    env.set_joint_qpos(initial_q)
    env.sim.forward()

    gripper_command = -1.0
    gripper_state = np.array([0.0])
    state = None
    
    x_img_list = []
    x_state_list = []
    y_img_hat_list = []
    y_state_hat_list = []
    if model_name == "hrnn":
        hid_dict = {"img_feat": None, "state": None, "union1": None, "union2": None}
    elif model_name == "stackrnn":
        hid_dict = {"union1": None, "union2": None, "union3": None}
    for loop_ct in range(nloop):
        if loop_ct % rate == 0:
            # get image
            img = env.get_image()
            
            # (h, w) = img.shape[:2]
            # center = (w // 2, h // 2)
            # angle = 180
            # rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # rot_img = cv2.warpAffine(img, rot_matrix, (w, h))
            # rot_img = rot_img[:,:,::-1]
            
            x_img = cv2.resize(img[::-1], (64, 64))
            x_img = np.expand_dims(x_img.transpose(2, 0, 1), 0)
            x_img = normalization(x_img.astype(np.float32), (0, 255), minmax)
            
            # get joint and gripper angles
            joint = env.get_joints()
            state = np.concatenate((joint, gripper_state), axis=-1)
            x_state = np.expand_dims(state, 0)
            x_state = normalization(x_state, state_bounds, minmax)
                
            # rt predict
            x_img = torch.Tensor(x_img).unsqueeze(0).to(device)
            x_state = torch.Tensor(x_state).unsqueeze(0).to(device)
            x_img_list.append(x_img)
            x_state_list.append(x_state)
            
            if loop_ct > 0:
                x_img = open_ratio * x_img + (1.0 - open_ratio) * y_img_hat_list[-1]
                x_state = open_ratio * x_state + (1.0 - open_ratio) * y_state_hat_list[-1]

            if model_name == "hrnn":
                x_img = x_img[0]
                x_state = x_state[0]
                y_img_hat, y_state_hat, hid_dict  = model(x_img, x_state, hid_dict)
                y_img_hat = y_img_hat.unsqueeze(0)
                y_state_hat = y_state_hat.unsqueeze(0)
                y_img_hat_list.append(y_img_hat)
                y_state_hat_list.append(y_state_hat)
            elif model_name == "stackrnn":
                stack = False
                if stack:
                    x_imgs = torch.cat(x_img_list).permute(1,0,2,3,4)
                    x_states = torch.cat(x_state_list).permute(1,0,2)
                    y_imgs_hat, y_states_hat, hids_dict, hid_dict  = model(x_imgs, x_states)
                    y_img_hat = y_imgs_hat[:,-1].unsqueeze(0)
                    y_state_hat = y_states_hat[:,-1].unsqueeze(0)
                else:
                    y_img_hat, y_state_hat, hids_dict, hid_dict  = model(x_img, x_state, hid_dict)
                y_img_hat_list.append(y_img_hat)
                y_state_hat_list.append(y_state_hat)
            
            _y_img_hat = normalization(y_img_hat, minmax, img_bounds)
            _y_img_hat = _y_img_hat.permute(0,1,3,4,2)
            pred_img = np.uint8(_y_img_hat[0,0].detach().clone().cpu().numpy())

            # post process
            # yv = tensor2numpy(_yv[0])
            # yv = normalization(yv, minmax, joint_bounds)
            _y_state_hat = normalization(y_state_hat, minmax, state_bounds)
            pred_state = _y_state_hat[0,0].detach().clone().cpu().numpy()
            
            if pred_state[-1] > 0.7 and gripper_command == -1.0:
                gripper_command = 1.0
            if pred_state[-1] < 0.3 and gripper_command == 1.0:
                gripper_command = -1.0
            gripper_state = pred_state[-1:]

            action = env.get_joint_action(pred_state[:-1], kp=rate)
            action[-1] = gripper_command

        env.step(action)
        env.render()
        _, success = env.get_state()
        if success:
            break
    
    
    ################################################
    y_imgs_hat = torch.cat(y_img_hat_list).permute(1,0,2,3,4)
    y_states_hat = torch.cat(y_state_hat_list).permute(1,0,2)
    
    y_imgs = torch.cat(x_img_list).permute(1,0,2,3,4)
    y_states = torch.cat(x_state_list).permute(1,0,2)
    
    # pred plot
    _y_imgs_hat = normalization(y_imgs_hat, minmax, img_bounds)
    _y_imgs_hat = _y_imgs_hat.permute(0,1,3,4,2)
    # _y_imgs_hat = _y_imgs_hat[[0,2,4,6,8,10,12,14,16]]
    
    _y_states_hat = normalization(y_states_hat, minmax, state_bounds)
    # _y_states_hat = _y_states_hat[[0,2,4,6,8,10,12,14,16]]

    pred_imgs = np.uint8(_y_imgs_hat.detach().clone().cpu().numpy())
    pred_states = _y_states_hat.detach().clone().cpu().numpy()
    
    # input plot
    _y_imgs = normalization(y_imgs, minmax, img_bounds)
    _y_imgs = _y_imgs.permute(0,1,3,4,2)
    # _y_imgs = _y_imgs[[0,2,4,6,8,10,12,14,16]]

    _y_states = normalization(y_states, minmax, state_bounds)
    # _y_states = _y_states[[0,2,4,6,8,10,12,14,16]]

    y_imgs = np.uint8(_y_imgs.detach().clone().cpu().numpy())
    y_states = _y_states.detach().clone().cpu().numpy()
    
    
    save_log_dir = f"../output/{args.log_dir_name}"
    try:
        os.makedirs(save_log_dir)
    except FileExistsError as e:
        pass
    
    # plot images
    T = y_imgs.shape[1]
    fig, ax = plt.subplots(3, 1, figsize=(5, 12), dpi=60)
    
    ################################################################
    show_inf = True
    if show_inf:
        # for idx in range(y_imgs.shape[0]):
        idx = 0
        def anim_update(i):
            print(i)
            for j in range(3):
                ax[j].cla()

            # plot camera image
            ax[0].imshow(y_imgs[idx, i, :, :, ::-1])
            # for j in range(params["k_dim"]):
            #     ax[0].plot(enc_pts[i, j, 0], enc_pts[i, j, 1], "bo", markersize=6)  # encoder
            #     ax[0].plot(
            #         dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
            #     )  # decoder
            ax[0].axis("off")
            ax[0].set_title("Input image")

            # plot predicted image
            ax[1].imshow(pred_imgs[idx, i, :, :, ::-1])
            ax[1].axis("off")
            ax[1].set_title("Predicted image")

            # plot joint angle
            # ax[2].set_ylim(-1.0, 2.0)
            ax[2].set_xlim(0, T)
            ax[2].plot(y_states[idx], linestyle="dashed", c="k")
            for joint_idx in range(8):
                ax[2].plot(np.arange(i + 1), pred_states[idx, :i+1, joint_idx])
            ax[2].set_xlabel("Step")
            ax[2].set_title("Joint angles")
        
        curr = time.strftime("%Y%m%d_%H%M%S")
        ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
        ani.save(f"{save_log_dir}/{model_name}_sim_{curr}_{open_ratio}.gif")

            # ipdb.set_trace()

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("filename", type=str, help="Model name")
    parser.add_argument("--log_dir_name", type=str, default="20241210_1427_23")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rate", type=int, default=5)  # Down sampling rate
    parser.add_argument("--open_ratio", type=float, default=1.0)
    args = parser.parse_args()
    
    # check args
    assert args.log_dir_name or args.pretrained, "Please set log_dir_name or pretrained"

    # set device id
    if args.device >= 0:
        device = "cuda:{}".format(args.device)
    else:
        device = "cpu"

    log_dir_path = f"../log/{args.log_dir_name}"
    params = restore_args(os.path.join(log_dir_path, "args.json"))
    # dir_name = os.path.split(args.filename)[0]
    # params = restore_args(os.path.join(dir_name, "args.json"))
    # minmax = [params["vmin"], params["vmax"]]

    # define model
    # model = SARNN(
    #     rec_dim=params["rec_dim"],
    #     joint_dim=8,
    #     k_dim=params["k_dim"],
    #     heatmap_size=params["heatmap_size"],
    #     temperature=params["temperature"],
    #     im_size=[64, 64],
    # )
    
    model_name = params["model"]["model_name"]
    # if model_name in ["stackrnn"]:
    #     model = StackRNN(
    #         img_feat_dim=8,
    #         state_dim=8,
    #         union1_dim=params["model"]["hid_dim"],
    #         union2_dim=params["model"]["hid_dim"],
    #         union3_dim=params["model"]["hid_dim"],
    #     ).to(device)
    
    # define model
    if model_name in ["sarnn",]:
        model = SARNN(
            union_dim=params["model"]["hid_dim"],
            state_dim=8,
            key_dim=params["key_dim"],
            heatmap_size=params["heatmap_size"],
            temperature=params["temperature"],
        ).to(device)
    elif model_name in ["stackrnn"]:
        model = StackRNN(
            img_feat_dim=8,
            state_dim=8,
            union1_dim=params["model"]["hid_dim"],
            union2_dim=params["model"]["hid_dim"],
            union3_dim=params["model"]["hid_dim"],
        ).to(device)
    elif model_name in ["hrnn"]:
        model = HRNN(
            img_feat_dim=8,
            state_dim=8,
            sensory_dim=params["model"]["hid_dim"],
            union1_dim=params["model"]["hid_dim"],
            union2_dim=params["model"]["hid_dim"],
        ).to(device)
    elif model_name in ["fasterhrnn"]:
        model = FasterHRNN(
            img_feat_dim=8,
            state_dim=8,
            sensory_dim=int(params["model"]["hid_dim"]/2),
            union1_dim=params["model"]["hid_dim"],
            union2_dim=params["model"]["hid_dim"],
        ).to(device)
    else:
        print(f"{model_name} is invalid model")
        exit()

    if params["inf"]["compile"]:
        model = torch.compile(model)

    # # load weight
    # ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
    # model.load_state_dict(ckpt["model_state_dict"])
    # model.eval()
    weight_pathes = natsorted(glob.glob(f"{log_dir_path}/*.pth"))
    ckpt = torch.load(weight_pathes[-1], map_location=torch.device("cuda:0"), weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Get controller config
    # Import controller config IK_POSE/OSC_POSE/OSC_POSITION/JOINT_POSITION
    controller_config = load_controller_config(default_controller="JOINT_POSITION")

    # Create argument configuration
    config = {
        "env_name": "Lift",
        "robots": "Panda",
        "controller_configs": controller_config,
    }

    # train position x7 (5 for train, 2 for test)
    # test  position x2 (all for test)
    #            Pos1 Pos2  Pos3  Pos4 Pos5 Pos6 Pos7 Pos8 Pos9
    # pos_train: -0.2       -0.1       0.0       0.1       0.2
    # pos_test:  -0.2 -0.15 -0.1 -0.05 0.0  0.05 0.1  0.15 0.2
    # reinforcement: -0.3/0.3

    x_pos = normalization(np.random.random(), (0, 1), (-0.2, 0.2))
    position_sampler = BiasedRandomSampler(
        name="ObjectSampler",
        mujoco_objects=cube,
        rotation=False,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
        pos_bias_list=[[0.0, x_pos]],
    )

    # create original environment
    # create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        placement_initializer=position_sampler,
    )

    # wrap the environment with data collection wrapper
    env = RTControlWrapper(env)
    
    state_bounds = params["data"]["state_bounds"]


    total_loop = 10
    # realtime contol
    for loop_ct in range(total_loop):
        res = rt_control(env, nloop=1000, rate=args.rate, open_ratio=args.open_ratio, params=params)
        if res:
            print("[{}/{}] Task succeeded!".format(loop_ct + 1, total_loop))
        else:
            print("[{}/{}] Task failed!".format(loop_ct + 1, total_loop))
