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
import japanize_matplotlib
import time

import torch
import torch.nn as nn
import argparse

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization
from eipl.utils import resize_img, cos_interpolation

try:
    from libs.model import MSAHSARNN
    from libs.model import AbSCAE
    from libs.utils import moving_average
except:
    sys.path.append("./libs/")
    from model import MSAHSARNN
    from model import AbSCAE
    from utils import moving_average



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


def remove_module_prefix(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 'module.'を削除
            else:
                new_state_dict[k] = v
        return new_state_dict


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    # restore parameters
    ckpt = sorted(glob.glob(os.path.join(args.state_dir, '*.pth')))
    latest = ckpt.pop()
    # dir_name = os.path.split(args.state_dir)[0]
    dir_name = args.state_dir
    day = dir_name.split('/')[-1]
    params = restore_args( os.path.join(dir_name, 'args.json') )
    idx = args.idx
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    
    try:
        os.makedirs(f'./output/test_idx_{idx}/{params["tag"]}/')    # fast_tau_{int(params["fast_tau"])}/
    except:
        pass

    # load dataset
    minmax = [params["vmin"], params["vmax"]]

    data_dir_path = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2"
    
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
    # test data
    #==============================================#
    # img
    raw_img_data = np.load(f"{data_dir_path}/test/right_img.npy")
    _img_data = raw_img_data.astype(np.float32)
    plt_img_data = _img_data = resize_img(_img_data, (params["img_size"], params["img_size"]))    # params["img_size"], params["img_size"]
    _img_data = normalization(_img_data, (0.0, 255.0), minmax)
    img_data = np.transpose(_img_data, (0, 1, 4, 2, 3))
    # img_data = img_data[:,:,:,128:,128-32:256-32]
    #==============================================#
    
    #==============================================#
    # arm state
    raw_arm_state_data = np.load(f"{data_dir_path}/test/joint_state.npy")
    plt_arm_state_data = raw_arm_state_data[:,:,7:]
    _arm_state_data = normalization(raw_arm_state_data, arm_state_bounds, minmax)
    arm_state_data = _arm_state_data[:,:,7:]

    # hand command
    raw_hand_cmd_data = np.load(f"{data_dir_path}/test/hand_cmd.npy")
    plt_hand_cmd_data = _hand_cmd_data = np.apply_along_axis(cos_interpolation, 1, raw_hand_cmd_data, step=10)
    hand_cmd_data = normalization(_hand_cmd_data, (0.0, 1.0), minmax)
    # vector
    vec_data = np.concatenate((arm_state_data, hand_cmd_data), axis=-1)
    
    # pressure
    raw_press_data = np.load(f"{data_dir_path}/test/pressure.npy")
    plt_press_data = raw_press_data[:,:,19+3:19+6+1]
    _press_data = normalization(raw_press_data, press_bounds, minmax)
    press_data = _press_data[:,:,19+3:19+6+1]
    # press_data = np.apply_along_axis(moving_average, 1, _press_data, size=3)
    #==============================================#

    print("vector: ", vec_data.min(), vec_data.max())
    print("image: ", img_data.min(), img_data.max())

    key_dim = params["key_dim"]
    vec_dim = vec_data.shape[-1]
    press_dim = press_data.shape[-1]

    # define model
    # model = MSAHSARNN(
    #     srnn_hid_dim=params["srnn_hid_dim"],
    #     urnn_hid_dim=params["urnn_hid_dim"],
    #     key_dim=key_dim,
    #     vec_dim=vec_dim,
    #     press_dim=press_dim,
    #     temperature=params["heatmap_size"],
    #     heatmap_size=params["heatmap_size"],
    #     kernel_size=3,
    #     activation="lrelu",
    #     img_size=[params["img_size"], params["img_size"]],
    #     device="cpu"
    # )
    model = AbSCAE()


    if params["compile"]:
        model = torch.compile(model)

    # load weight
    ckpt = torch.load(latest, map_location=torch.device("cpu"))
    ckpt['model_state_dict'] = remove_module_prefix(ckpt['model_state_dict'])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference
    loss_w_dic = {"i": params["img_loss"],
                "k": params["key_loss"], 
                "v": params["vec_loss"],
                "p": params["press_loss"]}

    # image: numpy to tensor
    xi = yi = torch.from_numpy(img_data).float()
    # joint: numpy to tensor
    xv = yv = torch.from_numpy(vec_data).float()
    xp = yp = torch.from_numpy(press_data).float()

    states = None
    img_size = params["img_size"]
    yi_hat_list, yv_hat_list, yp_hat_list = [], [], []
    enc_pts_list, dec_pts_list = [], []
    urnn_hid_list = []
    T = xi.shape[1]
    
    prev_dec_xfs = None
    prev_enc_xf_last = None
    
    pts_list = []
    epoch = 0
    
    for t in range(T):
        # predict rnn
        print(t)
        start_time = time.time()
        
        yi_hat, enc_xf_last, dec_xfs, pts = model(
            xi[:, t], prev_enc_xf_last, prev_dec_xfs, epoch, t
        ) 
        
        prev_dec_xfs = dec_xfs
        prev_enc_xf_last = enc_xf_last
        
        end_time = time.time()
        print("duration", end_time - start_time)
        
        yi_hat_list.append(yi_hat)
        pts_list.append(pts)
        
        # urnn_hid_list.append(states[3][0])
        
        # print("step:{}, vec:{}".format(t, yv_hat))

    yi_hat_data = torch.permute(torch.stack(yi_hat_list), (1, 0, 2, 3, 4))
    pts_data = torch.permute(torch.stack(pts_list), (1, 0, 2, 3))
    
    _yi_hat_data = yi_hat_data.permute(0,1,3,4,2).detach().clone().cpu().numpy()
    _pts_data = pts_data.detach().clone().cpu().numpy()
    _pts_data = (_pts_data*128).astype(int)
    
    for i in range(200):
        plt.figure()
        plt.imshow(_yi_hat_data[0,i], origin='upper')
        plt.scatter(_pts_data[0,i,:,0], _pts_data[0,i,:,1])
        plt.savefig(f"./fig/abs9/scatter_plot_{i}.png")
        plt.close()
    
    ipdb.set_trace()

    
    
    # _enc_pts_data = torch.permute(torch.stack(enc_pts_list[1:]), (1, 0, 2))
    # ipdb.set_trace()
    # enc_pts_data = torch.cat([_enc_pts_data, torch.unsqueeze(_enc_pts_data[:,0], dim=1)], dim=1)

    # calc loss
    img_loss = (yi_hat_data - yi)**2 * loss_w_dic["i"]  #[:, 1:]
    loss = img_loss     #+ vec_loss + press_loss + key_loss

    # tensor2numpy
    # img
    _pred_img_data = yi_hat_data.detach().numpy()
    _pred_img_data = np.transpose(_pred_img_data, (0,1,3,4,2))
    pred_img_data = deprocess_img(_pred_img_data, vmin=minmax[0], vmax=minmax[1])

    plt_pred_img_data = pred_img_data[idx].astype(int)
    
    pts_data = np.clip(pts_data*128, 0, img_size)
    plt_pts_data = pts_data[idx]
    
    # plot images
    # T = len(images)
    T = len(plt_img_data) - 1 
    # fig, ax = plt.subplots(1, 3, figsize=(12, 5), dpi=60)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), dpi=60)
    def anim_update(i):
        print(i)
        
        # for j in range(3):
        #     ax[j].cla()
        for j in range(2):
            for k in range(3):
                ax[j][k].cla()

        # # plot camera image
        # ax[0].imshow(images[i, :, :, ::-1])
        # for j in range(params["k_dim"]):
        #     ax[0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1],
        #                "bo", markersize=6)  # encoder
        #     ax[0].plot(
        #         dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        #     )  # decoder
        # ax[0].axis("off")
        # ax[0].set_title("Input image")

        # # plot predicted image
        # ax[1].imshow(pred_image[i, :, :, ::-1])
        # ax[1].axis("off")
        # ax[1].set_title("Predicted image")

        # # plot joint angle
        # ax[2].set_ylim(0.0, 1.0)
        # ax[2].set_xlim(0, T)
        # ax[2].plot(joints[1:], linestyle="dashed", c="k")
        # # om has 5 joints, not 8
        # for joint_idx in range(joint_dim):
        #     ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
        # ax[2].set_xlabel("Step")
        # ax[2].set_title("Joint angles")
        
        
        # plot cam image
        ax[0][0].imshow(plt_img_data[i, :, :, ::-1])
        for j in range(params["key_dim"]):  # *4
            # ax[0][0].plot(plt_enc_pts_data[i, j, 0], 
            #             plt_enc_pts_data[i, j, 1], "bo", markersize=6)  # encoder
            # ax[0][0].plot(plt_dec_pts_data[i, j, 0], 
            #             plt_dec_pts_data[i, j, 1], "rx", markersize=6, markeredgewidth=2)  # decoder
            ax[0][0].plot(plt_pts_data[i, j, 0], 
                        plt_pts_data[i, j, 1], "bo", markersize=6)  # encoder

        # ax[0][0].axis("off")
        # ax[0][0].set_title("cam img")
        
        # plot pred img
        ax[1][0].imshow(plt_pred_img_data[i, :, :, ::-1]) # [i, :, :, ::-1]
        ax[1][0].axis("off")
        ax[1][0].set_title("pred img")
        
        # # plot joint
        # ax[0][1].set_ylim(-2.0, 2.5)
        # ax[0][1].set_xlim(0, T)
        # ax[0][1].plot(plt_arm_state_data[1:], linestyle="dashed", c="k")
        # # right arm has 7 joints, not 8
        # for joint_idx in range(7):
        #     ax[0][1].plot(np.arange(i + 1), plt_pred_arm_state_data[: i + 1, joint_idx])
        # # ax[0][1].set_xlabel("Step")
        # ax[0][1].set_title("right arm joint angles")
        
        # # plot command
        # ax[1][1].set_ylim(-1.0, 2.0)
        # ax[1][1].set_xlim(0, T)
        # ax[1][1].plot(plt_hand_cmd_data[1:], linestyle="dashed", c="k")
        # # right command has 3
        # for cmd_idx in range(3):
        #     ax[1][1].plot(np.arange(i + 1), plt_pred_hand_cmd_data[: i + 1, cmd_idx])
        # # ax[0][2].set_xlabel("Step")
        # ax[1][1].set_title("right hand command")
        
        # # plot pressure
        # ax[0][2].set_ylim(-1.0, 4096)
        # ax[0][2].set_xlim(0, T)
        # ax[0][2].plot(plt_press_data[1:], linestyle="dashed", c="k")
        # # right command has 3
        # for press_idx in range(4):
        #     ax[0][2].plot(np.arange(i + 1), plt_pred_press_data[: i + 1, press_idx])
        # # ax[0][2].set_xlabel("Step")
        # ax[0][2].set_title("pressure")

    ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
    ani.save(f'./output/test_idx_{idx}/{params["tag"]}/HierarchicalRNN_{params["tag"]}.gif')

    # If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
    # ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")


    loss = loss[0].detach().numpy()
    fig = plt.figure()
    plt.plot(range(len(loss)), loss, linestyle='solid', c='k', label="online")
    plt.plot(range(len(loss)), loss.mean()*np.ones_like(loss), linestyle='dashed', c='r', label="average")
    plt.grid()
    # plt.ylim(0, 0.05)
    plt.xlabel("step")
    plt.ylabel("tesst_loss")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0) # fontsize=18
    plt.savefig(f'./output/test_idx_{idx}/{params["tag"]}/test_loss_trend.png')
    plt.clf()
    plt.close()

    result = {
        "hid_size": {"srnn": params["srnn_hid_dim"], 
                    "urnn": params["urnn_hid_dim"]},
        "loss": {"test": float(loss.mean())}
    }
    with open(f'./output/test_idx_{idx}/{params["tag"]}/result.json', 'w') as f:
        json.dump(result, f, indent=2)
        

    