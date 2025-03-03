import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import glob
import ipdb

class Visualize():
    def __init__(
        self, 
        log_dir_path, 
        model_name,
        epochs
        ):
        self.log_dir_path = log_dir_path
        self.model_name = model_name
        self.epochs = epochs
        
    def viz_train_time(
        self, 
        train_epoch_times, 
        figsize=(12,6), 
        fig_name="train_time"
        ):
        save_name = os.path.join(self.log_dir_path, f"{fig_name}.png")
        
        # train time
        plt.figure(figsize=figsize)
        plt.plot(range(train_epoch_times.shape[0]), train_epoch_times, label="train time")
        plt.xlabel("x10 epoch")
        plt.ylabel("sec")
        plt.legend()
        plt.title(f"training time(per epoch) of {self.model_name}")
        plt.savefig(save_name)
        plt.cla()
        plt.clf()
        plt.close()
        
    def viz_train_gpu(
        self, 
        train_epoch_gpu_usages, 
        train_epoch_mem_usages, 
        fig_name="train_gpu"
        ):
        save_name = os.path.join(self.log_dir_path, f"{fig_name}.png")

        # train gpu
        plt.figure(figsize=(12,6))
        plt.plot(range(train_epoch_gpu_usages.shape[0]), train_epoch_gpu_usages, label="train gpu usage")
        plt.plot(range(train_epoch_mem_usages.shape[0]), train_epoch_mem_usages, label="train mem usage")
        plt.xlabel("x10 epoch")
        plt.ylabel("usage")
        plt.ylim([0,1])
        plt.legend()
        plt.title(f"training gpu/memory usage(per epoch) of {self.model_name}")
        plt.savefig(save_name)
        plt.cla()
        plt.clf()
        plt.close()

    def viz_train_loss(
        self, 
        train_epoch_losses, 
        valid_epoch_losses, 
        offset=0,
        fig_name="train_loss"
        ):
        save_name = os.path.join(self.log_dir_path, f"{fig_name}.png")

        # train/valid loss
        plt.figure(figsize=(12,6))
        plt.plot(range(train_epoch_losses.shape[0])[offset:], train_epoch_losses[offset:], label="train loss")
        plt.plot(range(valid_epoch_losses.shape[0])[offset:], valid_epoch_losses[offset:], label="valid loss")
        plt.xlabel("x10 epoch")
        plt.ylabel("loss")
        # plt.ylim([0,1])
        plt.xlim([offset,self.epochs+1])
        plt.legend()
        plt.title(f"training/validation loss(per epoch) of {self.model_name}")
        plt.savefig(save_name)
        plt.cla()
        plt.clf()
        plt.close()



# # train plot
# if data_type == "sin":
#     pass
# elif data_type == "lissajous":
#     plt.figure(figsize=(12,12))
#     label_list = ["x,y,delta=(1,2,pi/2)", 
#                   "x,y,delta=(2,1,pi/2)", 
#                   "x,y,delta=(3,4,pi/2)", 
#                   "x,y,delta=(4,3,pi/2)",
#                   "x,y,delta=(-1,2,pi/2)", 
#                   "x,y,delta=(-2,1,pi/2)", 
#                   "x,y,delta=(-3,4,pi/2)", 
#                   "x,y,delta=(-4,3,pi/2)",]
#     for i in range(8):
#         plt.plot(x_train_dict["seq"][i,:,0].detach().cpu().numpy(), x_train_dict["seq"][i,:,1].detach().cpu().numpy(), label=label_list[i])
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title(f"train data")
#     plt.savefig(f"../log/{log_dir_name}/train_plot.png")
#     plt.cla()
#     plt.clf()
#     plt.close()
    
# # test plot
# if data_type == "sin":
#     pass
# elif data_type == "lissajous":
#     plt.figure(figsize=(12,12))
#     plt.plot(x_test_dict["seq"][0,:,0].detach().cpu().numpy(), x_test_dict["seq"][0,:,1].detach().cpu().numpy(), label="x,y,delta=(2,3,pi/2)")
#     plt.plot(x_test_dict["seq"][1,:,0].detach().cpu().numpy(), x_test_dict["seq"][1,:,1].detach().cpu().numpy(), label="x,y,delta=(3,2,pi/2)")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title(f"test data")
#     plt.savefig(f"../log/{log_dir_name}/test_plot.png")
#     plt.cla()
#     plt.clf()
#     plt.close()

# # inf plot
# if data_type == "sin":
#     plt.figure(figsize=(12,6))
#     plt.plot(t[offset:], x_test[0,:,0].detach().cpu().numpy(), label="sin(2x) (original)", linestyle="dotted")
#     plt.plot(t[offset:], x_test[1,:,0].detach().cpu().numpy(), label="sin(4x) (original)", linestyle="dotted")
#     plt.plot(t[offset:], y_test[0,:,0].detach().cpu().numpy(), label="sin(2x) (actual)")
#     plt.plot(t[offset:], y_test[1,:,0].detach().cpu().numpy(), label="sin(4x) (actual)")
#     plt.plot(t[offset:], preds[0,:,0], label="sin(2x) (prediction)", linestyle="--")
#     plt.plot(t[offset:], preds[1,:,0], label="sin(4x) (prediction)", linestyle="--")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title(f"inference of {model_name}")
#     plt.savefig(f"../log/{log_dir_name}/inf_plot.png")
#     plt.cla()
#     plt.clf()
#     plt.close()
# elif data_type == "lissajous":
#     plt.figure(figsize=(12,12))
#     # plt.plot(x_test[0,:,0].detach().cpu().numpy(), x_test[0,:,1].detach().cpu().numpy(), label="sin(2x) (original)", linestyle="dotted")
#     # plt.plot(x_test[1,:,0].detach().cpu().numpy(), x_test[1,:,1].detach().cpu().numpy(), label="sin(4x) (original)", linestyle="dotted")
#     plt.plot(y_test_dict["seq"][0,:,0].detach().cpu().numpy(), y_test_dict["seq"][0,:,1].detach().cpu().numpy(), label="x,y,delta=(2,3,pi/2) (actual)")
#     plt.plot(y_test_dict["seq"][1,:,0].detach().cpu().numpy(), y_test_dict["seq"][1,:,1].detach().cpu().numpy(), label="x,y,delta=(3,2,pi/2) (actual)")
#     plt.plot(_ys[0,:,0], _ys[0,:,1], label="x,y,delta=(2,3,pi/2) (prediction)", linestyle="--")
#     plt.plot(_ys[1,:,0], _ys[1,:,1], label="x,y,delta=(3,2,pi/2) (prediction)", linestyle="--")
#     plt.scatter([y_test_dict["seq"][0,0,0].detach().cpu().numpy(), 
#                  y_test_dict["seq"][1,0,0].detach().cpu().numpy(),
#                  _ys[0,0,0], _ys[1,0,0]],
#                 [y_test_dict["seq"][0,0,1].detach().cpu().numpy(), 
#                  y_test_dict["seq"][1,0,1].detach().cpu().numpy(),
#                  _ys[0,0,1], _ys[1,0,1]], c="black")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title(f"inference of {model_name}")
#     plt.savefig(f"../log/{log_dir_name}/inf_plot.png")
#     plt.cla()
#     plt.clf()
#     plt.close()

# # inf time
# plt.figure(figsize=(12,6))
# plt.plot(range(inf_seq_times.shape[0]), inf_seq_times, label="inf time")
# plt.xlabel("x10 seq")
# plt.ylabel("sec")
# plt.legend()
# plt.title(f"inference time(per seq) of {model_name}")
# plt.savefig(f"../log/{log_dir_name}/inf_time.png")
# plt.cla()
# plt.clf()
# plt.close()

# # inf gpu
# plt.figure(figsize=(12,6))
# plt.plot(range(inf_seq_gpu_usages.shape[0]), inf_seq_gpu_usages, label="inf gpu usage")
# plt.plot(range(inf_seq_mem_usages.shape[0]), inf_seq_mem_usages, label="inf mem usage")
# plt.xlabel("x10 seq")
# plt.ylabel("usage")
# plt.ylim([0,1])
# plt.legend()
# plt.title(f"inference gpu/memory usage(per seq) of {model_name}")
# plt.savefig(f"../log/{log_dir_name}/inf_gpu.png")
# plt.cla()
# plt.clf()
# plt.close()