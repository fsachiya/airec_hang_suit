import numpy as np
import glob
import argparse
from natsort import natsorted
import os

parser = argparse.ArgumentParser(
    description="Learning spatial autoencoder with recurrent neural network"
)
parser.add_argument("--model", type=str, default="hierachical_rnn")
parser.add_argument("--epoch", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=5)




ckpt = natsorted(glob.glob(os.path.join(args.state_dir, '*.pth')))

npz_data = np.load(f"{data_dir_path}/train/right_img.npy")