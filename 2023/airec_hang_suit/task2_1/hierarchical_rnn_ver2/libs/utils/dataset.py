#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn as nn
import albumentations as A


class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, img_data, vec_data, device, stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.device = device
        self.stdev = stdev
        self.img_data = torch.Tensor(img_data).to(self.device)
        self.vec_data = torch.Tensor(vec_data).to(self.device)
        self.transform = nn.Sequential(transforms.RandomErasing(),
                                       transforms.ColorJitter(brightness=0.4),
                                       transforms.ColorJitter(contrast=[0.6, 1.4]),
                                       transforms.ColorJitter(hue=[0.0, 0.04]),
                                       transforms.ColorJitter(saturation=[0.6, 1.4])).to(self.device)

        """
        self.transform = nn.Sequence(
            transforms.ColorJitter(
            contrast=[0.6, 1.4], brightness=0.4, saturation=[0.6, 1.4], hue=0.04
        ).to(self.device)
        """

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.img_data)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """
        x_img = self.img_data[idx]
        x_vec = self.vec_data[idx]
        y_img = self.img_data[idx]
        y_vec = self.vec_data[idx]

        if self.stdev is not None:
            x_img = self.transform(y_img)
            x_img = x_img + torch.normal(mean=0, std=0.02, size=x_img.shape, device=self.device)
            x_vec = y_vec + torch.normal(mean=0, std=self.stdev, size=y_vec.shape, device=self.device)
        
        return [[x_img, x_vec], [y_img, y_vec]]

