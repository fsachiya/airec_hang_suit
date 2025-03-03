#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, 
                 left_img_data, 
                 right_img_data,
                 arm_joint_data, 
                 hand_cmd_data, 
                 stdev=0.0, training=True):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.stdev = stdev
        self.training = training
        self.left_img_data = torch.from_numpy(left_img_data).float()
        self.right_img_data = torch.from_numpy(right_img_data).float()
        self.arm_joint_data = torch.from_numpy(arm_joint_data).float()
        self.hand_cmd_data = torch.from_numpy(hand_cmd_data).float()
        self.transform = transforms.ColorJitter(contrast=0.5, brightness=0.5, saturation=0.1)

        """
        self.transform_affine = transforms.Compose(
            [
                #transforms.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15)),
                transforms.RandomAutocontrast(),
            ]
        )
        self.transform_noise = transforms.Compose(
            [
                transforms.ColorJitter(contrast=0.5, brightness=0.5),
            ]
        )
        """

    def __len__(self):
        """
        Returns the number of the data.
        """
        return len(self.left_img_data)

    def __getitem__(self, idx):
        """
        Extraction and preprocessing of images and joints at the specified indexes.

        Args:
            idx (int): The index of the element.

        Returns:
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (y_img, y_joint).
        """

        if self.training:
            y_left_img = self.transform(self.left_img_data[idx])
            x_left_img = y_left_img + torch.normal(
                mean=0, std=self.stdev, size=y_left_img.shape
            )
            y_right_img = self.transform(self.right_img_data[idx])
            x_right_img = y_right_img + torch.normal(
                mean=0, std=self.stdev, size=y_right_img.shape
            )
            y_joint = self.arm_joint_data[idx]
            x_joint = self.arm_joint_data[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)
            y_cmd = self.hand_cmd_data[idx]
            x_cmd = self.hand_cmd_data[idx]
        else:
            x_left_img = self.left_img_data[idx]
            y_left_img = self.left_img_data[idx]
            x_right_img = self.right_img_data[idx]
            y_right_img = self.right_img_data[idx]
            x_joint = self.arm_joint_data[idx]
            y_joint = self.arm_joint_data[idx]
            y_cmd = self.hand_cmd_data[idx]
            x_cmd = self.hand_cmd_data[idx]

        return [[x_left_img, x_right_img, x_joint, x_cmd], [y_left_img, y_right_img, y_joint, y_cmd]]
