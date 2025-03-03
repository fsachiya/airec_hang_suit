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
import ipdb


class MultimodalDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, 
                 img_data, 
                 vec_data, 
                 press_data,
                 device, 
                 dataset_device,
                 stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.device = device
        self.dataset_device = dataset_device
        self.stdev = stdev
        self.img_data = torch.Tensor(img_data).to(self.device)
        self.vec_data = torch.Tensor(vec_data).to(self.device)
        self.press_data = torch.Tensor(press_data).to(self.device)
        self.transform = nn.Sequential( #transforms.RandomErasing(),
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
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (yi, y_joint).
        """
        
        xi = self.img_data[idx]
        xv = self.vec_data[idx]
        xp = self.press_data[idx]
        yi = self.img_data[idx]
        yv = self.vec_data[idx]
        yp = self.press_data[idx]

        if self.stdev is not None:
            xi = self.transform(yi)
            xi = xi + torch.normal(mean=0, std=0.02, size=xi.shape, device=self.device)
            xv = yv + torch.normal(mean=0, std=self.stdev, size=yv.shape, device=self.device)
            xp = yp +  torch.normal(mean=0, std=self.stdev, size=yp.shape, device=self.device)
        
        # if self.stdev is not None:
        #     xi = self.transform(xi)
        #     xi = xi + torch.normal(mean=0, std=0.02, size=xi.shape, device=self.device)
        #     xv = xv + torch.normal(mean=0, std=self.stdev, size=xv.shape, device=self.device)
        #     xp = xp +  torch.normal(mean=0, std=self.stdev, size=xp.shape, device=self.device)
        
        xi = xi.to(self.dataset_device)
        xv = xv.to(self.dataset_device)
        xp = xp.to(self.dataset_device)        
        yi = yi.to(self.dataset_device)
        yv = yv.to(self.dataset_device)
        yp = yp.to(self.dataset_device)    
        
        # print(xi.is_cuda)
        # print(xv.is_cuda)
        # print(xp.is_cuda)
        
        return [[xi, xv, xp], [yi, yv, yp]]





class ImgDataset(Dataset):
    """
    This class is used to train models that deal with multimodal data (e.g., images, joints), such as CNNRNN/SARNN.

    Args:
        images (numpy array): Set of images in the dataset, expected to be a 5D array [data_num, seq_num, channel, height, width].
        joints (numpy array): Set of joints in the dataset, expected to be a 3D array [data_num, seq_num, joint_dim].
        stdev (float, optional): Set the standard deviation for normal distribution to generate noise.
    """

    def __init__(self, 
                 img_data, 
                 device, 
                 dataset_device,
                 stdev=None):
        """
        The constructor of Multimodal Dataset class. Initializes the images, joints, and transformation.

        Args:
            images (numpy array): The images data, expected to be a 5D array [data_num, seq_num, channel, height, width].
            joints (numpy array): The joints data, expected to be a 3D array [data_num, seq_num, joint_dim].
            stdev (float, optional): The standard deviation for the normal distribution to generate noise. Defaults to 0.02.
        """
        self.device = device
        self.dataset_device = dataset_device
        self.stdev = stdev
        self.img_data = torch.Tensor(img_data).to(self.device)
        self.transform = nn.Sequential( #transforms.RandomErasing(),
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
            dataset (list): A list containing lists of transformed and noise added image and joint (x_img, x_joint) and the original image and joint (yi, y_joint).
        """
        
        xi = self.img_data[idx]
        yi = self.img_data[idx]

        # if self.stdev is not None:
        #     xi = self.transform(yi)
        #     xi = xi + torch.normal(mean=0, std=0.02, size=xi.shape, device=self.device)
        #     xv = yv + torch.normal(mean=0, std=self.stdev, size=yv.shape, device=self.device)
        #     xp = yp +  torch.normal(mean=0, std=self.stdev, size=yp.shape, device=self.device)
        
        if self.stdev is not None:
            xi = self.transform(xi)
            xi = xi + torch.normal(mean=0, std=0.02, size=xi.shape, device=self.device)
        
        xi = xi.to(self.dataset_device)
        yi = yi.to(self.dataset_device)
        
        # print(xi.is_cuda)
        # print(xv.is_cuda)
        # print(xp.is_cuda)
        
        return [[xi,], [yi,]]
    
    


"""
plt.figure()
plt.imshow(xi[0].permute(1,2,0).detach().clone().cpu().numpy())
plt.savefig("./fig/sample_input.png")
"""


