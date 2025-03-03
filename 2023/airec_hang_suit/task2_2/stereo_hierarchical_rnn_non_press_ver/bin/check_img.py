import sys
import numpy as np
import matplotlib.pylab as plt
import torch.nn as nn
from torchvision import transforms

sys.path.append("/home/fujita/work/eipl")
from eipl.utils import normalization, deprocess_img, tensor2numpy

def adjust_image(image):
    image = transforms.ToPILImage()(image)
    image = transforms.functional.adjust_brightness(image, 1.3)
    image = transforms.functional.adjust_saturation(image, 1.5)
    image = transforms.functional.adjust_contrast(image, 1.0)
    image = np.array(image)
    return image

def generate_image(img):
    new_img = np.zeros_like(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i,j] = adjust_image(img[i,j])

    return new_img

dir_name = "/home/fujita/job/2023/airec_hang_suit/rosbag/HangSuit_task2_2"

img = np.load(f'{dir_name}/train/right_img.npy')
new_img = generate_image(img)
import ipdb; ipdb.set_trace()
# np.save('../data/task2/train/images_right_adjust.npy', new_img)

img = np.load(f'{dir_name}/test/images_right.npy')
new_img = generate_image(img)
# np.save('../data/task2/test/images_right_adjust.npy', new_img)

plt.imshow(new_img[0,0,:,:,::-1])
plt.show()