from torchvision import transforms
import torch.nn as nn
import numpy as np
from eipl.utils import normalization, deprocess_img, tensor2numpy
import matplotlib.pylab as plt

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



img = np.load('../data/task2/train/images_right.npy')
new_img = generate_image(img)
np.save('../data/task2/train/images_right_adjust.npy', new_img)

img = np.load('../data/task2/test/images_right.npy')
new_img = generate_image(img)
np.save('../data/task2/test/images_right_adjust.npy', new_img)

plt.imshow(new_img[0,0,:,:,::-1])
plt.show()