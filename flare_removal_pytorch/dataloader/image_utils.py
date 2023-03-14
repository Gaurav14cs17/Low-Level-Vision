from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
import skimage
import skimage.morphology
import cv2
import torchvision.transforms.functional as TF
import random

'''
1e-7 = 0.0000001
'''
_EPS = 1e-7

def read_image(image_path):
    image = Image.open(image_path)
    return image


def _gaussian_kernel(kernel_size , sigma , n_channels , dtype ):
    x = torch.arange(-kernel_size//2 + 1 , kernel_size//2 +1 , dtype=dtype)
    pw = torch.pow(x , 2)
    sigma = torch.tensor(sigma)
    sigma_pow = 2*torch.pow(sigma , 2)
    val = pw /sigma_pow
    g = torch.exp(-val)
    g_norm2d = torch.pow(torch.sum(g) ,2)
    g_kernel = torch.tensordot(g , g , dims=0 ) / g_norm2d
    g_kernel = torch.unsqueeze(g_kernel , dim=-1)
    g_kernel = torch.tile(g_kernel, (1,1,n_channels))
    return torch.unsqueeze(g_kernel, dim=-1)

def apply_blur(im, sigma):
    pass

def save_image(path , image ):
    image = image * 255.0
    image = image.cpu().numpy()
    if image.shape[-1]==1:
        image = np.stack([image , image , image , image ] , axis=-1)
    image = image.astype('uint8')
    plt.imsave(path, image)


def remove_flare( combined, flare, gamma = 2.2):
    combined_image = torch.clip(combined , _EPS , 1.0)
    flare_image = torch.clip(flare , _EPS ,1.0)
    #--------------------------------------------
    combined_linear_image = torch.pow(combined_image , gamma)
    flare_linear_image = torch.pow(flare_image , gamma)
    #---------------------------------------------
    scene_linear_image = combined_linear_image - flare_linear_image
    scene_linear_image = torch.clip(scene_linear_image , _EPS,1.0)
    scene_linear_image = torch.pow(scene_linear_image , 1.0/gamma)
    return scene_linear_image


def get_highlight_mask(flare_image,threshold = 0.99):
    binary_mask = torch.mean(flare_image, dim=-1) > threshold
    mask = binary_mask.type(torch.uint8)
    return mask


def refine_mask(mask, morph_size = 0.01):
  mask_size = max(np.shape(mask))
  kernel_radius = .5 * morph_size * mask_size
  kernel = skimage.morphology.disk(np.ceil(kernel_radius))
  opened = skimage.morphology.binary_opening(mask, kernel)
  return opened

def _create_disk_kernel(kernel_size):
  x = np.arange(kernel_size) - (kernel_size - 1) / 2
  xx, yy = np.meshgrid(x, x)
  rr = np.sqrt(xx**2 + yy**2)
  kernel = np.float32(rr <= np.max(x)) + _EPS
  kernel = kernel / np.sum(kernel)
  return kernel


def remove_background(image):
    image = np.float32(np.array(image))
    _EPS = 1e-7
    rgb_max = np.max(image, (0, 1))
    rgb_min = np.min(image, (0, 1))
    image = (image - rgb_min) * rgb_max / (rgb_max - rgb_min + _EPS)
    image = torch.from_numpy(image)
    return image


def normalize_white_balance(im):
  channel_mean = torch.mean(im, dim=(-3, -2))
  max_of_mean = torch.mean(channel_mean, dim=(-3, -2, -1))
  normalized = max_of_mean * im / (channel_mean + _EPS)
  return normalized

def blend_light_source(scene_input, scene_pred):
    binary_mask = get_highlight_mask(scene_input ).numpy()
    binary_mask = np.squeeze(binary_mask , axis=-1)
    binary_mask = refine_mask(binary_mask)

    labeled = skimage.measure.label(binary_mask)
    properties = skimage.measure.regionprops(labeled)

    max_diameter = 0
    for p in properties:
        max_diameter = max(max_diameter, p['equivalent_diameter'])
    mask = np.float32(binary_mask)
    kernel_size = round(1.5 * max_diameter)

    if kernel_size > 0:
        kernel = _create_disk_kernel(kernel_size)
        mask = cv2.filter2D(mask, -1, kernel)
        mask = np.clip(mask * 3.0, 0.0, 1.0)
        mask_rgb = np.stack([mask] * 3, axis=-1)
    else:
        mask_rgb = 0
    blend = scene_input * mask_rgb + scene_pred * (1 - mask_rgb)
    return blend

class RandomGammaCorrection(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def __call__(self, image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5, 1, 2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image, self.gamma, gain=1)





if __name__ == '__main__':
    scance_image_path  = "D:/labs/LOW_LEVEL_IMAGEING/Low-Level-Vision/flare_removal_tensorflow/output/combined.png"
    flare_image_path = "D:/labs/LOW_LEVEL_IMAGEING/Low-Level-Vision/flare_removal_tensorflow/output/flare.png"
    scance_image = read_image(scance_image_path)
    scance_image = np.asarray(scance_image)
    scance_image = scance_image.astype('float32')/255.0

    flare_image = read_image(flare_image_path)
    flare_image = np.asarray(scance_image)
    #flare_image = flare_image.astype('float32') / 255.0

    #image = remove_flare(torch.tensor(scance_image) , torch.tensor(flare_image))
    flare_mask = get_highlight_mask(torch.tensor(flare_image))

    save_image("image.png",image)
    save_image("flare_mask.png",flare_mask)



