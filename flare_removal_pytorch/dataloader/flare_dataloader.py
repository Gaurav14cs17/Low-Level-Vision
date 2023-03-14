import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import glob
import random
import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
from image_utils import *
import torch


class Flare_dataloader(data.Dataset):
    def __init__(self, image_path, transform_base=None, transform_flare=None, mask_type=None):
        self.ext = ['png', 'jpeg', 'jpg', 'bmp', 'tif']
        self.data_list = []
        [self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
        self.flare_dict = {}
        self.flare_list = []
        self.flare_name_list = []
        self.reflective_flag = False
        self.reflective_dict = {}
        self.reflective_list = []
        self.reflective_name_list = []
        self.mask_type = mask_type  # It is a str which may be None,"luminance" or "color"
        self.transform_base = transform_base
        self.transform_flare = transform_flare
        print("Base Image Loaded with examples:", len(self.data_list))

    def image_processing_method(self):
        self.random_gamma = np.random.uniform(1.8, 2.2)
        self.adjust_gamma = RandomGammaCorrection(self.random_gamma)
        self.adjust_gamma_reverse = RandomGammaCorrection(1 / self.random_gamma)
        self.to_tensor = transforms.ToTensor()
        self.sigma_chi = 0.01 * np.random.chisquare(df=1)
        self.color_jitter = transforms.ColorJitter(brightness=(0.8, 3), hue=0.0)
        self.gaussian_blur_transform = transforms.GaussianBlur(21, sigma=(0.1, 3.0))
        self.gain = np.random.uniform(0.5, 1.2)
        self.flare_DC_offset = np.random.uniform(-0.02, 0.02)

    def __getitem__(self, item):
        base_image_path = self.data_list[item]
        self.image_processing_method()
        base_img = self.load_base_images(base_image_path)

        reflective_img_path = None
        flare_image_path = random.choice(self.flare_list)
        if self.reflective_flag:
            reflective_img_path = random.choice(self.reflective_list)
        flare_img = self.load_flare_images(flare_image_path , reflective_img_path)
        merge_img = flare_img + base_img
        merge_img = torch.clamp(merge_img, min=0, max=1)

        if self.mask_type is None:
            return self.adjust_gamma_reverse(base_img), self.adjust_gamma_reverse(flare_img), self.adjust_gamma_reverse(merge_img), self.random_gamma
        elif self.mask_type == "luminance":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)
            luminance = 0.3 * flare_img[0] + 0.59 * flare_img[1] + 0.11 * flare_img[2]
            threshold_value = 0.99 ** self.random_gamma
            flare_mask = torch.where(luminance > threshold_value, one, zero)

        elif self.mask_type == "color":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)
            threshold_value = 0.99 ** self.random_gamma
            flare_mask = torch.where(merge_img > threshold_value, one, zero)

        return self.adjust_gamma_reverse(base_img), self.adjust_gamma_reverse(flare_img), self.adjust_gamma_reverse(merge_img), flare_mask, self.random_gamma

    def save_image(self, in_path, image):
        self.output_dir_path = "./output/"
        if not os.path.exists(self.output_dir_path):
            os.makedirs(self.output_dir_path)
        plt.imsave(in_path, image.permute(1, 2, 0).cpu().numpy())

    def load_base_images(self, base_image_path):
        base_image = Image.open(base_image_path)
        base_image_tensor = self.to_tensor(base_image)
        base_image_tensor = self.adjust_gamma(base_image_tensor)
        if self.transform_base is not None:
            base_image_tensor = self.transform_base(base_image_tensor)
        else:
            base_image_tensor = base_image_tensor.permute(2, 0, 1)
        base_image_tensor = Normal(base_image_tensor, self.sigma_chi).sample()
        base_image_tensor = self.gain * base_image_tensor
        base_image_tensor = torch.clamp(base_image_tensor, min=0, max=1)
        return base_image_tensor

    def load_flare_images(self, flare_image_path , reflective_img_path = None):
        flare_image = Image.open(flare_image_path)
        flare_img_tensor = self.to_tensor(flare_image)
        flare_img_tensor = self.adjust_gamma(flare_img_tensor)

        if self.reflective_flag:
            print(self.reflective_flag)
            reflective_img = Image.open(reflective_img_path)
            reflective_img = self.to_tensor(reflective_img)
            reflective_img = self.adjust_gamma(reflective_img)
            flare_img_tensor = torch.clamp(flare_img_tensor + reflective_img, min=0, max=1)

        flare_img_tensor = remove_background(flare_img_tensor)
        if self.transform_flare is not None:
            flare_img_tensor = self.transform_flare(flare_img_tensor)
        # change_color
        flare_image = self.color_jitter(flare_img_tensor)
        flare_image = self.gaussian_blur_transform(flare_image)
        flare_image = flare_image + self.flare_DC_offset
        flare_image = torch.clamp(flare_image, min=0, max=1)
        return flare_image

    def __len__(self):
        return len(self.data_list)

    def load_scattering_flare(self, flare_name, flare_path):
        self.flare_name_list.append(flare_name)
        for file_name in os.listdir(flare_path):
            dir_path = os.path.join(flare_path , file_name)
            flare_list = []
            [flare_list.extend(glob.glob(dir_path + '/*.' + e)) for e in self.ext]
            self.flare_dict[file_name] = flare_list
            self.flare_list.extend(flare_list)
        len_flare_list = len(self.flare_list)
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:", flare_name, " is loaded successfully with examples", str(len_flare_list))
        print("Now we have", len(self.flare_list), 'scattering flare images')

    def load_reflective_flare(self, reflective_name, reflective_path):
        self.reflective_flag = True
        reflective_list = []
        [reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name] = reflective_list
        self.reflective_list.extend(reflective_list)
        len_reflective_list = len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            print("Reflective Flare Image:", reflective_name, " is loaded successfully with examples",str(len_reflective_list))
        print("Now we have", len(self.reflective_list), 'refelctive flare images')


if __name__ == '__main__':
    import os
    data_path = "D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/"
    output_path = "D:/labs/LOW_LEVEL_IMAGEING/data/output/"
    from matplotlib import pyplot as plt

    transform_base = transforms.Compose([transforms.RandomCrop((512, 512), pad_if_needed=True, padding_mode='reflect'),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip()
                                         ])

    transform_flare = transforms.Compose([transforms.RandomAffine(degrees=(0, 360), scale=(0.8, 1.5),
                                                                  translate=(300 / 1440, 300 / 1440), shear=(-20, 20)),
                                          transforms.CenterCrop((512, 512)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip()
                                          ])

    flare_image_loader = Flare_dataloader(data_path + 'Flickr24K', transform_base, transform_flare , "luminance")
    flare_image_loader.load_scattering_flare('Flare7K', "D:/labs/LOW_LEVEL_IMAGEING/data/Flare_Removal_DataSet/lens-flare/")
    #flare_image_loader.load_reflective_flare('Flare7K', data_path + 'Flare7k/Reflective_Flare')

    print(len(flare_image_loader))
    if not os.path.exists(os.path.join(output_path, "GT")):
        os.makedirs(os.path.join(output_path, "GT"))
        os.makedirs(os.path.join(output_path, "IN"))

    for img_index in range(len(flare_image_loader)):
        test_base_img, test_flare_img, test_merge_img, flare_mask_img , gamma = flare_image_loader[img_index]
        test_base_img_path = os.path.join(output_path, "GT", str(img_index) + "_test_base_img"+ ".png")
        test_flare_img_path = os.path.join(output_path, "GT", str(img_index) + "_test_flare_img" + ".png")
        test_merge_img_path = os.path.join(output_path, "GT", str(img_index) + "_test_merge_img" + ".png")
        flare_mask_img_path = os.path.join(output_path, "GT", str(img_index) + "_flare_mask_img" + ".png")

        plt.imsave(test_base_img_path, test_base_img.permute(1, 2, 0).cpu().numpy())
        plt.imsave(test_flare_img_path, test_flare_img.permute(1, 2, 0).cpu().numpy())
        plt.imsave(test_merge_img_path, test_merge_img.permute(1, 2, 0).cpu().numpy())
        plt.imsave(flare_mask_img_path, flare_mask_img.permute(1, 2, 0).cpu().numpy())
        print("done: ", img_index)
        if img_index >22:
            break

