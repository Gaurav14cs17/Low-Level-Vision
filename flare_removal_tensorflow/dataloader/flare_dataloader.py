import os
import numpy as np
import tensorflow as tf
from utils import *
import math


class Flare_dataloader:
    def __init__(self, dir_path=""):
        self.image_shape = (640, 640, 3)
        self.dir_path = dir_path
        self.images_list = self.get_images_name()
        self.IN_dir_path, self.GT_dir_path = self.get_in_and_gt_dir()
        self.noise_value = 0.1
        self.max_gain = 10.0
        self.batch_size = 4

    def get_images_name(self):
        return os.listdir(os.path.join(self.dir_path, "GT"))

    def load_image(self, file_path):
        blob = tf.io.read_file(file_path)
        image = tf.io.decode_image(blob, dtype=tf.float32)
        print(file_path)
        image.set_shape(self.image_shape)
        image = tf.expand_dims(image, 0)
        return image

    def get_in_and_gt_dir(self):
        IN_images_list = []
        GT_images_list = []
        for image_name in self.images_list:
            gt_name = "GT_" + image_name.split('_')[-1]
            in_name = "IN_" + image_name.split('_')[-1]
            gt_full_path = os.path.join(self.dir_path, "GT", gt_name)
            in_full_path = os.path.join(self.dir_path, "IN", in_name)
            IN_images_list.append(gt_full_path)
            GT_images_list.append(in_full_path)
        return IN_images_list, GT_images_list

    def apply_affine(self, flare_linear, batch_size):
        rotation = tf.random.uniform([batch_size], minval=-math.pi, maxval=math.pi)
        shift = tf.random.normal([batch_size, 2], mean=0.0, stddev=10.0)
        shear = tf.random.uniform([batch_size, 2], minval=-math.pi / 9, maxval=math.pi / 9)
        scale = tf.random.uniform([batch_size, 2], minval=0.9, maxval=1.2)
        flare_linear = apply_affine_transform(flare_linear, rotation=rotation, shift_x=shift[:, 0],
                                              shift_y=shift[:, 1], shear_x=shear[:, 0], shear_y=shear[:, 1],
                                              scale_x=scale[:, 0], scale_y=scale[:, 1])
        return flare_linear

    def add_flare(self, scene, flare, noise, flare_max_gain=10.0, apply_affine=False, training_res=512):
        batch_size , flare_input_height, flare_input_width,_  = flare.shape
        gamma = tf.random.uniform([], 1.8, 2.2)
        flare_linear = tf.image.adjust_gamma(flare, gamma)
        flare_linear = remove_background(flare_linear)

        if apply_affine:
            flare_linear = self.apply_affine(flare_linear, batch_size)

        flare_linear = tf.clip_by_value(flare_linear, 0.0, 1.0)
        flare_linear = tf.image.crop_to_bounding_box(flare_linear,
                                                     offset_height=(flare_input_height - training_res) // 2,
                                                     offset_width=(flare_input_width - training_res) // 2,
                                                     target_height=training_res, target_width=training_res)
        flare_linear = tf.image.random_flip_left_right(tf.image.random_flip_up_down(flare_linear))
        # First normalize the white balance. Then apply random white balance.
        flare_linear = normalize_white_balance(flare_linear)
        rgb_gains = tf.random.uniform([3], 0, flare_max_gain, dtype=tf.float32)
        flare_linear *= rgb_gains

        # Further augmentation on flare patterns: random blur and DC offset.
        blur_size = tf.random.uniform([], 0.1, 3)
        flare_linear = apply_blur(flare_linear, blur_size)
        offset = tf.random.uniform([], -0.02, 0.02)
        flare_linear = tf.clip_by_value(flare_linear + offset, 0.0, 1.0)
        flare_srgb = tf.image.adjust_gamma(flare_linear, 1.0 / gamma)

        # Scene augmentation: random crop and flips.
        scene_linear = tf.image.adjust_gamma(scene, gamma)
        scene_linear = tf.image.random_crop(scene_linear, flare_linear.shape)
        scene_linear = tf.image.random_flip_left_right(tf.image.random_flip_up_down(scene_linear))

        # Additive Gaussian noise. The Gaussian's variance is drawn from a Chi-squared
        # distribution. This is equivalent to drawing the Gaussian's standard
        # deviation from a truncated normal distribution, as shown below.
        sigma = tf.abs(tf.random.normal([], 0, noise))
        noise = tf.random.normal(scene_linear.shape, 0, sigma)
        scene_linear += noise

        # Random digital gain.
        gain = tf.random.uniform([], 0, 1.2)  # varying the intensity scale
        scene_linear = tf.clip_by_value(gain * scene_linear, 0.0, 1.0)
        scene_srgb = tf.image.adjust_gamma(scene_linear, 1.0 / gamma)
        # Combine the flare-free scene with a flare pattern to produce a synthetic
        # training example.
        combined_linear = scene_linear + flare_linear
        combined_srgb = tf.image.adjust_gamma(combined_linear, 1.0 / gamma)
        combined_srgb = tf.clip_by_value(combined_srgb, 0.0, 1.0)
        return quantize_8(scene_srgb), quantize_8(flare_srgb), quantize_8(combined_srgb), gamma

    def get_in_and_gt_image(self, scene_path, flare_path):
        scene_image = self.load_image(scene_path)
        flare_image = self.load_image(flare_path)
        scene_srgb_image , flare_srgb_image , combined_srgb_image , gamma_value = self.add_flare(scene_image , flare_image , self.noise_value , self.max_gain )
        print(" tf.Tensor(2.1994658, shape=(), dtype=float32) ",gamma_value)
        return scene_srgb_image , flare_srgb_image , combined_srgb_image


    # def get_in_and_gt(self):
    #     dataset_in = tf.data.Dataset.from_tensor_slices(self.IN_dir_path)
    #     dataset_gt = tf.data.Dataset.from_tensor_slices(self.GT_dir_path)
    #     dataset = tf.data.Dataset.zip((dataset_in, dataset_gt))
    #     dataset = dataset.map(lambda path_in_image, path_gt_image: self.get_in_and_gt_image(path_in_image, path_gt_image),num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #     dataset = dataset.unbatch()
    #     dataset = dataset.batch(self.batch_size, drop_remainder=True)
    #     #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #     return dataset


    def __len__(self):
        return int(len(self.images_list))


if __name__ == '__main__':
    gen = Flare_dataloader(dir_path="D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/train/")
    for  in_path , gt_path  in zip(gen.IN_dir_path , gen.GT_dir_path):
        data = gen.get_in_and_gt_image(in_path , gt_path)

    #     # write_image(scene_srgb_image[0], "./output/scene.png")
    #     # write_image(flare_srgb_image[0], "./output/flare.png")
    #     # pred_flare = remove_flare(flare_srgb_image, scene_srgb_image , gamma_value)
    #     # flare_mask = get_highlight_mask(flare_srgb_image)
    #     #
    #     # print(flare_mask.shape)
    #     #
    #     # write_image(pred_flare[0], "./output/pred_flare.png")
    #     # write_image(flare_mask[0], "./output/flare_mask.png")
    #     break
