import os.path
from typing import Iterable, Tuple, Union
import tensorflow as tf
from synthesis import *
from utils import *
from random import shuffle
import random

_SHUFFLE_BUFFER_SIZE = 10_000


class DataGenerator():
    def __init__(self, dir_path=""):
        self.dir_path = dir_path
        self.bit_depth = 8
        self.input_shape = (640, 640, 4)
        self.batch_size = 3
        self.images_dir_list = self.get_images_name()


    def get_images_name(self):
        return os.listdir(os.path.join(self.dir_path, "GT"))

    def load_images(self, path):
        path = tf.io.read_file(path)
        image = tf.io.decode_image(path, dtype=tf.float32)
        # image.set_shape(self.input_shape)
        image = tf.cast(image, dtype=tf.float32)
        #image = image / (2 ** self.bit_depth - 1)
        image = tf.expand_dims(image, 0)
        return image

    def get_in_and_gt_image(self, in_path, gt_path):
        return self.load_images(in_path), self.load_images(gt_path)

    def get_random_crop(self, value, size, seed=None, name=None):
        tf_value = tf.convert_to_tensor(value, name="value")
        tf_size = tf.convert_to_tensor(size, dtype=tf.int32, name='size')
        value_shape = tf.shape(tf_value)
        offset = tf.random.uniform(tf.shape(value_shape), dtype=tf_size.dtype, maxval=tf_size.dtype.max, seed=seed) % (
                    value_shape - tf_size + 1)
        offset_floor_mod = tf.math.floormod(offset, tf.stack([1, 2, 2, 1]))
        diff = offset - offset_floor_mod
        crop_value = tf.slice(value, diff, tf_size, name=name)
        return crop_value

    def get_patch(self, in_image, gt_image, num_patches, patch_size, gt_filter, in_filter):
        patches_list = list()
        image_cat = tf.concat([in_image, gt_image], -1)
        for _ in range(num_patches):
            crop_image = self.get_random_crop(image_cat, [1, patch_size, patch_size, in_filter + gt_filter])
            patches_list.append(crop_image)
        all_crop_image = tf.concat(patches_list, 0)
        return all_crop_image


    def split_in_and_gt_images(self , x):
        return tf.split(x,[3 , 3 ] , axis=-1)

    def get_data(self):
        IN_images_list = []
        GT_images_list = []
        print(self.images_dir_list)
        indx = random.randrange(0,len(self.images_dir_list))
        image_name_list = [self.images_dir_list[indx]]
        print(indx)
        for image_name in image_name_list:
            gt_name = "GT_" + image_name.split('_')[-1]
            in_name = "IN_" + image_name.split('_')[-1]
            gt_full_path = os.path.join(self.dir_path, "GT", gt_name)
            in_full_path = os.path.join(self.dir_path, "IN", in_name)
            IN_images_list.append(gt_full_path)
            GT_images_list.append(in_full_path)

        dataset_gt = tf.data.Dataset.from_tensor_slices(GT_images_list)
        dataset_in = tf.data.Dataset.from_tensor_slices(IN_images_list)
        dataset = tf.data.Dataset.zip((dataset_in, dataset_gt))
        dataset = dataset.map(lambda path_in_image, path_gt_image: self.get_in_and_gt_image(path_in_image, path_gt_image), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #dataset = dataset.map(lambda IN_image, GT_image: self.get_patch(IN_image, GT_image, num_patches=4, patch_size=256, gt_filter=3,in_filter=3) ,  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()
        #dataset = dataset.map(lambda x : tf.cast( x , dtype=tf.float32) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self.batch_size , drop_remainder=True)
        #dataset = dataset.map(lambda y : self.split_in_and_gt_images(y) , num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(dataset)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def __len__(self):
        return int(len(self.images_dir_list) / float(self.batch_size))






if __name__ == '__main__':
    gen = DataGenerator(dir_path="D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/train/")
    dataset = gen.get_data()
    for flare, scene in dataset:
        #scene, flare, combined, gamma = add_flare(scene, flare, flare_max_gain=10.0, noise=0.01, training_res=512)

        print(scene)

        write_image(scene[0], "./output/scene.png")
        write_image(flare[0], "./output/flare.png")


        pred_flare = remove_flare(flare, scene)
        flare_mask = get_highlight_mask(flare)

        print(flare_mask.shape)

        write_image(pred_flare[0], "./output/pred_flare.png")
        write_image(flare_mask[0], "./output/flare_mask.png")
        break





