import os.path
from typing import Iterable, Tuple, Union
import tensorflow as tf

_SHUFFLE_BUFFER_SIZE = 10_000


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self):
        self.dir_path = ""
        self.bit_depth = 8
        self.input_shape=(640, 640, 3)
        self.batch_size = 3
        self.images_dir_list = self.get_images_name()
        self.on_epoch_end()

    def get_images_name(self):
        return os.listdir(os.path.join(self.dir_path , "GT"))

    def load_images(self, path):
        path = tf.io.read_file(path)
        image = tf.io.decode_image(path, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)
        image = image / (2 ** self.bit_depth - 1)
        image = tf.expand_dims(image, 0)
        return image

    def get_random_crop(self, value, size, seed=None, name=None):
        tf_value = tf.convert_to_tensor(value, name="value")
        tf_size = tf.convert_to_tensor(size, dtype=tf.int32, name='size')
        value_shape = tf.shape(tf_value)
        offset = tf.random.uniform(tf.shape(value_shape), dtype=tf_size.dtype, maxval=tf_size.dtype.max, seed=seed) % (value_shape - tf_size + 1)
        offset_floor_mod = tf.math.floormod(offset, tf.stack([1, 2, 2, 1]))
        diff = offset - offset_floor_mod
        crop_value = tf.slice(value, diff, tf_size, name=name)
        return crop_value

    def get_patch(self , gt_image, in_image , num_patches , patch_size , gt_filter , in_filter):
        patches_list = list()
        image_cat = tf.concat([gt_image , in_image] ,-1)
        for _ in range(num_patches):
            crop_image = self.get_random_crop(image_cat ,[1,patch_size,patch_size , in_filter + gt_filter])
            patches_list.append(crop_image)
        all_crop_image = tf.concat(patches_list ,0)
        return all_crop_image

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.images_dir_list):
            self.batch_size = len(self.images_dir_list) - index * self.batch_size

        files_batch = self.images_dir_list[index * self.batch_size: (index + 1) * self.batch_size]
        IN_images = []
        GT_images = []
        for image_name in files_batch:
            gt_name = "GT"+ image_name.split('_')[-1]
            in_name = "IN" + image_name.split('_')[-1]
            gt_full_path = os.path.join(self.dir_path , "GT" , gt_name)
            in_full_path = os.path.join(self.dir_path , "IN" , in_name)
            in_image = self.load_images(in_full_path)
            gt_image = self.load_images(gt_full_path)

            GT_image , IN_image = self.get_patch(gt_image , in_image , num_patches=4 , patch_size=256 , gt_filter=3 , in_filter=3)
            for x , y in zip(IN_image , GT_image):
                IN_images.append(x)
                IN_images.append(y)
        return IN_images, IN_images

    def __len__(self):
        return int(len(self.images_dir_list)/float(self.batch_size))

    def on_epoch_end(self):
        pass







def image_dataset_from_files(data_dir, image_shape, batch_size=0, shuffle=True, repeat=-1):
    extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    globs = [os.path.join(data_dir, f'*.{e}') for e in extensions]
    globs += [os.path.join(data_dir, '**', f'*.{e}') for e in extensions]
    files = tf.data.Dataset.list_files(globs, shuffle, seed=0)

    def _parser(file_name):
        blob = tf.io.read_file(file_name)
        image = tf.io.decode_image(blob, dtype=tf.float32)
        image.set_shape(image_shape)
        return image

    images = files.map(_parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    if repeat < 0:
        images = images.repeat()
    elif repeat > 0:
        images = images.repeat(repeat)
    if batch_size > 0:
        images = images.batch(batch_size, drop_remainder=True)
    images = images.prefetch(tf.data.AUTOTUNE)
    return images


def image_dataset_from_tfrecords(globs, tag, image_shape, batch_size=0, shuffle=True, repeat=-1):
    files = tf.data.Dataset.list_files(globs, shuffle, seed=0)
    examples = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    if shuffle:
        examples = examples.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE, seed=0, reshuffle_each_iteration=True)

    def _parser(example):
        features = tf.io.parse_single_example(example, features={tag: tf.io.FixedLenFeature([], tf.string)})
        image_u8 = tf.reshape(tf.io.decode_raw(features[tag], tf.uint8), image_shape)
        image_f32 = tf.image.convert_image_dtype(image_u8, tf.float32)
        return image_f32

    images = examples.map(_parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)
    if repeat < 0:
        images = images.repeat()
    elif repeat > 0:
        images = images.repeat(repeat)
    if batch_size > 0:
        images = images.batch(batch_size, drop_remainder=True)
    images = images.prefetch(tf.data.AUTOTUNE)
    return images


def get_scene_dataset(path, source, batch_size, input_shape=(640, 640, 3), repeat=0):
    """Returns scene images according to configuration."""
    if source == 'tfrecord':
        return image_dataset_from_tfrecords(
            globs=os.path.join(path, '*.tfrecord'),
            tag='image',
            image_shape=input_shape,
            batch_size=batch_size,
            repeat=repeat)

    elif source == 'jpg':
        return image_dataset_from_files(
            data_dir=path,
            image_shape=input_shape,
            batch_size=batch_size,
            repeat=repeat)

    else:
        raise ValueError('Unrecognized data source', source)


def get_flare_dataset(path, source, batch_size, input_shape=(752, 1008, 3), repeat=-1):
    """Returns flare images according to configuration."""
    if source == 'tfrecord':
        return image_dataset_from_tfrecords(
            globs=path,
            tag='flare',
            image_shape=input_shape,
            batch_size=batch_size,
            repeat=repeat)

    elif source == 'jpg':
        return image_dataset_from_files(
            data_dir=path,
            image_shape=input_shape,
            batch_size=batch_size,
            repeat=repeat)

    else:
        raise ValueError('Unrecognized data source', source)


'''


 --scene_dir=/path/to/flare-free/
  --flare_dir=/path/to/flare-only/
  
  
   scenes = data_provider.get_scene_dataset(
      FLAGS.scene_dir, FLAGS.data_source, FLAGS.batch_size, repeat=FLAGS.epochs)
  flares = data_provider.get_flare_dataset(FLAGS.flare_dir, FLAGS.data_source,
                                           FLAGS.batch_size)
'''
