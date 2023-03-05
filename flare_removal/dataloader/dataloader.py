import os.path
from typing import Iterable, Tuple, Union
import tensorflow as tf
_SHUFFLE_BUFFER_SIZE = 10_000


def image_dataset_from_files(data_dir,image_shape,batch_size = 0,shuffle = True,repeat = -1):
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


def image_dataset_from_tfrecords(globs,tag,image_shape,batch_size = 0,shuffle = True,repeat = -1):

  files = tf.data.Dataset.list_files(globs, shuffle, seed=0)
  examples = files.interleave(tf.data.TFRecordDataset,num_parallel_calls=tf.data.AUTOTUNE,deterministic=not shuffle)
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


def get_scene_dataset(path,source,batch_size,input_shape = (640, 640, 3),repeat = 0):
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


def get_flare_dataset(path,source,batch_size,input_shape = (752, 1008, 3),repeat = -1):
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
'''