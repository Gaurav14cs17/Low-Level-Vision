import os.path
from typing import Iterable, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloader.synthesis import *
from dataloader.utils import *

_SHUFFLE_BUFFER_SIZE = 10_000

from tensorflow.keras.utils import img_to_array





def image_dataset_from_files(data_dir,
                             image_shape,
                             batch_size = 1,
                             shuffle = True,
                             repeat = -1):
  """Loads images from individual JPG or PNG files.
  Args:
    data_dir: Parent directory where input images are located. All JPEG and PNG
      files under this directory (either directly or indirectly) will be
      included.
    image_shape: Shape of the images in (H, W, C) format.
    batch_size: 0 means images are not batched. Positive values define the batch
      size. The batched images have shape (B, H, W, C).
    shuffle: Whether to randomize the order of the images.
    repeat: 0 means the dataset is not repeated. -1 means it's repeated
      indefinitely. A positive value means it's repeated for the specified
      number of times (epochs).
  Returns:
    A Dataset object containing (H, W, C) or (B, H, W, C) image tensors.
  """
  extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
  # Images directly under the given directory.
  globs = [os.path.join(data_dir, f'*.{e}') for e in extensions][:10]
  # Images under subdirectories.
  globs += [os.path.join(data_dir, '**', f'*.{e}') for e in extensions]
  files = tf.data.Dataset.list_files(globs, shuffle, seed=0)

  def _parser(file_name):
    blob = tf.io.read_file(file_name)
    image = tf.io.decode_image(blob, dtype=tf.float32)
    image.set_shape = image_shape
    print("data_dir: ", data_dir,image.shape)
    return image

  images = files.map(_parser, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle)

  if repeat < 0:
    images = images.repeat()
  elif repeat > 0:
    images = images.repeat(repeat)

  if batch_size > 0:
    images = images.batch(batch_size, drop_remainder=True)
  #images = images.prefetch(tf.data.AUTOTUNE)
  return images





def save_image(image , name , use_one_dim = False ):
    if use_one_dim==False:
        plt.imsave(name ,image.numpy())
    else:
        image = tf.stack([image.numpy() , image.numpy() , image.numpy() ] , axis=-1)
        plt.imsave(name, image)




if __name__ == '__main__':
    scence_path = "D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/Flickr24K/"
    flare_path = "D:/labs/LOW_LEVEL_IMAGEING/data/Flare7K_dataset/Flare7k/Scattering_Flare/Compound_Flare/"
    scenes = image_dataset_from_files(scence_path, (640, 640, 3))
    flares = image_dataset_from_files(flare_path, (640, 640, 3))

    for scene, flare in tf.data.Dataset.zip((scenes, flares)):

        scene, flare, combined, gamma = add_flare(scene,flare,flare_max_gain=10.0,noise=0.01,training_res=512)
        print(gamma)

        print(scene.shape)


        write_image(scene[0] , "./output/scene.png")
        write_image(flare[0], "./output/flare.png")
        write_image(combined[0], "./output/combined.png")

        pred_flare = remove_flare(combined, flare , gamma)
        flare_mask = get_highlight_mask(flare )

        print(flare_mask.shape)

        write_image(pred_flare[0], "./output/pred_flare.png")
        write_image(flare_mask[0], "./output/flare_mask.png")
        break



