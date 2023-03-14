from typing import Callable, Dict, Mapping, Optional, Union
import tensorflow as tf
from .perceptual_loss import Vgg19


class PerceptualLoss(tf.keras.losses.Loss):
  DEFAULT_COEFFS = {'block1_conv2': 1 / 2.6,'block2_conv2': 1 / 4.8,'block3_conv2': 1 / 3.7,'block4_conv2': 1 / 5.6,'block5_conv2': 10 / 1.5,}
  def __init__(self,coeffs = None,name = 'perceptual'):
    super(PerceptualLoss, self).__init__(name=name)
    coeffs =  self.DEFAULT_COEFFS
    layers, self._coeffs = zip(*coeffs.items())
    self._model = Vgg19(tap_out_layers=layers)

  def call(self, y_true, y_pred):
    true_features = self._model(y_true)
    pred_features = self._model(y_pred)
    total_loss = tf.constant(0.0)
    print(self._coeffs)
    print(true_features)
    for ft, fp, coeff in zip(true_features, pred_features, self._coeffs):
      loss = tf.keras.losses.MAE(ft, fp)
      loss = tf.reduce_mean(loss, axis=[1, 2], keepdims=True)
      total_loss += loss * coeff
    return total_loss


class CompositeLoss(tf.keras.losses.Loss):
  def __init__(self, name = 'composite'):
    super(CompositeLoss, self).__init__(name=name)
    self.losses: Dict[tf.keras.losses.Loss, float] = {}

  def add_loss(self, loss, weight):
    if weight <= 0.0:
      raise ValueError(f'Weight must be > 0, but is {weight}.')
    if isinstance(loss, str):
      loss = loss.lower()
      loss = {'l1': 'mae', 'l2': 'mse'}.get(loss, loss)
      loss_fn = tf.keras.losses.get(loss)
    else:
      loss_fn = loss
    if loss_fn in self.losses:
      raise ValueError('The same loss already exists.')
    self.losses[loss_fn] = weight

  def call(self, y_true, y_pred):
    assert self.losses, 'At least one component loss must be added.'
    loss_sum = tf.constant(0.0)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    for loss, weight in self.losses.items():
      loss_sum = loss(y_true, y_pred) * weight + loss_sum
    return loss_sum


def get_loss(name):
  name = name.lower()
  if name == 'l2':
    return tf.keras.losses.MeanSquaredError()
  elif name == 'l1':
    return tf.keras.losses.MeanAbsoluteError()
  elif name in ['percep', 'perceptual']:
    loss_fn = CompositeLoss()
    loss_fn.add_loss(PerceptualLoss(), weight=1.0)
    # Note that `PerceptualLoss` uses [0, 255] range internally. Since our API
    # assumes [0, 1] range for input images, we actually need to scale the L1
    # loss by 255 to achieve a true 1:1 weighting.
    loss_fn.add_loss('L1', weight=255.0)
    return loss_fn
  else:
    raise ValueError(f'Unrecognized loss function name: {name}')