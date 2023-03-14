import numpy as np
import tensorflow as tf


class Vgg19(tf.keras.Model):
  """A modified VGG-19 network with configurable tap-outs.
  The network is modified such that all max pooling are replaced by average
  pooling.
  Supported layers and their output shapes are:
  - block1_conv1 .. 2: [B,    H,    W,  64]
  - block1_pool:       [B,  H/2,  W/2,  64]
  - block2_conv1 .. 2: [B,  H/2,  W/2, 128]
  - block2_pool:       [B,  H/4,  W/4, 128]
  - block3_conv1 .. 4: [B,  H/4,  W/4, 256]
  - block3_pool:       [B,  H/8,  W/8, 256]
  - block4_conv1 .. 4: [B,  H/8,  W/8, 512]
  - block4_pool:       [B, H/16, W/16, 512]
  - block5_conv1 .. 4: [B, H/16, W/16, 512]
  - block5_pool:       [B, H/32, W/32, 512]
  where [B, H, W, 3] is the batched input image tensor.
  """

  def __init__(self,tap_out_layers,trainable = False,weights = 'imagenet'):
    super(Vgg19, self).__init__(name='vgg19')
    if len(set(tap_out_layers)) != len(tap_out_layers):
      raise ValueError(f'There are duplicates in the provided layers: 'f'{tap_out_layers}')
    model = tf.keras.applications.VGG19(include_top=False, weights=weights)
    # Replace max pooling by average pooling according to the following paper:
    #   Zhang et al., Single Image Reflection Removal with Perceptual Losses,
    #   CVPR 2018.
    model = self._replace_max_by_average_pool(model)
    model.trainable = trainable

    # Configure tap-out layers. Note that we need `layer.get_output_at(1)` below
    # to use the modified graph (at node 1) with average pooling. The
    # `layer.output` attribute will default to node 0, which is the unmodified
    # model.
    invalid_layers = set(tap_out_layers) - set(l.name for l in model.layers)
    if invalid_layers:
      raise ValueError(f'Unrecognized layers: {invalid_layers}')
    tap_outs = [model.get_layer(l).get_output_at(1) for l in tap_out_layers]
    self._model = tf.keras.Model(inputs=model.inputs, outputs=tap_outs)

  def call(self, images, **kwargs):
    x = tf.keras.applications.vgg19.preprocess_input(images * 255.0)
    return self._model(x)

  @staticmethod
  def _replace_max_by_average_pool(model):
    """Replaces MaxPooling2D layers in a model with AveragePooling2D."""
    input_layer, *other_layers = model.layers
    if not isinstance(input_layer, tf.keras.layers.InputLayer):
      raise ValueError('The first layer should be InputLayer, but is:',input_layer)
    x = input_layer.output
    for layer in other_layers:
      if isinstance(layer, tf.keras.layers.MaxPooling2D):
        layer = tf.keras.layers.AveragePooling2D(pool_size=layer.pool_size,strides=layer.strides,padding=layer.padding,data_format=layer.data_format, name=layer.name,)
      x = layer(x)
    return tf.keras.models.Model(inputs=input_layer.input, outputs=x)


class IdentityInitializer(tf.keras.initializers.Initializer):
  def __call__(self,shape,dtype = tf.float32,**kwargs):
    array = np.zeros(shape, dtype=dtype.as_numpy_dtype)
    kernel_height, kernel_width, in_channels, out_channels = shape
    cy, cx = kernel_height // 2, kernel_width // 2
    for i in range(np.minimum(in_channels, out_channels)):
      array[cy, cx, i, i] = 1
    return tf.constant(array)


class _CanBlock(tf.keras.layers.Layer):
  """A convolutional block in the context aggregation network."""
  def __init__(self, channels, size, rate, **kwargs):
    super(_CanBlock, self).__init__(**kwargs)
    self.channels = channels
    self.size = size
    self.rate = rate

  def build(self, input_shape):
    self.conv = tf.keras.layers.Conv2D(filters=self.channels,kernel_size=self.size,dilation_rate=self.rate,
        padding='same',use_bias=False,kernel_initializer=IdentityInitializer(),input_shape=input_shape)

    self.w0 = self.add_weight('w0',dtype=tf.float32,initializer=tf.keras.initializers.Constant(1.0),trainable=True)
    self.w1 = self.add_weight('w1',dtype=tf.float32,initializer=tf.keras.initializers.Constant(0.0),trainable=True)
    self.batch_norm = tf.keras.layers.BatchNormalization(scale=False)
    self.activation = tf.keras.layers.LeakyReLU(0.2)

  def call(self, inputs):
    convolved = self.conv(inputs)
    normalized = self.w0 * convolved + self.w1 * self.batch_norm(convolved)
    outputs = self.activation(normalized)
    return outputs





class Vgg_model(tf.keras.Model):
    def __init__(self  , conv_channels=64,out_channels=3,name='can'):
        super(Vgg_model, self).__init__()
        self.vgg = Vgg19( tap_out_layers=[f'block{i}_conv2' for i in range(1, 6)], trainable=False)
        self.CanBlock_layer = _CanBlock(conv_channels, size=1, rate=1, name=f'{name}_g_conv0')

        self.CanBlock_layer_list = []
        for i, rate in enumerate([1, 2, 4, 8, 16, 32, 64, 1]):
            self.CanBlock_layer_list.append(_CanBlock(conv_channels, size=3, rate=rate, name=f'{name}_g_conv{i + 1}'))

        self.output_layer = tf.keras.layers.Conv2D(out_channels,kernel_size=1,dilation_rate=1,padding='same',use_bias=False,name=f'{name}_g_conv_last')

    def call(self , input_layer ):
        features = self.vgg(input_layer)
        print(features)
        features = [tf.image.resize(f, input_layer.shape[1:3]) / 255.0 for f in features]
        x = tf.concat([input_layer] + features, axis=-1)
        print(x)
        x = self.CanBlock_layer(x)
        for _CanBlock  in self.CanBlock_layer_list :
            x = _CanBlock(x)

        output = self.output_layer(x)
        return output





if __name__ == '__main__':
    input_shape = (512, 512, 3)
    input_layer = tf.keras.Input(shape=input_shape, name='input')
    print(input_layer.shape)
    model  = Vgg_model()
    # model.build(input_shape=(None, 224, 224, 3))
    # model.summary()

    output = model(input_layer)
    print(output.shape)
