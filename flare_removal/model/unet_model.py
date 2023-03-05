"""
Reference:
  Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for
  Biomedical Image Segmentation, MICCAI 2015.
  https://doi.org/10.1007/978-3-319-24574-4_28
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf




class down_block(tf.keras.Model):
    def __init__(self, depth, name_prefix=" down"):
        super(down_block, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu',
                                                 name=f'{name_prefix}_conv1')
        self.skip_layer = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu',
                                                 name=f'{name_prefix}_conv2')
        self.down_2x_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name=f'{name_prefix}_pool')

    def call(self, x):
        conv_output = self.conv_layer(x)
        skip_output = self.skip_layer(conv_output)
        down_output = self.down_2x_layer(skip_output)
        return skip_output, down_output


class up_block(tf.keras.Model):
    def __init__(self, depth, interpolation='bilinear', name_prefix="up"):
        super(up_block, self).__init__()
        self.up_2x_layer_1 = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation=interpolation,
                                                          name=f'{name_prefix}_2x')
        self.up_2x_layer_2 = tf.keras.layers.Conv2D(filters=depth, kernel_size=2, padding='same', activation='relu',
                                                    name=f'{name_prefix}_2xconv')

        self.conv_layer_1 = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu',
                                                   name=f'{name_prefix}_conv1')
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=depth, kernel_size=3, padding='same', activation='relu',
                                                   name=f'{name_prefix}_conv2')

    def call(self, x_output, skip_output, name_prefix="up"):
        up_2x = self.up_2x_layer_1(x_output)
        up_2x = self.up_2x_layer_2(up_2x)
        concat = tf.keras.layers.concatenate([up_2x, skip_output], name=f'{name_prefix}_concat')
        conv = self.conv_layer_1(concat)
        conv = self.conv_layer_2(conv)
        return conv


class UNet_model(tf.keras.Model):
    def __init__(self, input_shape  =3  , scales=4, bottleneck_depth=1024, number_bottleneck_layers=2):
        super(UNet_model, self).__init__()
        depths = [bottleneck_depth // 2 ** i for i in range(scales, 0, -1)]
        self.down_layers = []
        for depth in depths:
            self.down_layers.append(down_block(depth, name_prefix=f'down{depth}'))

        self.bottleneck_layers = []
        for i in range(number_bottleneck_layers):
            self.bottleneck_layers.append(
                tf.keras.layers.Conv2D(filters=bottleneck_depth, kernel_size=3, padding='same', activation='relu',
                                       name=f'bottleneck_conv{i + 1}'))

        self.up_layers = []
        for depth in depths:
            self.up_layers.append(up_block(depth, name_prefix=f'up{depth}'))

        self.output_layer = tf.keras.layers.Conv2D(filters=input_shape,kernel_size=1,activation='sigmoid',name='output')

    def call(self, input_layer):
        previous_output = input_layer
        skips = []
        for _down_block in self.down_layers:
            skip, previous_output = _down_block(previous_output)
            skips.append(skip)

        for bottleneck_block in self.bottleneck_layers:
            previous_output = bottleneck_block(previous_output)

        for _up_block, skip in zip(reversed(self.up_layers), reversed(skips)):
            previous_output = _up_block( previous_output, skip)

        output_layer = self.output_layer(previous_output)
        return output_layer

    def model_summry(self):
        model.build(input_shape=(None, 224, 224, 3))
        model.summary()




if __name__ == '__main__':
    input_shape = (512, 512, 3)
    input_layer = tf.keras.Input(shape=input_shape, name='input')
    print(input_layer.shape)
    model  = UNet_model()
    model.model_summry()
    output = model(input_layer)
    print(output.shape)


'''
Model: "u_net_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
down_block (down_block)      multiple                  38720     
_________________________________________________________________
down_block_1 (down_block)    multiple                  221440    
_________________________________________________________________
down_block_2 (down_block)    multiple                  885248    
_________________________________________________________________
down_block_3 (down_block)    multiple                  3539968   
_________________________________________________________________
bottleneck_conv1 (Conv2D)    multiple                  4719616   
_________________________________________________________________
bottleneck_conv2 (Conv2D)    multiple                  9438208   
_________________________________________________________________
up_block (up_block)          multiple                  143552    
_________________________________________________________________
up_block_1 (up_block)        multiple                  573824    
_________________________________________________________________
up_block_2 (up_block)        multiple                  2294528   
_________________________________________________________________
up_block_3 (up_block)        multiple                  9176576   
_________________________________________________________________
output (Conv2D)              multiple                  195       
=================================================================
Total params: 31,031,875
Trainable params: 31,031,875
Non-trainable params: 0
_________________________________________________________________
(None, 512, 512, 3)    
'''