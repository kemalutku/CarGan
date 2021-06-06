import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.activations import relu, tanh, sigmoid
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import Model

conv_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Generator(Model):
    def __init__(self, feature_map_size, n_channels):
        super(Generator, self).__init__()

        self.t_conv1 = Conv2DTranspose(feature_map_size * 8, (4, 4), 1, padding='valid', use_bias=False,
                                       kernel_initializer=conv_initializer)
        self.t_conv2 = Conv2DTranspose(feature_map_size * 4, (4, 4), 2, padding='same', use_bias=False,
                                       kernel_initializer=conv_initializer)
        self.t_conv3 = Conv2DTranspose(feature_map_size * 2, (4, 4), 2, padding='same', use_bias=False,
                                       kernel_initializer=conv_initializer)
        self.t_conv4 = Conv2DTranspose(feature_map_size, (4, 4), 2, padding='same', use_bias=False,
                                       kernel_initializer=conv_initializer)
        self.t_conv5 = Conv2DTranspose(n_channels, (4, 4), 2, padding='same', use_bias=False,
                                       kernel_initializer=conv_initializer)

        self.b_norm1 = BatchNormalization()
        self.b_norm2 = BatchNormalization()
        self.b_norm3 = BatchNormalization()
        self.b_norm4 = BatchNormalization()

    def call(self, inputs, **kwargs):
        x = self.t_conv1(inputs)
        x = self.b_norm1(x)
        x = relu(x)

        x = self.t_conv2(x)
        x = self.b_norm2(x)
        x = relu(x)

        x = self.t_conv3(x)
        x = self.b_norm3(x)
        x = relu(x)

        x = self.t_conv4(x)
        x = self.b_norm4(x)
        x = relu(x)

        x = self.t_conv5(x)
        x = tanh(x)

        return x


if __name__ == '__main__':
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    import cv2

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    generator_model = Generator(64, 3)
    generator_model.build((1, 1, 1, 100))
    generator_model.summary()

    fixed_noise = tf.cast(np.random.rand(1, 1, 1, 100), tf.float32)
    results = generator_model(fixed_noise, training=False)

    results = results * 127.5 + 127.5
    results = results.numpy().astype(np.uint8)
    for i, result in enumerate(results):
        cv2.imshow(str(i), result[:,:,1])
        cv2.waitKey(0)

    # plt.imshow(result[0, :, :])
    # plt.show()
