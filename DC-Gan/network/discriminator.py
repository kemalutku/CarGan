import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.activations import relu, tanh, sigmoid
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import Model

conv_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


class Discriminator(Model):
    def __init__(self, feature_map_size):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2D(feature_map_size, 4, 2, padding='same', use_bias=False,
                            kernel_initializer=conv_initializer)
        self.conv2 = Conv2D(feature_map_size * 2, 4, 2, padding='same', use_bias=False,
                            kernel_initializer=conv_initializer)
        self.conv3 = Conv2D(feature_map_size * 4, 4, 2, padding='same', use_bias=False,
                            kernel_initializer=conv_initializer)
        self.conv4 = Conv2D(feature_map_size * 8, 4, 2, padding='same', use_bias=False,
                            kernel_initializer=conv_initializer)
        self.conv5 = Conv2D(1, 4, 1, use_bias=False)

        self.b_norm1 = BatchNormalization()
        self.b_norm2 = BatchNormalization()
        self.b_norm3 = BatchNormalization()
        self.b_norm4 = BatchNormalization()

        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.b_norm1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.b_norm2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.b_norm3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.b_norm4(x)
        x = self.leaky_relu(x)

        x = self.conv5(x)
        x = sigmoid(x)

        return x


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    discriminator_model = Discriminator(64)
    discriminator_model.build((1, 64, 64, 3))
    discriminator_model.summary()
