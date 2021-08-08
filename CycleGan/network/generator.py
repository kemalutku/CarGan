import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.activations import relu, tanh
from tensorflow import pad


class ResNetGenerator(tf.keras.models.Model):
    def __init__(self, n_features, n_blocks, **kwargs):
        super(ResNetGenerator, self).__init__()
        self.n_downsampling = kwargs.pop('n_downsampling', 2)
        self.n_blocks = n_blocks
        self.conv1 = Conv2D(n_features, 7, padding='valid')
        self.norm1 = InstanceNormalization()

        # Define the downsampling convs
        self.downsampling_convs = []
        self.downsampling_norms = []
        for i in range(self.n_downsampling):
            multiplier = 2 ** i
            self.downsampling_convs.append(
                Conv2D(n_features * multiplier * 2, 3, 2, padding='valid')
            )
            self.downsampling_norms.append(InstanceNormalization())

        # Define the resnet blocks
        self.res_blocks = []
        multiplier = 2 ** self.n_downsampling
        for i in range(n_blocks):
            self.res_blocks.append(ResBlock(n_features * multiplier))

        # Define the upsampling convs
        self.upsampling_convs = []
        self.upsampling_norms = []
        for i in range(self.n_downsampling):
            multiplier = 2 ** (self.n_downsampling - i)
            self.upsampling_convs.append(Conv2DTranspose(int(n_features * multiplier / 2), 3, 2, 'same', 1))
            self.upsampling_norms.append(InstanceNormalization())

        self.conv2 = Conv2D(3, 7, padding='valid')

    def call(self, inputs, **kwargs):
        x = pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.conv1(x)
        x = self.norm1(x)
        x = relu(x)

        for i in range(self.n_downsampling):
            x = self.downsampling_convs[i](x)
            x = self.downsampling_norms[i](x)
            x = relu(x)

        for i in range(self.n_blocks):
            x = self.res_blocks[i](x)

        for i in range(self.n_downsampling):
            x = self.upsampling_convs[i](x)
            x = self.upsampling_norms[i](x)
            x = relu(x)

        x = pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
        x = self.conv2(x)
        x = tanh(x)

        return x


class ResBlock(tf.keras.Model):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = Conv2D(dim, 3, padding='valid')
        self.conv2 = Conv2D(dim, 3, padding='valid')

        self.norm1 = InstanceNormalization()
        self.norm2 = InstanceNormalization()
        self.norm3 = InstanceNormalization()

    def call(self, inputs, **kwargs):
        x = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT')
        x = self.conv1(x)
        x = self.norm1(x)
        x = relu(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.norm3(x)
        return x + inputs


if __name__ == '__main__':
    import numpy as np
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    generator = ResNetGenerator(64, 6)
    fixed_noise = tf.cast(np.random.rand(1, 256, 256, 3), tf.float32)
    generator.build(fixed_noise.shape)
    generator.summary()
