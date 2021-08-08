import tensorflow as tf


class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, paddings=(1, 1, 1, 1)):
        super(ReflectionPad2D, self).__init__()
        self.paddings = paddings

    def call(self, input, **kwargs):
        l, r, t, b = self.paddings

        return tf.pad(input, paddings=[[t, b], [l, r]], mode='REFLECT')


if __name__ == '__main__':
    import numpy as np
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    reflection_pad = ReflectionPad2D()

    x = np.arange(9).reshape((3, 3))
    y = reflection_pad(x)
    print(y.numpy())
