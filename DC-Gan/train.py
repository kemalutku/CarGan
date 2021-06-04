import tensorflow as tf
import numpy as np
import time
import os
import cv2

import configuration as config
from network import Generator, Discriminator
from data.DatasetFeeder import DataFeeder

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

# Load data generator
data_generator = DataFeeder(config.dataset_dir, config.image_size, config.batch_size).generator()

# Load models
discriminator = Discriminator(config.discriminator_feature_map_count)
generator = Generator(config.generator_feature_map_count, n_channels=3)

# Load loss and optimizers
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.beta1)

# Use a fixed noise to visualize generated data along epochs
fixed_noise = tf.cast(np.random.rand(64, config.latent_vector_size, 1, 1), tf.float32)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint_dir = r"DC-Gan/model/checkpoint"
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')


def discriminator_loss(real_output, fake_output):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return loss(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([config.batch_size, 1, 1, config.latent_vector_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))


for epoch in range(config.num_epochs):
    start = time.time()

    for i, image_batch in enumerate(data_generator):
        if i % 10 == 0:
            print("Batch:", i)
        train_step(image_batch)

    if (epoch + 1) % 1 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
