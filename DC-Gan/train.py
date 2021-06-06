import tensorflow as tf
import numpy as np
import time
import os
import cv2
import datetime

import configuration as config
from network import Generator, Discriminator
from data.DataFeeder import create_car_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

# Load data generator
data_generator = create_car_dataset(config.dataset_dir, config.batch_size, config.image_size)

# Load models
discriminator = Discriminator(config.discriminator_feature_map_count)
generator = Generator(config.generator_feature_map_count, n_channels=3)

# Load loss and optimizers
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.beta1)
discriminator_optimizer = tf.keras.optimizers.Adam(config.learning_rate, config.beta1)

# Use a fixed noise to visualize generated data along epochs
fixed_noise = tf.cast(np.random.rand(64, 1, 1, config.latent_vector_size), tf.float32)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint_dir = r"checkpoint/car-gan"
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt-epoch{}-step{}')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Tensorboard setup
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = r"logs/" + current_time
log_dir = os.path.join(os.getcwd(), log_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
train_summary_writer = tf.summary.create_file_writer(log_dir)

generator_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)


def discriminator_loss(real_output, fake_output):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return loss(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([12, 1, 1, config.latent_vector_size])

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

    generator_metric(gen_loss)
    discriminator_metric(disc_loss)


for epoch in range(config.num_epochs):
    start = time.time()
    step_counter = 0

    for i, image_batch in enumerate(data_generator):
        step_counter += 1
        train_step(image_batch)

        if i % 100 == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar('generator_loss', generator_metric.result(), step=(epoch + 1) * step_counter)
                tf.summary.scalar('discriminator_loss', discriminator_metric.result(), step=(epoch + 1) * step_counter)

            template = "Epoch: {}, Batch: {}, Generator Loss: {}, Discriminator Loss: {}"
            print(template.format(
                epoch + 1,
                i,
                generator_metric.result(),
                discriminator_metric.result()
            ))
        if i % 5000 == 0:
            generated_images = generator(fixed_noise, training=False)
            generated_images = tf.cast(generated_images * 127.5 + 127.5, tf.uint8)
            with train_summary_writer.as_default():
                tf.summary.image('Generated Images', generated_images, max_outputs=12, step=step_counter)

            checkpoint.save(file_prefix=checkpoint_prefix.format(epoch, step_counter))

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
