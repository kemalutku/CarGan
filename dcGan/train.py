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
np.random.seed(42)
fixed_noise = tf.cast(np.random.rand(12, 1, 1, config.latent_vector_size), tf.float32)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Tensorboard setup
run_name = config.run_name.format(
    config.batch_size,
    config.latent_vector_size,
    config.generator_feature_map_count,
    config.discriminator_feature_map_count,
    config.learning_rate,
    config.beta1,
    config.gen_disc_train_ratio,
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    config.info
)

checkpoint_dir = r"checkpoint/" + run_name
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt-epoch{}-step{}')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

log_dir = r"logs/" + run_name
log_dir = os.path.join(os.getcwd(), log_dir)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
train_summary_writer = tf.summary.create_file_writer(log_dir)

generator_loss_metric = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
discriminator_loss_metric = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)


def discriminator_loss(real_output, fake_output):
    real_loss = loss(tf.ones_like(real_output), real_output)
    fake_loss = loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return loss(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, train_disc=False):
    noise = tf.random.normal([12, 1, 1, config.latent_vector_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))
    generator_loss_metric(gen_loss)

    if train_disc:
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        discriminator_loss_metric(disc_loss)


step_counter = 0
for epoch in range(config.num_epochs):
    start = time.time()
    train_ratio_counter = config.gen_disc_train_ratio
    for i, image_batch in enumerate(data_generator):
        step_counter += 1
        if train_ratio_counter == config.gen_disc_train_ratio:
            train_step(image_batch, train_disc=True)
            train_ratio_counter = 1
        else:
            train_step(image_batch)
            train_ratio_counter += 1

        if step_counter % 100 == 0:
            generated_images = generator(fixed_noise, training=False)
            generated_images = tf.cast(generated_images * 127.5 + 127.5, tf.uint8)

            with train_summary_writer.as_default():
                tf.summary.scalar('generator_loss', generator_loss_metric.result(), step=step_counter)
                tf.summary.scalar('discriminator_loss', discriminator_loss_metric.result(), step=step_counter)
                # tf.summary.image('Generated Images', generated_images, max_outputs=12, step=step_counter)

            template = "Epoch: {}, Step: {}, Generator Loss: {}, Discriminator Loss: {}"
            print(template.format(
                epoch + 1,
                step_counter,
                generator_loss_metric.result(),
                discriminator_loss_metric.result()
            ))

        if step_counter % 2000 == 0:
            generated_images = generator(fixed_noise, training=False)
            generated_images = tf.cast(generated_images * 127.5 + 127.5, tf.uint8)
            with train_summary_writer.as_default():
                tf.summary.image('Generated Images', generated_images, max_outputs=12, step=step_counter)

            checkpoint.save(file_prefix=checkpoint_prefix.format(epoch, step_counter))

    print('Time for epoch {} is {} sec'.format(epoch + 1, np.round(time.time() - start)))
