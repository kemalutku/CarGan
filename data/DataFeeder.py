import tensorflow as tf
import pandas as pd
import os
import re


def create_car_dataset(ds_dir, batch_size, image_size):
    def create_csv_dataset():
        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [image_size, image_size])
            image = (image - 127.5) / 127.5
            return image

        def configure_for_performance(ds: tf.data.Dataset):
            ds = ds.shuffle(buffer_size=100)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        df = pd.read_csv(ds_dir)
        filenames = list(df['path'])

        filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
        images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        images_ds = configure_for_performance(images_ds)
        return images_ds

    def create_tfrecord_dataset():
        def _parse_image_function(example):
            image_feature_description = {
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
            }
            features = tf.io.parse_single_example(example, image_feature_description)
            image = tf.image.decode_jpeg(features['image'])
            image = tf.image.resize(image, [image_size, image_size])
            image = (image - 127.5) / 127.5
            return image

        dataset = tf.data.TFRecordDataset(ds_dir)
        dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(500)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    _, ext = os.path.splitext(ds_dir)
    if ext == '.csv':
        image_dataset = create_csv_dataset()
    elif ext == '.record':
        image_dataset = create_tfrecord_dataset()
    else:
        image_dataset = None

    return image_dataset


if __name__ == '__main__':
    import os
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    df_path = r"D:\Dataset\MarkaModel\images.record"
    dataset = create_car_dataset(df_path, 16, 64)

    last_time = time.time()
    for i, image in enumerate(dataset):
        t = time.time()
        fetch_time = (t - last_time) * 1000
        if i % 100 == 0:
            print("Batch: {}".format(i), round(fetch_time, 2), 'ms')

        last_time = time.time()


