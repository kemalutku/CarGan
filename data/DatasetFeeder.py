import tensorflow as tf
import numpy as np
import cv2


class DataFeeder:
    def __init__(self, dataset_dir, image_size, batch_size, model='dcgan'):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size

        self.image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        self.raw_dataset = tf.data.TFRecordDataset(dataset_dir).map(self._parse_image_function)

    def _parse_image_function(self, proto):
        return tf.io.parse_single_example(proto, self.image_feature_description)

    def decode_image(self, image):
        image = tf.image.decode_image(image, channels=3)
        image = tf.cast(image, tf.float32)
        return image

    def preprocess(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image - 127.5) / 127.5
        return image

    def generator(self):

        for image_features in self.raw_dataset.batch(self.batch_size):
            image_list = []
            images = np.array()
            # images = np.array([self.decode_image(raw_image).numpy() for raw_image in image_features['image']])
            for image in images:
                image_list.append(self.preprocess(image))
            yield np.array(image_list)


if __name__ == '__main__':
    import cv2
    import numpy as np
    import os
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    record_dir = r"D:\Dataset\MarkaModel\images.record"
    feeder = DataFeeder(record_dir, 64, 64)
    generator = feeder.generator()
    while True:
        s = time.time()
        image_batch = generator.__next__()
        e = time.time()
        print("Acquire time: {}ms".format(np.round((e - s) * 1000, 2)))
        # for i, image in enumerate(image_batch):
        #     image = np.uint8(image * 127.5 + 127.5)
        #     cv2.imshow(str(i), image)
        # cv2.waitKey(0)
