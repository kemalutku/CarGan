import pandas as pd
import tensorflow as tf
import numpy as np
import re
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Recorder:
    def __init__(self, df_dir, classes_dir):
        self.df_dir = df_dir

        self.df = pd.read_csv(df_dir, index_col=0)
        self.classes = self.pbtxt2dict(classes_dir)

    @staticmethod
    def pbtxt2dict(pbtxt_dir):
        pb_file = open(pbtxt_dir, "r")
        pb_txt = pb_file.read()
        keys = re.findall(r"{\n\tid: (.*)\n\tname:", pb_txt)
        items = re.findall(r"\n\tname: \'(.*)\'\n}\n", pb_txt)
        classes_dict = {i: int(k) for k, i in zip(keys, items)}
        return classes_dict

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_mm(self, image_feature, label_feature):
        feature = {
            'image': self._bytes_feature(image_feature),
            'label': self._int64_feature(label_feature)
        }
        mm_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return mm_proto.SerializeToString()

    def record(self, output_dir=None):
        if output_dir is None:
            output_dir = os.path.dirname(self.df_dir)
        if not os.path.isfile(output_dir):
            output_dir = os.path.join(output_dir, 'images.record')
        if not os.path.isdir(os.path.dirname(output_dir)):
            os.makedirs(os.path.dirname(output_dir))

        with tf.io.TFRecordWriter(output_dir) as writer:
            counter = 0
            for _, row in self.df.iterrows():
                if counter % 500 == 0:
                    print(counter, '/', len(self.df))
                counter += 1
                image_dir = row['path']
                if not os.path.isfile(image_dir):
                    print('File not found:', image_dir)
                    continue
                image_string = open(image_dir, 'rb').read()
                try:
                    label = self.classes[row['label']]
                except KeyError:
                    print("Skipped label:", row['label'], "label not defined")
                    continue
                serialized_feature = self.serialize_mm(image_string, label)
                writer.write(serialized_feature)


if __name__ == '__main__':
    df_path = r"D:\Dataset\MarkaModel\MarkaModelMerged.csv"
    classes_path = r"D:\Dataset\MarkaModel\make_model.pbtxt"

    recorder = Recorder(df_path, classes_path)
    recorder.record(r"C:\Users\Utku\Documents\HelloWorld\CarGan")
