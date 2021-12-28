# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/17; 13:18
# -*- python3.5
"""
特别注意: 17行VOC_LABELS标签要修改，189行的path地址要正确
"""

import os
import sys
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET

# 我的标签定义只有手表这一类，所以下面的VOC_LABELS要根据自己的图片标签而定，第一组'none': (0, 'Background')是不能删除的；
VOC_LABELS = {
    'none': (0, 'Background'),
    'arrow_left': (1, 'arrow_left'),
    'singleP_left': (2, 'singleP_left'),
    'doubleP_left': (3, 'singleP_left'),
    'ground_right': (4, 'ground_right'),
    'turn_right': (5, 'turn_right'),
    'arrow_right': (6, 'arrow_right'),

}

# 图片和标签存放的文件夹.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# 随机种子.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 4  # 每个.tfrecords文件包含几个.xml样本



def int64_feature(value):
    """
    生成整数型，浮点型和字符串型的属性
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _process_image(directory, name):
    """
    图片处理
    """
    # Read the image file.
    filename = directory + DIRECTORY_IMAGES + name + '.png'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Read the XML annotation file.
    filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
    tree = ET.parse(filename)
    root = tree.getroot()

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))  # 变为ascii格式

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        a = float(bbox.find('ymin').text) / shape[0]
        b = float(bbox.find('xmin').text) / shape[1]
        a1 = float(bbox.find('ymax').text) / shape[0]
        b1 = float(bbox.find('xmax').text) / shape[1]
        a_e = a1 - a
        b_e = b1 - b
        if abs(a_e) < 1 and abs(b_e) < 1:
            bboxes.append((a, b, a1, b1))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def _convert_to_example(image_data, labels, labels_text, bboxes, shape,difficult, truncated):
    """
    转化样例
    """
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    image_format = b'png'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_feature(labels),
        'image/object/bbox/label_text': bytes_feature(labels_text),
        'image/object/bbox/difficult': int64_feature(difficult),
        'image/object/bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """
    增加到tfrecord
    """
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name, idx):
    """
    name为转化文件的前缀
    """
    return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)


def run(dataset_dir, output_dir, name='voc_2007_train', shuffling=False):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames) and j < SAMPLES_PER_FILES:
                sys.stdout.write(' Converting image %d/%d \n' % (i + 1, len(filenames)))  # 终端打印，类似print
                sys.stdout.flush()  # 缓冲

                filename = filenames[i]
                img_name = filename[:-4]
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

    print('\nFinished converting the Pascal VOC dataset!')


def main(_):
    # 原数据集路径，输出路径以及输出文件名，要根据自己实际做改动
    dataset_dir = "/root/SSD-Tensorflow-On-Custom-Dataset/VOC2007/"
    output_dir = "/root/SSD-Tensorflow-On-Custom-Dataset/tfrecords"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run(dataset_dir, output_dir)


if __name__ == '__main__':
    tf.app.run()
