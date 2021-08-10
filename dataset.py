"""This module provides the functions to build a TensorFlow dataset."""

import tensorflow as tf
import imageio
from skimage.transform import resize
import numpy as np
from config import *


def _parse(example):
    """Extract data from a line.
    Args:
        example: image_path, labels

    Returns:
        a parsed data and label pair.
    """

    # label = tf.strings.split(file_path, os.sep)[-2]

    example = example.numpy().decode("utf-8")
    record = example.split(",")
    image_path = record[0]
    points = record[1:]
    img = imageio.imread(image_path)
    orig_size = img.shape
    img = resize(img, (input_shape[0], input_shape[1]))
    ratio_x = input_shape[1] / orig_size[1]
    ratio_y = input_shape[0] / orig_size[0]
    points = [float(t) * ratio_x if idx % 2 == 0 else float(t) * ratio_y for idx, t in enumerate(points)]

    return tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(np.array(points), dtype=tf.float32)


def make_dataset(dataset_file, batch_size, shuffle=True):
    """Return a dataset for model.
    Args:
        dataset_file: the dataset_file file.
        batch_size: batch size.
        shuffle: whether to shuffle the data.

    Returns:
        a dataset.
    """
    # Init the dataset from the dataset_file.
    dataset = tf.data.TextLineDataset(dataset_file)

    autotune = tf.data.experimental.AUTOTUNE
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(lambda x: tf.py_function(_parse, [x], (tf.float32, tf.float32)), num_parallel_calls=autotune)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=autotune)

    return dataset
