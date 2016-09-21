import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
  
  
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image = tf.image.decode_jpeg(features["image/encoded"], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 299, 299)
#    image = tf.cast(image, tf.float32) * (1./255) - 0.5
    label = tf.cast(features["image/class/label"], tf.int32)
    height = features["image/height"]
    filename = features["image/filename"]

    return image, label, filename

filename_queue = tf.train.string_input_producer(
        ["./train-00000-of-00004", "./train-00001-of-00004","./train-00002-of-00004","./train-00003-of-00004"],
        shuffle = True)

image, label, filename = read_and_decode(filename_queue)

tfImage = tf.Variable(tf.zeros([128*128*3]), dtype = tf.float32)
tfImage = image
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print "from the train set:"
    for i in range(2000): # number of examples in your tfrecord
        print sess.run(filename)
        re_image = sess.run(tfImage)
        img = Image.fromarray(re_image, "RGB")
        img.save(os.path.join("./re_steak/"+str(i)+".jpeg"))
        print i
#        print sess.run(label)
        #print sess.run(features)
    coord.request_stop()
    coord.join(threads)
    sess.close()
print("Line~~~~~~~~~~~~~~~~~~~~~~~~~~")

"""
images, sparse_labels = tf.train.shuffle_batch(
        [image, label], 
        batch_size = 5, 
        num_threads=2, 
        capacity = 10, 
        min_after_dequeue= 5)
"""

#with tf.Session() as sess:
#    sess.run(tf.initialize_all_variables())
#    for i in range(1):
#        print sess.run(images)
#    sess.close()
