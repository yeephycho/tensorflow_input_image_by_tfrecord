import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

from dir_traversal_tfrecord import tfrecord_auto_traversal


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
    
    return image, label

filename_queue = tf.train.string_input_producer(
        tfrecord_auto_traversal(),
        shuffle = True)

cur_image, cur_label = read_and_decode(filename_queue)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Write cropped and resized image to the folder './resized_image'") 
    j = k = l = 0
    for i in range(300): # number of examples in your tfrecord
        pre_image, pre_label = sess.run([cur_image, cur_label])
        img = Image.fromarray(pre_image, "RGB")
        if not os.path.isdir("./resized_image/"):
            os.mkdir("./resized_image")
        if pre_label == 1:
            img.save(os.path.join("./resized_image/steak"+str(j)+".jpeg"))
            j += 1
        elif pre_label == 2:
            img.save(os.path.join("./resized_image/sushi"+str(k)+".jpeg"))
            k += 1
        else:
            img.save(os.path.join("./resized_image/waffles"+str(l)+".jpeg"))
            l += 1
        if i % 10 == 0:
            print ("%d images in has finished!" % i)
    print("Write finished!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
print("Go to current directory, the folder resized_image should contains 300 images with 299x299 size.")

"""
images, sparse_labels = tf.train.shuffle_batch(
        [image, label], 
        batch_size = 5, 
        num_threads=2, 
        capacity = 10, 
        min_after_dequeue= 5)
"""
