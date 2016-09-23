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
  
class image_object:
    def __init__(self):
        self.image = tf.Variable("0",dtype=tf.string)
        self.label = tf.Variable(0, dtype=tf.int32)
        self.filename = tf.Variable("0",dtype = tf.string)
  
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
    
    current_image = image_object()
    current_image.image = image
    current_image.label = label
    current_image.filename = filename

    return current_image.toString()

filename_queue = tf.train.string_input_producer(
        tfrecord_auto_traversal(),
#        ["train-00-of-04.tfrecord", "train-01-of-04.tfrecord","train-02-of-04.tfrecord","train-03-of-04.tfrecord"],
        shuffle = True)

#image, label, filename = read_and_decode(filename_queue)
cur_image = image_object()
cur_image = read_and_decode(filename_queue)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Write cropped and resized image to the folder './resized_image'") 
    for i in range(200): # number of examples in your tfrecord
        re_image = sess.run(cur_image)
        img = Image.fromarray(re_image[i], "RGB")
        if not os.path.isdir("./resized_image/"):
            os.mkdir("./resized_image")
        img.save(os.path.join("./resized_image/"+str(i)+".jpeg"))
        if i % 10 == 0:
            print ("%d images in has finished!" % i)
#        print sess.run(label)
        #print sess.run(features)
    print("Write finished!")
    coord.request_stop()
    coord.join(threads)
    sess.close()
print("Go to current directory, the folder resized_image should contains 200 images with 299x299 size.")

"""
images, sparse_labels = tf.train.shuffle_batch(
        [image, label], 
        batch_size = 5, 
        num_threads=2, 
        capacity = 10, 
        min_after_dequeue= 5)
"""
