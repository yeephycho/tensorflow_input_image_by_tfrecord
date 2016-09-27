import tensorflow as tf
import numpy as np
import os

from PIL import Image
from dir_traversal_tfrecord import tfrecord_auto_traversal

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("image_number", 300, "Number of images in your tfrecord, default is 300.")
flags.DEFINE_integer("class_number", 3, "Number of class in your dataset/label.txt, default is 3.")
flags.DEFINE_integer("image_height", 299, "Height of the output image after crop and resize. Default is 299.")
flags.DEFINE_integer("image_width", 299, "Width of the output image after crop and resize. Default is 299.")

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
		self.image = tf.Variable([], dtype = tf.string)
		self.height = tf.Variable([], dtype = tf.int64)
		self.width = tf.Variable([], dtype = tf.int64)
		self.filename = tf.Variable([], dtype = tf.string)
		self.label = tf.Variable([], dtype = tf.int32)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features = {
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64),})

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)

    current_image_object = image_object()

    current_image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_height, FLAGS.image_width) # cropped image with size 299x299
#    current_image_object.image = tf.cast(image_crop, tf.float32) * (1./255) - 0.5
    current_image_object.height = features["image/height"] # height of the raw image
    current_image_object.width = features["image/width"] # width of the raw image
    current_image_object.filename = features["image/filename"] # filename of the raw image
    current_image_object.label = tf.cast(features["image/class/label"], tf.int32) # label of the raw image
    
    return current_image_object


def generate_mini_batch(image, label, batch_size = 50):
	images, labels = tf.train.shuffle_batch(
		[image, label],
		batch_size = batch_size,
		capacity = min_queue_examples + 3 * batch_size,
		min_after_dequeue = min_queue_examples
	)
	return images, labels


filename_queue = tf.train.string_input_producer(
        tfrecord_auto_traversal(),
        shuffle = True)


current_image_object = read_and_decode(filename_queue)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Write cropped and resized image to the folder './resized_image'") 
    for i in range(FLAGS.image_number): # number of examples in your tfrecord
        pre_image, pre_label = sess.run([current_image_object.image, current_image_object.label])
        img = Image.fromarray(pre_image, "RGB")
        if not os.path.isdir("./resized_image/"):
            os.mkdir("./resized_image")
        img.save(os.path.join("./resized_image/class_"+str(pre_label)+"_Index_"+str(i)+".jpeg"))
        if i % 10 == 0:
            print ("%d images in %d has finished!" % (i, FLAGS.image_number))
    print("Complete!!")
    coord.request_stop()
    coord.join(threads)
    sess.close()

print("cd to current directory, the folder 'resized_image' should contains %d images with %dx%d size." % (FLAGS.image_number,FLAGS.image_height, FLAGS.image_width))

"""
images, sparse_labels = tf.train.shuffle_batch(
        [image, label], 
        batch_size = 5, 
        num_threads=2, 
        capacity = 10, 
        min_after_dequeue= 5)
"""
