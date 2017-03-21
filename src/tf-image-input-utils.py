import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import os # path join


DATA_DIR = "../data/"
TRAINING_SET_SIZE = 3380
BATCH_SIZE = 20
IMAGE_SIZE = 224


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class _image_object: # image object from protobuf
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
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.filename = features["image/filename"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)
    return image_object

def flower_input(if_random):
    filenames = [os.path.join(DATA_DIR, "flower-train-0000%d-of-00002" % i) for i in xrange(0, 1)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError("Failed to find file: " + f)
    filename_queue = tf.train.string_input_producer(filenames)
    image_object = read_and_decode(filename_queue)
    image = image_object.image
    image = tf.image.per_image_standardization(image)
    label = image_object.label
    filename = image_object.filename

    if(if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print("Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = num_preprocess_threads,
            capacity = min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue = min_queue_examples)
        return image_batch, label_batch, filename_batch
    else:
        image_batch, label_batch, filename_batch = tf.train.batch(
            [image, label, filename],
            batch_size = BATCH_SIZE,
            num_threads = 1)
        return image_batch, label_batch, filename_batch

def debug_single_example():
    filenames = [os.path.join(DATA_DIR, "flower-train-0000%d-of-00002" % i) for i in xrange(0, 1)]
    filename_queue = tf.train.string_input_producer(filenames)

    image_object = read_and_decode(filename_queue)
    image = image_object.image
    label = image_object.label
    filename = image_object.filename

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    for i in range(2):
        image_out, label_out, filename_out = sess.run([image, label, filename])
        print(image_out.shape)
        print(label_out)
        print(filename_out)

    coord.request_stop()
    coord.join(threads)
    sess.close()


def debug_batch_example():
    filenames = [os.path.join(DATA_DIR, "flower-train-0000%d-of-00002" % i) for i in xrange(0, 1)]
    filename_queue = tf.train.string_input_producer(filenames)

    image_batch, label_batch, filename_batch = flower_input(if_random = True)
    image_batch = tf.reshape(image_batch, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
    label_offset = -tf.ones([BATCH_SIZE], dtype=tf.int64, name="label_batch_offset")
    label_batch = tf.add(label_batch, label_offset)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess = sess)

    for i in range(5):
        image_out, label_out, filename_out = sess.run([image_batch, label_batch, filename_batch])
        print(i)
        print(image_out.shape)
        print(label_out)
        print(filename_out)

    coord.request_stop()
    coord.join(threads)
    sess.close()


debug_single_example()

debug_batch_example()
