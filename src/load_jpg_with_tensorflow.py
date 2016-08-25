# Typical setup to include TensorFlow
import tensorflow as tf
import matplotlib.pyplot as plt

# Make a queue of file names including all the JPEG images files in the
# relative image directory.
filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("./human_dataset/s0/*.jpg"))

# Read an entie image file which is required since they're JPEGs, if the 
# images are too large they could be split in advance to smaller files or
# use the fixed reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple
# is the filename which we are ignoring.
key, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which
# we can then use in training.
image = tf.image.decode_jpeg(image_file)

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

# Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    image_tensor = sess.run([image])
    print(image_tensor[0].shape)
    print image.get_shape()
    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# images and labels array as input
def convert_to(images, labels, name):
    num_examples = labels.shape[0]
    if images.shape[0] != num_examples:
        raise ValueError("Image size %d does not match label size %d."%(images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features = tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())


with tf.Session as sess:
    convert_to(image, labels, 's0')

