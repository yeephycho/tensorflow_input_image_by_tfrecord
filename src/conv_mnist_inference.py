from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc
import PIL
from PIL import Image

import tensorflow as tf

# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# pre-process data if necessary

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("image_path", "../num4_1.jpg", "Path to your input digit image.")


img = Image.open(FLAGS.image_path)
print("Orignial image size is: ")
print(img.size)
img = img.resize((28, 28), PIL.Image.ANTIALIAS)
img.save("../tmp_image.jpg")
print("Resized image size is: ")
print(img.size)

import tensorflow as tf
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Your own data
img = misc.imread("../tmp_image.jpg")
img.shape=(1, 784)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "../model_simple.ckpt")
print("model restored.")

result = sess.run(y_conv, feed_dict={x: img, keep_prob: 1.0})
print("The output of the network is: ")
print(result)
print("Prediction is (output one-hot digit from 0 to 9):")
print(sess.run(tf.argmax(result, 1)))
# print("test accuracy %g"%sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
