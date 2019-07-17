from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.compat.v1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.compat.v1.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME')


Fmnist = input_data.read_data_sets("FMNIST/", one_hot=True)

X_train = Fmnist.train.images
Y_train = Fmnist.train.labels
X_test = Fmnist.test.images
Y_test = Fmnist.test.labels

x = tf.compat.v1.placeholder("float", shape=[None, 784])
x_image = tf.compat.v1.reshape(x, [-1, 28, 28, 1])
y_ = tf.compat.v1.placeholder("float", shape=[None, 10])
w1 = weight_variable([5, 5, 1, 32])
w2 = weight_variable([5, 5, 32, 64])
b1 = bias_variable([32])
b2 = bias_variable([64])

h_conv1 = tf.compat.v1.nn.relu(conv2d(x_image, w1) + b1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.compat.v1.nn.relu(conv2d(h_pool1, w2) + b2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.compat.v1.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.compat.v1.placeholder("float")  # probability for drop out
h_fc1_drop = tf.compat.v1.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.compat.v1.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.compat.v1.log(y_conv))
train_step = tf.compat.v1.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

with sess.as_default():
    for i in range(20000):
        batch = Fmnist.train.next_batch(50)
        # train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy), end=" ")
        if train_accuracy > 0.94:
            break
        print(train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}))
        # sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

print("test accuracy %g" % sess.run(accuracy,
                                    feed_dict={x: Fmnist.test.images, y_: Fmnist.test.labels, keep_prob: 1.0}))
