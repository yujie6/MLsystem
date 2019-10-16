from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

Fmnist = input_data.read_data_sets("FMNIST/", one_hot=True)

x = tf.compat.v1.placeholder("float", [None, 784])
W2 = tf.Variable(tf.zeros([784, 10]))
b2 = tf.Variable(tf.zeros([10]))
y = tf.compat.v1.nn.softmax(tf.matmul(x, W2) + b2)
y_ = tf.compat.v1.placeholder("float", [None, 10])
"""
3 choices for cost function, but L2 norm works not so well
"""
cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y))
# best learning rate = 0.001 acc: 85%
softmax_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
# best learning rate = 0.25 acc: 80%
square_cost = tf.reduce_mean(tf.squared_difference(y_, y))
# not working well acc: 50%

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

for i in range(5000):
    batch_xs, batch_ys = Fmnist.train.next_batch(100)
    if i % 500 == 0:
        print("Epoch: %d" % i, end=", Loss:")
        print(sess.run(square_cost, feed_dict={x: batch_xs, y_: batch_ys}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: Fmnist.test.images, y_: Fmnist.test.labels}))
