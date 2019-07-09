from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.400

bias = tf.Variable(tf.zeros([1]), name='bias')
Weight = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='Weight')
y_real = tf.matmul(Weight, x_data) + bias

loss = tf.reduce_mean(tf.square(y_real - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning rate
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(1, 800):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weight), sess.run(bias))

file_writer = tf.compat.v1.summary.FileWriter('/home/yujie6/Music/tensorboard/log', sess.graph)

