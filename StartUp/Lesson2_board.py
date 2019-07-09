from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
"""
Must use absolute path! 
God damn, waste plenty of time on this...
"""
a = tf.constant([1.0, 2.0, 3.0], name='input1')
b = tf.Variable(tf.random.uniform([3]), name='input2')
add = tf.add_n([a, b], name='addOP')
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter('/home/yujie6/Music/tensorboard/log', sess.graph)
    print(sess.run(add))
writer.close()

