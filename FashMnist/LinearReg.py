from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

Fmnist = input_data.read_data_sets("FMNIST/", one_hot=True)

X_train = Fmnist.train.images
Y_train = Fmnist.train.labels
X_test = Fmnist.test.images
Y_test = Fmnist.test.labels

print(X_train.shape)
print(Y_train.shape)

x = tf.compat.v1.placeholder("float", [None, 784])
# W1 = tf.Variable(tf.zeros([784, 784]))
W2 = tf.Variable(tf.zeros([784, 10]))
# b1 = tf.Variable(tf.zeros([784]))
b2 = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y1 = tf.compat.v1.nn.softmax(tf.matmul(x, W1) + b1)
y = tf.compat.v1.nn.softmax(tf.matmul(x, W2) + b2)
y_ = tf.compat.v1.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.math.log(y))
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

for i in range(4000):
    batch_xs, batch_ys = Fmnist.train.next_batch(100)
    print("Epoch: %d" % i, end=", Loss:")
    print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: Fmnist.test.images, y_: Fmnist.test.labels}))
