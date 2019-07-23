from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
import tensorflow as tf
import numpy as np






def forward_prop(X, parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    Z1 = tf.compat.v1.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding="SAME")
    A1 = tf.compat.v1.nn.relu(Z1)
    P1 = tf.compat.v1.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)
    return Z3


def GetParameters():
    W1 = tf.get_variable('W1', [4, 4, 1, 4], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2, 2, 4, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2": W2}
    return parameters


def ComputeCost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def ComputeAcc(y, y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy


def main1():
    X_test = Fmnist.test.images
    Y_test = Fmnist.test.labels

    X = tf.compat.v1.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    y = tf.compat.v1.placeholder(shape=[None, 10], dtype=tf.float32)
    parameters = GetParameters()
    z3 = forward_prop(X, parameters=parameters)
    cost = ComputeCost(z3, y)
    optimizer = tf.compat.v1.train.AdamOptimizer(0.009).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(10000):
            batch_xs, batch_ys = Fmnist.train.next_batch(100)
            if i % 500 == 0:
                print("Epoch %d" % i, end=" Loss: ")
            sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys})

        print("Final Acc %g" % sess.run(ComputeAcc(z3, y), feed_dict={X: X_test, y: Y_test}))

def main:



if __name__ == "__main__":
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    main()
