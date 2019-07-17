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
