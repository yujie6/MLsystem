from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def ComputeSingleCost(theta, X, b_, y):
    m = len(y)
    h = np.matmul(X, theta) + b_
    return -1 / m * (np.dot(y, np.log(h)) +
                     np.dot((1 - y), 1 - np.log(h)))


def ComputeGrad(theta, X, b_, y):
    h = np.matmul(X, theta) + b_
    return 1/len(y)*np.matmul(np.transpose(X), h-y)

def GradientDescentOptimize(theta, alpha, X, y, b_):
    return theta


def Optimize(w, alpha, X, y, b):
    m = np.size(X, 0)
    n = np.size(X, 1)
    for i in range(m):
        theta = w[:, i]
        y = y[:, i]
        theta = GradientDescentOptimize(theta, alpha, X, y, b)


if __name__ == "__main__":
    Fmnist = input_data.read_data_sets("FMNIST/", one_hot=True)
    # initialize
    X_train = Fmnist.train.images
    Y_train = Fmnist.train.labels
    X_test = Fmnist.test.images
    Y_test = Fmnist.test.labels
    w = np.ones([784, 10])
    b = np.ones([None, 10])
    learning_rate = 0.001
    for i in range(4000):
        batch_xs, batch_ys = Fmnist.train.next_batch(100)
        if i % 10 == 0:
            print("Epoch %d" % i, end=" Loss: ")
        Optimize(w, learning_rate, batch_xs, batch_ys, b)

    y_predict = np.matmul(X_test, w) + b
    accuracy = np.equal(np.argmax(y_predict, 1), np.argmax(Y_test, 1))
    print("Final Accuracy: %f" % np.sum(accuracy) / accuracy.shape[1])
