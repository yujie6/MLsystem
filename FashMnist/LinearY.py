from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def ComputeCost(w, X, b_, y):
    m = len(y)
    h = sigmoid(np.matmul(X, w) + b_ )
    return -1 / m * (np.matmul(np.transpose(y), np.log(h + 1e-7)) +
                     np.matmul(np.transpose(1 - y), np.log(1 - h + 1e-7)))


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def ComputeGrad(w, X, b_, y):
    h = np.matmul(X, w) + b_
    return 1 / len(y) * np.matmul(np.transpose(X), h - y)


def GradientDescentOptimize(w, alpha, X, y, b_):
    return w - alpha * ComputeGrad(w, X, b_, y)


def Optimize(w, alpha, X, y, b):
    m = np.size(X, 0)
    n = np.size(X, 1)
    w = GradientDescentOptimize(w, alpha, X, y, b)
    return w


if __name__ == "__main__":
    Fmnist = input_data.read_data_sets("FMNIST/", one_hot=True)
    # initialize
    X_train = Fmnist.train.images
    Y_train = Fmnist.train.labels
    X_test = Fmnist.test.images
    Y_test = Fmnist.test.labels
    w = np.ones([784, 10])
    b = np.ones([10])
    learning_rate = 0.001
    for i in range(10000):
        batch_xs, batch_ys = Fmnist.train.next_batch(100)
        if i % 200 == 0:
            print("Epoch %d" % i, end=" Loss: ")
            print(np.sum(ComputeCost(w, batch_xs, b, batch_ys)))
        w = Optimize(w, learning_rate, batch_xs, batch_ys, b)

    y_predict = np.matmul(X_test, w) + b
    accuracy = np.equal(np.argmax(y_predict, 1), np.argmax(Y_test, 1))
    print("Final Accuracy: %g" % (np.sum(accuracy) / accuracy.shape[0]))
    print(w)
    print(b)
