#!/bin/python3.7
import tensorflow as tf
import numpy as np
import autodiff as ad
from vector import *


def test_mul_two_vars():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


if __name__ == "__main__":
    test_mul_two_vars()
