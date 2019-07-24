from ._base import *
from . import autodiff
import numpy as np

# global list of all variable initializers
_all_variable_inits = []


def constant(init, shape):
    x = np.zeros(shape)
    x[:] = init
    return x


def placeholder(dtype=float32, shape=None):
    return autodiff.placeholder_op()


def Variable(init=None, dtype=float32):
    v = autodiff.Variable()
    if init is not None:
        if not isinstance(init, np.ndarray):
            if not isinstance(init, list):
                init = [init]
            init = np.array(init)
        c = autodiff.const_op(init)
        _all_variable_inits.append(autodiff.assign(v, c))
    return v


def sqrt(node):
    return autodiff.power_op(node, 0.5)


def power(node_a, node_b):
    return autodiff.power_op(node_a, node_b)


def log(node):
    return autodiff.log(node)


def matmul(node_a, node_b):
    return autodiff.matmul_op(node_a, node_b)


def reduce_sum(node, reduction_indices=None, keepdims=False):
    if not isinstance(reduction_indices, list):
        reduction_indices = [0]
    assert len(reduction_indices) == 1
    return autodiff.reduce_sum(node, reduction_indices[0], keepdims)


def reduce_mean(node, reduction_indices=None):
    return reduce_sum(node, reduction_indices) / autodiff.shape_op(node,
                                                                   reduction_indices)


def zeros(shape):
    return np.zeros(shape)


def equal(node_a, node_b):
    return autodiff.equal_op(node_a, node_b)


def argmax(node, axis=0):
    return autodiff.argmax_op(node, axis)


def cast(node, dtype=float32):
    return node


def assign(assign_to, value):
    return autodiff.assign(assign_to, value)


def initialize_all_variables():
    global _all_variable_inits
    init_node = autodiff.init_op(_all_variable_inits)
    _all_variable_inits = []
    return init_node


def global_variables_initializer():
    return initialize_all_variables()


def gradients(output_node, node_list):
    assert isinstance(output_node, autodiff.Node)
    assert isinstance(node_list, list)
    return autodiff.gradients(output_node, node_list)


def random_normal(shape, mean=0.0, stddev=1.0):
    return np.random.normal(loc=mean, scale=stddev, size=shape)


def reshape(node, shape):
    return autodiff.reshape_op(node, shape)
