from . import autodiff


def softmax(node):
    exp_node = autodiff.exp(node)
    return exp_node / autodiff.reduce_sum(exp_node,
                                          reduction_indices=1, keep_dims=True)


def relu(node):
    return autodiff.relu(node)


def conv2d(input, filter, strides, padding):
    assert isinstance(strides, list) and len(strides) == 4
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return autodiff.conv2d_op(input, filter, strides, padding)


def max_pool(value, ksize, strides, padding):
    assert isinstance(ksize, list) and len(ksize) == 4
    assert isinstance(strides, list) and len(strides) == 4
    assert ksize[0] == 1 and ksize[3] == 1
    assert strides[0] == 1 and strides[3] == 1
    assert padding == "SAME" or padding == "VALID"
    return autodiff.maxpool_op(value, ksize, strides, padding)


def dropout(input, keep_prob):
    return autodiff.dropout_op(input, keep_prob)
