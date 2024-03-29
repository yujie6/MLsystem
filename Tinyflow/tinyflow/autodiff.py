import numpy as np
from operator import add
from functools import reduce
from ctypes import *
from ._base import lib
from ._base import cast_to_ndarray

use_cpp = True


class Node(object):
    """
    node in the computation graph
    """
    op = None
    name = ""
    inputs = []
    const_attr = None
    matmul_attr_tans_A = False
    matmul_attr_tans_B = False

    def __init__(self):
        self.name = ""

    def __add__(self, other):
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            new_node = addconst_op(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            return sub_op(self, other)
        else:
            return subconst_op(self, other)

    def __rsub__(self, other):
        if isinstance(other, Node):
            # this line will never be implemented
            print("wtf??")
        else:
            return rsubconst_op(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            new_node = mulconst_op(self, other)
        return new_node

    def __neg__(self):
        new_node = neg_op(self)
        return new_node

    def __truediv__(self, other):
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            new_node = divconst_op(self, other)
        return new_node

    def __rtruediv__(self, other):
        """other / self when other is not a node??"""
        if isinstance(other, Node):
            new_node = rdiv_op(self, other)
        else:
            new_node = rdivconst_op(self, other)
        return new_node

    def eval(self, feed_dict=None):
        from ._session import Session
        with Session() as sess:
            return sess.run(self, feed_dict=feed_dict)

    def run(self, feed_dict=None):
        return self.eval(feed_dict=feed_dict)

    __radd__ = __add__
    __rmul__ = __mul__


class Op(object):
    def __call__(self):
        new_node = Node()
        new_node.op = self
        # new_node.name = ""
        return new_node

    def gradients(self, tnode, output_grad):
        assert False

    def compute(self, tnode, input_vals):
        assert False


class Mul_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "%s * %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(tnode.inputs[1] * output_grad, tnode.inputs[0]),
                reduce_sum_to(tnode.inputs[0] * output_grad, tnode.inputs[1])]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]


class MulConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.const_attr = const_val
        # new_node.name = "%s * %s" % (node_a.name, str(const_val))
        return new_node

    def gradients(self, tnode, output_grad):
        return [output_grad * tnode.const_attr]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] * tnode.const_attr


class Add_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "%s + %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(output_grad, tnode.inputs[0]),
                reduce_sum_to(output_grad, tnode.inputs[1])]

    def compute(self, tnode, input_vals):
        return input_vals[0] + input_vals[1]


class Inv_Op(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        output_val[:] = 1 / input_vals[0]

    def gradient(self, node, output_grad):
        return [-1 * inv(node.inputs[0] * node.inputs[0]) * output_grad]


class AddConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.const_attr = const_val
        # new_node.name = "%s + %s" % (node_a.name, str(const_val))
        return new_node

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] + tnode.const_attr

    def gradients(self, tnode, output_grad):
        return [output_grad]


class Neg_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        # new_node.name = "-%s" % node_a.name
        return new_node

    def compute(self, tnode, input_vals):
        return -input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-output_grad]


class Sub_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "%s - %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(output_grad, tnode.inputs[0]),
                reduce_sum_to(-output_grad, tnode.inputs[1])]

    def compute(self, tnode, input_vals):
        return input_vals[0] - input_vals[1]


class SubConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.const_attr = const_val
        # new_node.name = "%s - %s" % (node_a.name, str(const_val))
        return new_node

    def compute(self, tnode, input_vals):
        return input_vals[0] - tnode.const_attr

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(output_grad, tnode.inputs[0])]


class RSubConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_a]
        # new_node.name = "%s - %s" % (str(const_val), node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return tnode.const_attr - input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-reduce_sum_to(output_grad, tnode.inputs[0])]


class Div_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "%s / %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(output_grad / tnode.inputs[1], tnode.inputs[0]),
                reduce_sum_to(-output_grad * tnode.inputs[0] / (tnode.inputs[1] * tnode.inputs[1])
                              , tnode.inputs[1])]

    def compute(self, tnode, input_vals):
        return input_vals[0] / input_vals[1]


class DivConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_a]
        # new_node.name = "%s / %s" % (node_a.name, str(const_val))
        return new_node

    def compute(self, tnode, input_vals):
        return input_vals[0] / tnode.const_attr

    def gradients(self, tnode, output_grad):
        return [output_grad / tnode.const_attr]


class RDiv_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "%s / %s" % (node_b.name, node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return input_vals[1] / input_vals[0]

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(-output_grad * tnode.inputs[1] / (tnode.inputs[0] * tnode.inputs[0]),
                              tnode.inputs[0]),
                reduce_sum_to(output_grad / tnode.inputs[0],
                              tnode.inputs[1])]


class RDivConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_a]
        # new_node.name = "%s / %s" % (str(const_val), node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return tnode.const_attr / input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-output_grad * tnode.const_attr / (tnode.inputs[0] * tnode.inputs[0])]


class MatMul_Op(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        # new_node.name = "matmul(%s, %s, %s, %s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def gradients(self, tnode, output_grad):
        """matrix gradient is a bit different, but almost the same as
        the one of multi-variable function
        Given gradient of multiply node, return gradient contributions to each input.
               Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [matmul_op(output_grad, tnode.inputs[1], False, True),
                matmul_op(tnode.inputs[0], output_grad, True, False)]

    def compute(self, tnode, input_vals):
        if use_cpp:
            output_shape = [
                input_vals[0].shape[1] if tnode.matmul_attr_trans_A else input_vals[0].shape[0],
                input_vals[1].shape[0] if tnode.matmul_attr_trans_B else input_vals[1].shape[1]
            ]
            A = input_vals[0].astype(np.float32)
            B = input_vals[1].astype(np.float32)
            C = np.zeros(output_shape).astype(np.float32)
            A_data = A.ctypes.data_as(POINTER(c_float))
            B_data = B.ctypes.data_as(POINTER(c_float))
            C_data = C.ctypes.data_as(POINTER(c_float))
            m = C.shape[0]
            n = C.shape[1]
            k = A.shape[0] if tnode.matmul_attr_trans_A else A.shape[1]
            lib.matmul(A_data, B_data, C_data,
                       tnode.matmul_attr_trans_A,
                       tnode.matmul_attr_trans_B,
                       m, k, n)
            return C
        else:
            mat_A = input_vals[0]
            mat_B = input_vals[1]
            if tnode.matmul_attr_trans_A:
                mat_A = mat_A.T
            if tnode.matmul_attr_trans_B:
                mat_B = mat_B.T
            return np.matmul(mat_A, mat_B)


class Exp_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        # new_node.name = "exp(%s)" % (node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def gradients(self, tnode, output_grad):
        return [output_grad * tnode]


class Log_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        # new_node.name = "log(%s)" % node_a.name
        return new_node

    def compute(self, tnode, input_vals):
        return np.log(input_vals[0])

    def gradients(self, tnode, output_grad):
        return [output_grad / tnode.inputs[0]]


class Power_Op(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        if not isinstance(node_A, Node):
            node_A = const_op(node_A)
        if not isinstance(node_B, Node):
            node_B = const_op(node_B)
        new_node.inputs = [node_A, node_B]
        # new_node.name = "%s ^ %s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, tnode, input_vals):
        return np.power(input_vals[0], input_vals[1])

    def gradients(self, tnode, output_grad):
        return [
            output_grad * tnode.inputs[1] * power_op(tnode.inputs[0], tnode.inputs[1] - 1),
            output_grad * log(tnode.inputs[0]) * tnode
        ]
        # d(a^x)/dx = ln(a) * a^x


class PlaceholderOp(Op):
    """Op to feed values to a node"""

    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        pass


class VariableOp(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        new_node.value = None
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        assert tnode.value is not None
        assert len(input_vals) == 0
        return tnode.value


class Const_Op(Op):
    def __call__(self, value):
        new_node = Op.__call__(self)
        new_node.const_attr = value  # should be an instance of np.ndarray??
        Const_Op.name = str(value)
        return new_node

    def compute(self, tnode, input_vals):
        return tnode.const_attr

    def gradients(self, tnode, output_grad):
        return None


class Oneslike_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        # new_node.name = "Oneslike(%s)" % node_a.name
        return new_node

    def gradients(self, tnode, output_grad):
        return [zerolike_op(tnode.inputs[0])]

    def compute(self, tnode, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)


class Zerolike_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        # new_node.name = "Oneslike(%s)" % node_a.name
        return new_node

    def gradients(self, tnode, output_grad):
        return [zerolike_op(tnode.inputs[0])]

    def compute(self, tnode, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)


class ReduceSum_Op(Op):
    def __call__(self, node_a, reduction_indices=0, keep_dims=False):
        assert isinstance(reduction_indices, int)
        new_node = Op.__call__(self)
        # new_node.name = "reduce_sum(%s)" % node_a.name
        new_node.inputs = [node_a]
        new_node.reduction_indices = reduction_indices
        new_node.keep_dims = keep_dims
        return new_node

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return np.sum(input_vals[0], axis=tnode.reduction_indices,
                      keepdims=tnode.keep_dims)

    def gradients(self, tnode, output_grad):
        return [broadcastto_op(output_grad, tnode.inputs[0])]


class Init_Op(Op):
    def __call__(self, all_variables):
        new_node = Op.__call__(self)
        # new_node.name = "Initializer"
        new_node.inputs = all_variables
        return new_node

    def compute(self, tnode, input_vals):
        return


class Broadcastto_Op(Op):
    def __call__(self, node_A, node_B):
        """
        Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        tmp = input_vals[0]
        input_shape = ()
        output_shape = input_vals[1].shape
        j = 0
        for i in range(len(output_shape)):
            if j < len(tmp.shape) and output_shape[i] == tmp.shape[j]:
                input_shape = input_shape + (tmp.shape[j],)
                j = j + 1
            else:
                input_shape = input_shape + (1,)
        tmp = tmp.reshape(input_shape)
        return np.broadcast_to(tmp, output_shape)

    def gradients(self, node, output_grad):
        grad_a = reduce_sum_to(output_grad, node.inputs[0])
        grad_b = zerolike_op(node.inputs[1])
        return [grad_a, grad_b]


class Assign_Op(Op):
    def __call__(self, assign_node, value):
        input_node = const_op(value) if not isinstance(value, Node) else value
        assert isinstance(assign_node, Node)
        new_node = Op.__call__(self)
        new_node.inputs = [input_node]
        new_node.assign_to = assign_node
        # new_node.name = "Assign %s to %s" % (input_node.name, assign_node.name)
        return new_node

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        tnode.assign_to.value = input_vals[0]
        return input_vals[0]

    def gradients(self, tnode, output_grad):
        assert False, "No gradient for assign node"


class Shape_Op(Op):
    def __call__(self, node_a, reduction_indices):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.reduction_indices = reduction_indices
        if not isinstance(new_node.reduction_indices, list):
            new_node.reduction_indices = [reduction_indices]
        return new_node

    def gradients(self, tnode, output_grad):
        return [0]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        shape = np.shape(input_vals[0])
        if len(tnode.reduction_indices) == 1:
            if tnode.reduction_indices[0] is None:
                num = 1
                for i in range(len(shape)):
                    num *= shape[i]
                return num
            return shape[tnode.reduction_indices[0]]
        else:
            num = 1
            for it in tnode.reduction_indices:
                num = num * shape[it]
            return num


class ReduceSumToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        tmp = input_vals[0]
        output_shape = input_vals[1].shape
        for i in range(len(output_shape)):
            while (i < len(tmp.shape) and
                   tmp.shape[i] != output_shape[i]):
                tmp = np.sum(tmp, axis=i)
        while len(tmp.shape) < len(output_shape):
            tmp = tmp.reshape(tmp.shape + (1,))
        assert tmp.shape == output_shape
        return tmp

    def gradients(self, node, output_grad):
        grad_A = broadcastto_op(output_grad, node.inputs[0])
        grad_B = zerolike_op(node.inputs[1])
        return [grad_A, grad_B]


class Argmax_Op(Op):
    def __call__(self, node_a, axis):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.reduction_indices = axis
        return new_node

    def gradients(self, tnode, output_grad):
        return [0]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return np.argmax(input_vals[0], axis=tnode.reduction_indices)


class Equal_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        # new_node.name = "(%s == %s)" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 2
        return np.equal(input_vals[0], input_vals[1])


class Conv2D_Op(Op):

    def __call__(self, input, filter, strides, padding):
        new_node = Op.__call__(self)
        new_node.inputs = [input, filter]
        new_node.padding = padding
        new_node.strides = strides
        return new_node

    def gradients(self, tnode, output_grad):
        return [
            conv2dgradientx_op(tnode.inputs[0], tnode.inputs[1], output_grad, tnode),
            conv2dgradientw_op(tnode.inputs[0], tnode.inputs[1], output_grad, tnode)
        ]

    def compute(self, tnode, input_vals):
        input, filter = input_vals
        import math
        (f, f, n_C_prev, n_C) = filter.shape
        if tnode.padding == "SAME":
            pad_h = (input.shape[0] - 1) * tnode.strides[1] + f - input.shape[0]
            pad_w = (input.shape[1] - 1) * tnode.strides[2] + f - input.shape[1]
            pad_t = tnode.pad_t = pad_h // 2
            pad_b = tnode.pad_b = pad_h - pad_t
            pad_l = tnode.pad_l = pad_w // 2
            pad_r = tnode.pad_r = pad_w - pad_l
            A_pad = np.pad(input, ((0, 0), (pad_t, pad_b), (pad_l, pad_r),
                                   (0, 0)), "constant")
        if tnode.padding == "VALID":
            A_pad = input

        tnode.X_pad = A_pad
        (m, n_H_prev, n_W_prev, n_C_prev) = A_pad.shape
        n_H = math.floor((n_H_prev - f) / tnode.strides[1] + 1)
        n_W = math.floor((n_W_prev - f) / tnode.strides[2] + 1)
        ans = np.zeros([m, n_H, n_W, n_C])
        if use_cpp:
            A = A_pad.astype(np.float32)
            filter = filter.astype(np.float32)
            ans = ans.astype(np.float32)
            A_in = A.ctypes.data_as(POINTER(c_float))
            ans_in = ans.ctypes.data_as(POINTER(c_float))
            filter_in = filter.ctypes.data_as(POINTER(c_float))
            lib.conv2d(
                A_in, filter_in, ans_in,
                m, n_H, n_W, n_C,
                n_H_prev, n_W_prev, n_C_prev,
                f, tnode.strides[1], tnode.strides[2]
            )
            return ans
        else:
            """Reduce 4 loops to 3 loops by img2col"""
            A_sub_col = np.zeros([n_H * n_W, f * f * n_C_prev])
            W_col = filter.reshape((f * f * n_C_prev, n_C))
            for i in range(m):
                for h in range(n_H):
                    for w in range(n_W):
                        A_sub = A_pad[i, h * tnode.strides[1]:h * tnode.strides[1] + f,
                                w * tnode.strides[2]:w * tnode.strides[2] + f, :]
                        A_sub_col[h * n_W + w, :] = \
                            A_sub.reshape((1, f * f * n_C_prev))
                Y_sub_col = np.matmul(A_sub_col, W_col)
                ans[i, :] = Y_sub_col.reshape([n_H, n_W, n_C])
            return ans


# TODO: finish the iteration
class Conv2DGradientXOp(Op):
    """
    dW = sum sum a_slice * dH[h,w]
    dX_slice = sum sum W * dH[h,w]
    now I finally understand this trick
    """
    def __call__(self, X, W, dH, src_node):
        new_node = Op.__call__(self)
        new_node.inputs = [X, W, dH]
        new_node.src_node = src_node
        return new_node

    def gradients(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        X, W, dH = input_vals
        dX = np.zeros(X.shape)
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        f, f, n_C_prev, n_C = W.shape
        m, n_H, n_W, n_C = dH.shape
        _, stride1, stride2, _ = tnode.src_node.strides
        W_col = np.reshape(W, (f * f * n_C_prev, n_C))
        dH_col = np.reshape(dH, (m * n_H * n_W, n_C))
        """
        First compute gradient for X_pad, then pad back 
        to the original size
        """
        if tnode.src_node.padding == "VALID":
            dX = np.zeros(X.shape)
            for i in range(m):
                dH_sub_col = dH_col[i * n_H * n_W: (i + 1) * n_H * n_W, :]
                dX_col = np.matmul(dH_sub_col, W_col.T)
                for h in range(n_H):
                    for w in range(n_W):
                        dX[i, h * stride1:h * stride1 + f, w * stride2:w * stride2 + f, :] += \
                            np.reshape(dX_col[h * n_W + w, :], [f, f, n_C])
        elif tnode.src_node.padding == "SAME":
            X_pad = tnode.src_node.X_pad
            m, n_H_prev, n_W_prev, n_C_prev = X_pad.shape
            dX_pad = np.zeros(X_pad.shape)
            if use_cpp:
                dX_pad = dX_pad.astype(np.float32)
                W = W.astype(np.float32)
                dH = dH.astype(np.float32)
                dX_in = dX_pad.ctypes.data_as(POINTER(c_float))
                W_in = W.ctypes.data_as(POINTER(c_float))
                dH_in = dH.ctypes.data_as(POINTER(c_float))
                lib.Conv2dGradientX(
                    dX_in, dH_in, W_in, m,
                    n_H_prev, n_W_prev, n_C_prev,
                    n_H, n_W, n_C,
                    f, stride1, stride2
                )

            else:
                for i in range(m):
                    dH_sub_col = dH_col[i * n_H * n_W: (i + 1) * n_H * n_W, :]
                    dX_col = np.matmul(dH_sub_col, W_col.T)
                    for h in range(n_H):
                        for w in range(n_W):
                            dX_pad[i, h * stride1:h * stride1 + f, w * stride2:w * stride2 + f, :] += \
                                np.reshape(dX_col[h * n_W + w, :], [f, f, n_C_prev])

            dX = dX_pad[:, tnode.src_node.pad_t: X_pad.shape[1] - tnode.src_node.pad_b,
                 tnode.src_node.pad_l: X_pad.shape[2] - tnode.src_node.pad_r, :]
        assert X.shape == dX.shape
        return dX


class Conv2DGradientWOp(Op):
    """
    dW = sum sum a_slice * dH[h,w]
    dX_slice = sum sum W * dH[h,w]
    now I finally understand this trick
    """
    def __call__(self, X, W, dH, src_node):
        new_node = Op.__call__(self)
        new_node.inputs = [X, W, dH]
        new_node.src_node = src_node
        return new_node

    def gradients(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        X, W, dH = input_vals
        dW = np.zeros(W.shape)
        f, f, n_C_prev, n_C = W.shape
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        m, n_H, n_W, n_C = dH.shape
        _, stride1, stride2, _ = tnode.src_node.strides
        if tnode.src_node.padding == "SAME":
            X = tnode.src_node.X_pad
            m, n_H_prev, n_W_prev, n_C_prev = X.shape
        if use_cpp:
                dW = dW.astype(np.float32)
                dH = dH.astype(np.float32)
                X = X.astype(np.float32)
                dW_in = dW.ctypes.data_as(POINTER(c_float))
                dH_in = dH.ctypes.data_as(POINTER(c_float))
                X_in = X.ctypes.data_as(POINTER(c_float))
                lib.Conv2dGradientW(
                    X_in, dH_in, dW_in,
                    m, n_H_prev, n_W_prev, n_C_prev,
                    n_H, n_W, n_C,
                    f, stride1, stride2
                )
                return dW
        dW_col = np.zeros((f * f * n_C_prev, n_C))
        X_sub_col = np.zeros((n_H * n_W, n_C_prev * f * f))
        if tnode.src_node.padding == "VALID":
            for i in range(m):
                for h in range(n_H):
                    for w in range(n_W):
                        for c in range(n_C):
                            dW[:, :, :, c] += np.multiply(
                                X[i, h * stride1:h * stride1 + f, w * stride2:w * stride2 + f, :],
                                dH[i, h, w, c]
                            )
        elif tnode.src_node.padding == "SAME":
            X_pad = tnode.src_node.X_pad
            for i in range(m):
                dH_sub = dH[i].reshape([n_W * n_H, n_C])
                for h in range(n_H):
                    for w in range(n_W):
                        X_sub = X_pad[i, h * stride1:h * stride1 + f, w * stride2:w * stride2 + f, :]
                        X_sub_col[h * n_W + w, :] = X_sub.reshape([f * f * n_C_prev])
                dW_col[:] += np.matmul(X_sub_col.T, dH_sub)
        dW = dW_col.reshape([f, f, n_C_prev, n_C])
        assert W.shape == dW.shape
        return dW


class MaxPool_Op(Op):
    def __call__(self, input, ksize, strides, padding):
        new_node = Op.__call__(self)
        new_node.inputs = [input]
        new_node.ksize = ksize
        new_node.padding = padding
        new_node.strides = strides
        return new_node

    def gradients(self, tnode, output_grad):
        return [maxpoolgradient_op(tnode.inputs[0], tnode, output_grad)]

    def compute(self, tnode, input_vals):
        input = input_vals[0]
        import math
        (n_C_prev, f, f, n_C) = tnode.ksize
        A_pad = input
        (m, n_H_prev, n_W_prev, n_C_prev) = input.shape
        if tnode.padding == "VALID":
            n_H = math.floor((n_H_prev - f) / tnode.strides[1] + 1)
            n_W = math.floor((n_W_prev - f) / tnode.strides[2] + 1)
        elif tnode.padding == "SAME":
            n_H = math.ceil(1.00 * n_H_prev / tnode.strides[1])
            n_W = math.ceil(1.00 * n_W_prev / tnode.strides[2])
        ans = np.ones([m, n_H, n_W, n_C_prev])
        if use_cpp:
            Y = ans.astype(np.float32)
            X = input.astype(np.float32)
            X_in = X.ctypes.data_as(POINTER(c_float))
            Y_in = Y.ctypes.data_as(POINTER(c_float))
            lib.maxpool(
                X_in, Y_in,
                m, n_H_prev, n_W_prev, n_C_prev,
                n_H, n_W, n_C,
                f, tnode.strides[1], tnode.strides[2]
            )
            return Y
        for i in range(n_H):
            for j in range(n_W):
                ans[:, i, j, :] = np.max(A_pad[:, i * tnode.strides[1]:i * tnode.strides[1] + f,
                                         j * tnode.strides[2]:j * tnode.strides[2] + f, :],
                                         axis=(1, 2))
        return ans


class MaxPoolGradient_Op(Op):
    def __call__(self, input, tnode, outputgrad):
        new_node = Op.__call__(self)
        new_node.inputs = [input, outputgrad]
        new_node.src_node = tnode
        return new_node

    def compute(self, tnode, input_vals):
        X, dH = input_vals
        m, n_H_prev, n_W_prev, n_C_prev = X.shape
        m, n_H, n_W, n_C = dH.shape
        padding, ksize = tnode.src_node.padding, tnode.src_node.ksize
        _, f, f, _ = ksize
        stride1, stride2 = tnode.src_node.strides[1], tnode.src_node.strides[2]
        dX = np.zeros(X.shape)
        if use_cpp:
            dX = dX.astype(np.float32)
            X = X.astype(np.float32)
            dH = dH.astype(np.float32)
            dX_in = dX.ctypes.data_as(POINTER(c_float))
            X_in = X.ctypes.data_as(POINTER(c_float))
            dH_in = dH.ctypes.data_as(POINTER(c_float))
            lib.maxpoolgradient(
                X_in, dH_in, dX_in,
                m, n_H_prev, n_W_prev, n_C_prev,
                n_H, n_W, n_C,
                f, stride1, stride2
            )
            return dX
        for h in range(n_H):
            for w in range(n_W):
                subX = X[:, h * stride1: h * stride1 + ksize[1], w * stride2: w * stride2 + ksize[2], :]
                subdX = dX[:, h * stride1: h * stride1 + ksize[1], w * stride2: w * stride2 + ksize[2], :]
                subdX[:] += np.equal(subX, np.max(subX, axis=(1, 2), keepdims=True)) * \
                            dH[:, h: h + 1, w: w + 1, :]
        return dX

    def gradients(self, tnode, output_grad):
        pass


class DropOut_Op(Op):
    def __call__(self, input, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [input, keep_prob]
        return new_node

    def gradients(self, tnode, output_grad):
        return [dropoutgradient_op(tnode, output_grad), 0]

    def compute(self, tnode, input_vals):
        arr = np.random.random(input_vals[0].shape)
        tnode.keep_status = arr <= input_vals[1]
        return input_vals[0] * tnode.keep_status


class DropOutGradient_Op(Op):
    def __call__(self, src_node, dH):
        new_node = Op.__call__(self)
        new_node.inputs = [dH]
        new_node.src_node = src_node
        return new_node

    def gradients(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        return input_vals[0] * tnode.src_node.keep_status


class Relu_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        return new_node

    def gradients(self, tnode, output_grad):
        return [relugradient_op(tnode.inputs[0], output_grad)]

    def compute(self, tnode, input_vals):
        # maximum is elementwise
        if use_cpp:
            input = input_vals[0].astype(np.float32)
            shape = input_vals[0].shape
            size = input.size
            output = np.ndarray(shape=shape, dtype=np.float32)
            in_data = input.ctypes.data_as(POINTER(c_float))
            out_data = output.ctypes.data_as(POINTER(c_float))
            lib.relu(in_data, out_data, size)
            return output
        return np.maximum(input_vals[0], 0)


class ReluGradient_Op(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        if not use_cpp:
            shape = input_vals[0].shape
            size = input_vals[0].size
            x = input_vals[0].astype(np.float32)
            grad = input_vals[1].astype(np.float32)
            x_in = x.ctypes.data_as(POINTER(c_float))
            grad_in = grad.ctypes.data_as(POINTER(c_float))
            output = np.ndarray(shape=shape, dtype=np.float32)
            out_data = output.ctypes.data_as(POINTER(c_float))
            lib.relugradient(x_in, grad_in, out_data, size)
            return output
        return (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

    def gradient(self, node, output_grad):
        raise NotImplementedError


class Reshape_Op(Op):
    def __call__(self, node_a, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.shape_to = shape
        return new_node

    def gradients(self, tnode, output_grad):
        return [reshapegradient_op(output_grad, tnode.inputs[0])]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return np.reshape(input_vals[0], tnode.shape_to)


class ReshapeGradient_Op(Op):
    def __call__(self, output_grad, input):
        new_node = Op.__call__(self)
        new_node.inputs = [output_grad, input]
        return new_node

    def gradients(self, tnode, output_grad):
        return [reshapegradient_op(output_grad, tnode.inputs[0])]

    def compute(self, tnode, input_vals):
        dH, X = input_vals
        return np.reshape(dH, X.shape)


class Executor(object):
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list
        self.ValueNode = dict()

    def run(self, feed_dict):
        """Core function to compute the gradients"""
        self.ValueNode = dict(feed_dict)
        TopoOrder = find_topo_sort(self.eval_node_list)
        for node in TopoOrder:
            if isinstance(node.op, PlaceholderOp):
                continue
            vals = [self.ValueNode[subnode] for subnode in node.inputs] \
                if not isinstance(node.op, Const_Op) else []
            res = node.op.compute(node, vals)
            self.ValueNode[node] = res if isinstance(res, np.ndarray) else np.array(res)
        ans = [self.ValueNode[node] for node in self.eval_node_list]
        return ans


def gradients(node_y, node_x_list):
    PartialList = {node_y: [oneslike_op(node_y)]}
    NodeToGrad = {}
    ReverseTopoOrder = reversed(find_topo_sort([node_y]))
    for node in ReverseTopoOrder:
        grad = sum_parital(PartialList[node])
        NodeToGrad[node] = grad
        grads = node.op.gradients(node, grad)
        for i in range(len(node.inputs)):
            son = node.inputs[i]
            grad_list = PartialList.get(son, [])
            grad_list.append(grads[i])
            PartialList[son] = grad_list

    ans = [NodeToGrad[node] for node in node_x_list]
    return ans


equal_op = Equal_Op()
reshape_op = Reshape_Op()
reshapegradient_op = ReshapeGradient_Op()
conv2d_op = Conv2D_Op()
conv2dgradientx_op = Conv2DGradientXOp()
conv2dgradientw_op = Conv2DGradientWOp()
maxpool_op = MaxPool_Op()
maxpoolgradient_op = MaxPoolGradient_Op()
dropout_op = DropOut_Op()
dropoutgradient_op = DropOutGradient_Op()
placeholder_op = PlaceholderOp()
shape_op = Shape_Op()
broadcastto_op = Broadcastto_Op()
relu = Relu_Op()
relugradient_op = ReluGradient_Op()
Variable = VariableOp()
power_op = Power_Op()
init_op = Init_Op()
const_op = Const_Op()
assign = Assign_Op()
argmax_op = Argmax_Op()
mul_op = Mul_Op()
mulconst_op = MulConst_Op()
add_op = Add_Op()
addconst_op = AddConst_Op()
sub_op = Sub_Op()
subconst_op = SubConst_Op()
rsubconst_op = RSubConst_Op()
div_op = Div_Op()
divconst_op = DivConst_Op()
rdiv_op = RDiv_Op()
rdivconst_op = RDivConst_Op()
neg_op = Neg_Op()
inv = Inv_Op()
oneslike_op = Oneslike_Op()
zerolike_op = Zerolike_Op()
matmul_op = MatMul_Op()
log = Log_Op()
exp = Exp_Op()
reduce_sum = ReduceSum_Op()
reduce_sum_to = ReduceSumToOp()


def find_topo_sort(node_list):
    """
    Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_parital(node_list):
    return reduce(add, node_list)
