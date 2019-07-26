import numpy as np
from operator import add
from functools import reduce
from ._session import *


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

    def eval(self, feed_dict):
        with Session() as sess:
            return sess.run(self, feed_dict=feed_dict)

    def run(self, feed_dict):
        return self.eval(feed_dict=feed_dict)

    __radd__ = __add__
    __rmul__ = __mul__


class Op(object):
    """
    Operation in the graph, such as mul, add, exp...
    """

    def __call__(self):
        new_node = Node()
        new_node.op = self
        new_node.name = ""
        return new_node

    def gradients(self, tnode, output_grad):
        assert False

    def compute(self, tnode, input_vals):
        assert False


class Mul_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s * %s" % (node_a.name, node_b.name)
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
        new_node.name = "%s * %s" % (node_a.name, str(const_val))
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
        new_node.name = "%s + %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        # return [output_grad, output_grad]
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
        new_node.name = "%s + %s" % (node_a.name, str(const_val))
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
        new_node.name = "-%s" % node_a.name
        return new_node

    def compute(self, tnode, input_vals):
        return -input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-output_grad]


class Sub_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s - %s" % (node_a.name, node_b.name)
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
        new_node.name = "%s - %s" % (node_a.name, str(const_val))
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
        new_node.name = "%s - %s" % (str(const_val), node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return tnode.const_attr - input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-reduce_sum_to(output_grad, tnode.inputs[0])]


class Div_Op(Op):
    """
    the computation of grad is relatively complex, it is ok for a graph
    """

    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s / %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(output_grad / tnode.inputs[1], tnode.inputs[0]),
                reduce_sum_to(-output_grad * tnode.inputs[0] / (tnode.inputs[1] * tnode.inputs[1])
                              , tnode.inputs[1])]

    def compute(self, tnode, input_vals):
        # try:
        #     assert not np.equal(input_vals[1].all(), 0)
        # except AssertionError:
        #     print("shit!!!!", end=" ")
        #     print(input_vals[1])
        # else:
        #     print("ok")
        return input_vals[0] / input_vals[1]


class DivConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_a]
        new_node.name = "%s / %s" % (node_a.name, str(const_val))
        return new_node

    def compute(self, tnode, input_vals):
        return input_vals[0] / tnode.const_attr

    def gradients(self, tnode, output_grad):
        return [output_grad / tnode.const_attr]


class RDiv_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s / %s" % (node_b.name, node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return input_vals[1] / input_vals[0]

    def gradients(self, tnode, output_grad):
        return [reduce_sum_to(-output_grad * tnode.inputs[1] / (tnode.inputs[0] * tnode.inputs[0])
                              , tnode.inputs[0]),
                reduce_sum_to(output_grad / tnode.inputs[0], tnode.inputs[1])]


class RDivConst_Op(Op):
    def __call__(self, node_a, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_a]
        new_node.name = "%s / %s" % (str(const_val), node_a.name)
        return new_node

    def compute(self, tnode, input_vals):
        return tnode.const_attr / input_vals[0]

    def gradients(self, tnode, output_grad):
        return [-output_grad * tnode.const_attr / (tnode.inputs[0] * tnode.inputs[0])]


class MatMul_Op(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "matmul(%s, %s, %s, %s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))
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
        """Given values of input nodes, return result of matrix multiplication."""
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
        new_node.name = "exp(%s)" % (node_a.name)
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
        new_node.name = "log(%s)" % node_a.name
        return new_node

    def compute(self, tnode, input_vals):
        return np.log(input_vals[0])

    def gradients(self, tnode, output_grad):
        return [output_grad / tnode.inputs[0]]


class Power_Op(Op):
    def __call__(self, node_a, power):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, power]
        new_node.name = "%s ^ %s" % (node_a.name, str(power))
        return new_node

    def compute(self, tnode, input_vals):
        return np.power(input_vals[0], input_vals[1])

    def gradients(self, tnode, output_grad):
        return [output_grad * tnode.inputs[1] * power_op(tnode.inputs[0], tnode.inputs[1] - 1)
            , output_grad * log(tnode.inputs[0]) * tnode]
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
        new_node.name = "Oneslike(%s)" % node_a.name
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
        new_node.name = "Oneslike(%s)" % node_a.name
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
        new_node.name = "reduce_sum(%s)" % node_a.name
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
        new_node.name = "Initializer"
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
        if not isinstance(value, Node):
            input_node = const_op(value)
        else:
            input_node = value
        assert isinstance(assign_node, Node)
        new_node = Op.__call__(self)
        new_node.inputs = [input_node]
        new_node.assign_to = assign_node
        new_node.name = "Assign %s to %s" % (input_node.name, assign_node.name)
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
        new_node.name = "(%s == %s)" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 2
        return np.equal(input_vals[0], input_vals[1])


class Conv2D_Op(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        pass


class MaxPool_Op(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        pass


class DropOut_Op(Op):
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        pass


class Relu_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        return new_node

    def gradients(self, tnode, output_grad):
        return [relugradient_op(tnode.inputs[0], output_grad)]

    def compute(self, tnode, input_vals):
        # maximum is elementwise
        return np.maximum(input_vals[0], 0)


class ReluGradient_Op(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        # heaviside function, 0.5 at x=0
        return (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]

    def gradient(self, node, output_grad):
        raise NotImplementedError


class Reshape(Op):
    def __call__(self, node_a, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.shape = shape
        return new_node

    def gradients(self, tnode, output_grad):
        return [1]

    def compute(self, tnode, input_vals):
        assert len(input_vals) == 1
        return np.reshape(input_vals[0], tnode.shape)


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
            if not isinstance(node.op, Const_Op):
                vals = [self.ValueNode[subnode] for subnode in node.inputs]
            else:
                vals = []
            res = node.op.compute(node, vals)
            self.ValueNode[node] = res if isinstance(res, np.ndarray) else np.array(res)

        ans = [self.ValueNode[node] for node in self.eval_node_list]
        return ans


def gradients(node_y, node_x_list):
    """
    Core function
    :param node_y:
    :param node_x_list:
    :return: [partial_y/partial_xi ]
    """
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
conv2d_op = Conv2D_Op()
maxpool_op = MaxPool_Op()
dropout_op = DropOut_Op()
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
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_parital(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    return reduce(add, node_list)


# TODO: pass test 7
# TODO: pass test 8
# TODO: pass test 9
# TODO: pass test 10
