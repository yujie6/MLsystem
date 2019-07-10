import numpy as np


class Node(object):
    """
    node in the computation graph
    """
    op = None
    name = ""
    inputs = []
    const_attr = None

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
        return [tnode.inputs[1] * output_grad, tnode.inputs[0] * output_grad]

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
        return [output_grad, output_grad]

    def compute(self, tnode, input_vals):
        return input_vals[0] + input_vals[1]

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
        return [output_grad, -output_grad]

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
        return [output_grad]


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
        return [-output_grad]


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
        return [output_grad / tnode.inputs[1],
            -output_grad * tnode.inputs[0] / (tnode.inputs[1] * tnode.inputs[1])]

    def compute(self, tnode, input_vals):
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
        return [-output_grad * tnode.inputs[1] / (tnode.inputs[0] * tnode.inputs[0]) ,
              output_grad / tnode.inputs[0] ]


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
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s * %s" % (node_a.name, node_b.name)
        return new_node

    def gradients(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        return np.matmul(input_vals[0], input_vals[1])


class PlaceholderOp(Op):
    """Op to feed values to a node"""
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradients(self, tnode, output_grad):
        return None

    def compute(self, tnode, input_vals):
        pass


class Oneslike_Op(Op):
    def __call__(self, node_a):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a]
        new_node.name = "Oneslike(%s)" % node_a.name
        return new_node

    def gradients(self, tnode, output_grad):
        return [zerolike_op(tnode.inputs[0])]

    def compute(self, tnode, input_vals):
        assert(isinstance(input_vals[0], np.ndarray))
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
            # print(node.name)
            vals = [self.ValueNode[subnode] for subnode in node.inputs]
            res = node.op.compute(node, vals)
            self.ValueNode[node] = res if isinstance(res, np.ndarray) else np.array(res)

        ans = [self.ValueNode[node] for node in self.eval_node_list]
        return ans


def Variable(name):
    new_op = PlaceholderOp()
    new_node = new_op()
    new_node.name = name
    return new_node


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
oneslike_op = Oneslike_Op()
zerolike_op = Zerolike_Op()
matmul_op = MatMul_Op()


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
    from operator import add
    from functools import reduce
    return reduce(add, node_list)
