import numpy as np


class Node(object):
    """
    node in the computation graph
    """
    def __init__(self):
        self.Op = None
        self.name = ""
        self.inputs = []

    def __add__(self, other):
        new_add_node = Add_Op(self, other)
        return new_add_node

    def __sub__(self, other):
        if isinstance(self, other):
            return Sub_Op(self, other)
        else:
            return SubConst_Op(self, other)



class Op(object):
    """
    Operation in the graph, such as mul, add, exp...
    """
    def __call__(self):
        new_node = Node()
        new_node.Op = self
        new_node.name = ""
        return new_node

    def gradient(self, tnode, output_grad):
        assert False

    def compute(self, tnode, input_vals):
        assert False


class Mul_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s * %s" % (node_a.name, node_b.name)

    def gradient(self, tnode, output_grad):
        return [tnode.inputs[1] * output_grad, tnode.inputs[0] * output_grad]


class Add_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s + %s" % (node_a.namem, node_b.name)

    def gradient(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        pass


class Sub_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s - %s" % (node_a.name, node_b.name)

    def gradient(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        return np.subtract(input_vals[0], input_vals[1])

class SubConst_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s - %s" % (node_a.name, node_b.name)


class Div_Op(Op):
    def __call__(self, node_a, node_b):
        new_node = Op.__call__(self)
        new_node.inputs = [node_a, node_b]
        new_node.name = "%s / %s" % (node_a.namem, node_b.name)

    def gradient(self, tnode, output_grad):
        pass

    def compute(self, tnode, input_vals):
        pass



class PlaceholderOp(Op):
    """Op to feed values to a node"""
    def __call__(self):
        new_node = Op.__call__(self)
        return new_node

    def gradient(self, tnode, output_grad):
        return None


class Executor(object):
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        self.node_to_val_map = dict(feed_dict)



def Variablel(name):
    new_op = PlaceholderOp()
    new_op.name = name
    return new_op


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
