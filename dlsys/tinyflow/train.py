from . import autodiff
from ._base import *
from ._func import *


class Optimizer(object):
    learning_rate = None
    optimize_target = None

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, cost_function):
        assert False


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001):
        Optimizer.__init__(self, learning_rate)

    def minimize(self, cost_function):
        self.optimize_target = cost_function
        assert isinstance(cost_function, autodiff.Node)
        topo_order = autodiff.find_topo_sort([cost_function])
        para = []
        for node in topo_order:
            if isinstance(node.op, autodiff.VariableOp):
                para.append(node)
        grad = gradients(cost_function, para)
        assign_nodes = []
        for i in range(len(para)):
            assign_nodes.append(assign(para[i],
                                       para[i] - self.learning_rate * grad[i]))
        optimizer = autodiff.init_op(assign_nodes)
        # use function assign to update
        return optimizer


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-08):
        Optimizer.__init__(self, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(self, node):
        self.optimize_target = node
        return
