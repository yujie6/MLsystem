class Optimizer(object):
    learning_rate = None
    optimize_target = None

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        assert False

    def minimize(self, node):
        assert False


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate):
        Optimizer.__init__(self, learning_rate=learning_rate)

    def minimize(self, node):
        self.optimize_target = node
        return


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate):
        Optimizer.__init__(self, learning_rate)

    def minimize(self, node):
        self.optimize_target = node
        return
