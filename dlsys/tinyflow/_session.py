import numpy as np
from . import autodiff
from ._base import *


class Session(object):
    def __enter__(self):
        return self

    def __exit__(self, e_t, e_v, t_b):
        return None

    def run(self, fetch, feed_dict=None):
        if not isinstance(fetch, list):
            fetch = [fetch]
        if not isinstance(feed_dict, dict):
            feed_dict = {}
        # change dict to list, change value to np.array
        for node in feed_dict:
            value = feed_dict[node]
            if not isinstance(value, np.ndarray):
                if not isinstance(value, list):
                    value = [value]
                value = np.array(value)
            feed_dict[node] = value

        executor = autodiff.Executor(fetch)
        res = executor.run(feed_dict)
        for i in range(len(res)):
            if res[i].shape == (1,):
                res[i] = res[i][0]
        if len(res) == 1:
            return res[0]
        else:
            return res

