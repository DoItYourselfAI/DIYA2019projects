import numpy as np


class ArgumentParser(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def onehot(array, max_index):
    assert array.ndim == 1
    out = np.zeros((len(array), max_index))
    out[np.arange(len(array)), array] = 1
    return out
