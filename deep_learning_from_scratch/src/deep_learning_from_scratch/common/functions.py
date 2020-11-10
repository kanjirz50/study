import numpy as np


def softmax(a):
    """オーバーフロー対策をしたソフトマックス
    """
    c = np.max(a)
    exp_a = np.exp(a - c) 

    return exp_a / np.sum(exp_a)


def cross_entropy_error(y, t):
    """交差エントロピー誤差
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size