import numpy as np

from .common.functions import softmax, cross_entropy_error
from .common.gradient import numerical_gradient


class SimpleNet:
    def __init__(self):
        # ガウス分布による初期化
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
