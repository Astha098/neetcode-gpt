import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:

        n = X.shape[0]

        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs):
            y_hat = X @ w + b

            dw = (2 / n) * (X.T @ (y_hat - y))
            db = (2 / n) * np.sum(y_hat - y)

            w = w - lr * dw
            b = b - lr * db

        return np.round(w, 5), round(b, 5)