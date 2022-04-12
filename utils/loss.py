import numpy as np


class Loss:

    @staticmethod
    def mse(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        return np.mean(np.power(Y_hat - Y, 2), axis=0)
