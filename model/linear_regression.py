import time

import numpy as np

from model.training_listener import TrainingListener
from utils.loss import Loss


class LinearRegressionModel:

    def __init__(self, n_features: int, n_targets: int, training_listener: TrainingListener = None):
        self.n_features = n_features
        self.n_targets = n_targets
        self.training_listener = training_listener

    def fit(self, X: np.ndarray, Y: np.ndarray, total_epoch: int, learning_rate: float):
        self.training_listener is not None and self.training_listener.on_training_start()
        self.__validate(X, Y)
        n_samples = X.shape[0]
        A = np.random.uniform(size=(self.n_features + 1, self.n_targets))
        X = np.hstack((np.ones([n_samples, 1], dtype=X.dtype), X))
        Y = np.reshape(Y, (n_samples, self.n_targets))
        losses = np.zeros((0, self.n_targets))
        for i in range(0, total_epoch):
            time.sleep(0.01)  # Just for show the training process
            Y_hat = self.evaluate(X, A)
            A = self.__one_step_gradient_descent(A, Y, Y_hat, X, learning_rate)
            loss = Loss.mse(Y, Y_hat)[:, np.newaxis].T
            losses = np.vstack((losses, loss))
            self.training_listener is not None and self.training_listener.on_epoch_end(i, loss, A, Y_hat)
        self.training_listener is not None and self.training_listener.on_training_end(losses)

    def evaluate(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.matmul(X, A)

    def __one_step_gradient_descent(self,
                                    A: np.ndarray,
                                    Y: np.ndarray,
                                    Y_hat: np.ndarray,
                                    X: np.ndarray,
                                    learning_rate: float) -> np.ndarray:
        gradient = np.zeros_like(A)
        for i in range(0, self.n_targets):
            gradient[:, i] = np.mean(((Y_hat - Y)[:, i, np.newaxis]) * X, axis=0)
        return A - learning_rate * gradient

    def __validate(self, X: np.ndarray, Y: np.ndarray):
        n_samples = X.shape[0]
        if X.shape != (n_samples, self.n_features):
            raise Exception('Incompatible shape of X, shape must be: ({}, {})'.format(n_samples, self.n_features))
        if Y.shape != (n_samples,) and Y.shape != (n_samples, self.n_targets):
            raise Exception('Incompatible shape of Y, shape must be: ({}, 1) or ({}, )'.format(n_samples, n_samples))
