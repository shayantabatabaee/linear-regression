import numpy as np


class LinearRegressionModel:

    def __init__(self, n_features: int, n_targets: int):
        self.n_features = n_features
        self.n_targets = n_targets

    def fit(self, X, Y, epoch, learning_rate):
        self.__validate(X, Y)
        n_samples = X.shape[0]
        A = np.random.uniform(size=(self.n_features + 1, self.n_targets))
        X = np.hstack((np.ones([n_samples, 1], dtype=X.dtype), X))
        Y = np.reshape(Y, (n_samples, self.n_targets))
        for i in range(0, epoch):
            Y_hat = self.evaluate(X, A)
            A = self.__one_step_gradient_descent(A, Y, Y_hat, X, learning_rate)

    def evaluate(self, X, A):
        return np.matmul(X, A)

    def __one_step_gradient_descent(self, A, Y, Y_hat, X, learning_rate):
        gradient = np.zeros_like(A)
        for i in range(0, self.n_targets):
            gradient[:, i] = np.mean(((Y_hat - Y)[:, i, np.newaxis]) * X, axis=0)
        return A - learning_rate * gradient

    def __validate(self, X, Y):
        n_samples = X.shape[0]
        if X.shape != (n_samples, self.n_features):
            raise Exception('Incompatible shape of X, shape must be: ({}, {})'.format(n_samples, self.n_features))
        if Y.shape != (n_samples,) and Y.shape != (n_samples, self.n_targets):
            raise Exception('Incompatible shape of Y, shape must be: ({}, 1) or ({}, )'.format(n_samples, n_samples))
