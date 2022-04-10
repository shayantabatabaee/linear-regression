from sklearn.datasets import make_regression


class LinearDataset:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def generate(self):
        return make_regression(self.n_samples, 1, n_informative=1, n_targets=1, noise=10, random_state=1)
