from sklearn.datasets import make_regression


class RegressionDataset:

    def __init__(self, n_samples: int, n_features: int = 1, n_informative: int = 1, n_targets: int = 1):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_targets = n_targets

    def generate(self):
        return make_regression(n_samples=self.n_samples,
                               n_features=self.n_features,
                               n_informative=self.n_informative,
                               n_targets=self.n_targets,
                               noise=10,
                               random_state=1)
