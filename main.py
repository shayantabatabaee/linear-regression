from dataset.regression import RegressionDataset
from model.linear_regression import LinearRegressionModel

NUMBER_OF_SAMPLES = 1000
NUMBER_OF_FEATURES = 1
NUMBER_OF_TARGETS = 1
EPOCH = 2000
LEARNING_RATE = 0.01


def train():
    dataset = RegressionDataset(n_samples=NUMBER_OF_SAMPLES, n_features=NUMBER_OF_FEATURES, n_targets=NUMBER_OF_TARGETS)
    X, Y = dataset.generate()

    model = LinearRegressionModel(NUMBER_OF_FEATURES, NUMBER_OF_TARGETS)
    model.fit(X=X, Y=Y, epoch=EPOCH, learning_rate=LEARNING_RATE)


if __name__ == '__main__':
    train()
