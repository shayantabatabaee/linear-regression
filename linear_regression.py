from dataset.regression import RegressionDataset

NUMBER_OF_SAMPLES = 1000


def train():
    dataset = RegressionDataset(NUMBER_OF_SAMPLES)
    X, Y = dataset.generate()


if __name__ == '__main__':
    train()
