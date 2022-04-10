from dataset.linear import LinearDataset

NUMBER_OF_SAMPLES = 1000


def train():
    dataset = LinearDataset(NUMBER_OF_SAMPLES)
    X, Y = dataset.generate()


if __name__ == '__main__':
    train()
