import numpy as np
import datetime

from dataset.regression import RegressionDataset
from model.linear_regression import LinearRegressionModel
from model.training_listener import TrainingListener

NUMBER_OF_SAMPLES = 1000
NUMBER_OF_FEATURES = 1
NUMBER_OF_TARGETS = 1
TOTAL_EPOCH = 1000
LEARNING_RATE = 0.01


class Train(TrainingListener):

    def do(self):
        dataset = RegressionDataset(n_samples=NUMBER_OF_SAMPLES,
                                    n_features=NUMBER_OF_FEATURES,
                                    n_targets=NUMBER_OF_TARGETS)
        X, Y = dataset.generate()

        model = LinearRegressionModel(NUMBER_OF_FEATURES, NUMBER_OF_TARGETS, self)
        model.fit(X=X, Y=Y, total_epoch=TOTAL_EPOCH, learning_rate=LEARNING_RATE)

    def on_training_start(self):
        print("")
        print("[{} - Training starts]".format(datetime.datetime.now().strftime("%H:%M:%S")))

    def on_epoch_end(self, epoch: int, loss: np.ndarray, A: np.ndarray, Y_hat: np.ndarray):
        print("[{} - Epoch:{} ,Loss: {}]".format(datetime.datetime.now().strftime("%H:%M:%S"), epoch, loss), end='\r')

    def on_training_end(self, losses):
        print("[{} - Total Epoch: {}, Final Loss: {}]".format(
            datetime.datetime.now().strftime("%H:%M:%S"),
            losses.shape[0],
            losses[-1]))


if __name__ == '__main__':
    train = Train()
    train.do()
