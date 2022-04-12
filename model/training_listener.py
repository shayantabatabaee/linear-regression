import numpy as np


class TrainingListener:

    def on_training_start(self):
        pass

    def on_epoch_end(self, epoch: int, loss: np.ndarray, A: np.ndarray, Y_hat: np.ndarray):
        pass

    def on_training_end(self, loss: np.ndarray):
        pass
