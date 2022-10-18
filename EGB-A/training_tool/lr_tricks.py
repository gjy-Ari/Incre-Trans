import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


class LearningRateFinder(keras.callbacks.Callback):
    def __init__(self, lr_min=1e-7, lr_max=0.5, number_of_batches=120):
        super(LearningRateFinder, self).__init__()
        self.number_of_batches = number_of_batches
        lr_range = np.geomspace(lr_min, lr_max, num=number_of_batches).tolist()
        self.lr_range = lr_range
        self.lr_range_iter = itertools.cycle(lr_range)
        self.lr_min = lr_min
        self.lr_max = lr_max

    def on_train_begin(self, logs={}):
        self.update_lr()
        self.losses = []

    def plot_lr_losses(self):
        fig, ax = plt.subplots(1, 1)

        ax.semilogx(self.lr_range, self.losses)

        ax.set_xlim(self.lr_min / 5.0, self.lr_max)
        ax.set_ylim(0.9 * np.min(self.losses),
                    3.0 * np.mean(self.losses[0:20]))

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Loss')

        fig.savefig('lr_find.pdf')

    def update_lr(self):
        K.set_value(self.model.optimizer.lr, next(self.lr_range_iter))

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.update_lr()
        if len(self.losses) == self.number_of_batches:
            self.plot_lr_losses()
            print('')
            print('The figure of learning rates has been plotted.')


class CyclicCosineRestart(keras.callbacks.Callback):
    def __init__(self,
                 lr_min=1e-6,
                 lr_max=1e-4,
                 number_of_lr_warm_epochs=5,
                 number_of_epochs=200,
                 use_warmup=True):
        super(CyclicCosineRestart, self).__init__()
        self.number_of_lr_warm_epochs = number_of_lr_warm_epochs
        self.number_of_epochs = number_of_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.use_warmup = use_warmup

        self.cosine_lr()
        self.warmup_lr()

    def on_train_begin(self, logs={}):
        K.set_value(self.model.optimizer.lr, 0.0)
        if self.use_warmup:
            self.lr_iter = self.warmup_lr_iter
        else:
            self.lr_iter = self.cos_lr_iter

    def warmup_lr(self):
        lr_warm = np.linspace(self.lr_min, self.lr_max,
                              self.number_of_lr_warm_epochs)
        self.warmup_lr_iter = itertools.cycle(lr_warm.tolist())

    def cosine_lr(self):
        T_current = np.linspace(0, self.number_of_epochs,
                                self.number_of_epochs)
        lr_current = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + np.cos((T_current / (self.number_of_epochs)) * np.pi))
        self.cos_lr_iter = itertools.cycle(lr_current.tolist())

    def update_lr(self):
        K.set_value(self.model.optimizer.lr, next(self.lr_iter))

    def on_epoch_begin(self, epoch, logs={}):
        if self.use_warmup:
            if (epoch + 1) > self.number_of_lr_warm_epochs:
                self.lr_iter = self.cos_lr_iter
        self.update_lr()
        print(
            f'The current learning rate is {K.get_value(self.model.optimizer.lr)}.'
        )
