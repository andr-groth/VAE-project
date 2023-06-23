# -*- coding: utf-8 -*-
"""Learning rate utilities.

"""

import numpy as np
import tensorflow.keras as ks


class LearningRateFinder:
    """Learning rate finder.

    See also:
        https://github.com/surmenok/keras_lr_finder
    """
    def __init__(self, model, beta=0.98, stop_factor=4, filename=None, update_freq=10):

        self.model = model
        self.best_loss = 1e9
        self.beta = beta
        self.stop_factor = stop_factor
        self.ave_loss = None
        self.filename = filename
        self.update_freq = update_freq
        self.log = []

    def on_batch_end(self, batch, logs):
        """Get loss and increase learning rate"""

        # Get the learning rate and loss
        lr = ks.backend.get_value(self.model.optimizer.lr)
        loss = logs['loss']

        # Exponentially smoothed loss
        if self.ave_loss is None:
            self.ave_loss = loss
        else:
            self.ave_loss = (self.beta * self.ave_loss) + ((1 - self.beta) * loss)

        self.log.append('{:5d} {:10.8f} {:10.8f} {:10.8f}\n'.format(batch, lr, loss, self.ave_loss))

        # Save log to file
        if (self.filename is not None) and (batch % self.update_freq == 0):
            file = open(self.filename, 'a')
            file.writelines(self.log)
            file.close()
            self.log = []

        # Check whether the loss gets large again
        decaytime = 1 / (1 - self.beta)
        if (batch > decaytime) and (self.ave_loss > self.best_loss * self.stop_factor):
            self.model.stop_training = True
            return

        if (batch > decaytime) and (self.ave_loss < self.best_loss):
            self.best_loss = self.ave_loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        ks.backend.set_value(self.model.optimizer.lr, lr)

    def on_train_end(self, logs):
        """Final save log file"""
        if (self.filename is not None):
            file = open(self.filename, 'a')
            file.writelines(self.log)
            file.close()

    def find_generator(self, generator, start_lr=1e-4, end_lr=1., callbacks=[], **kw_fit):
        """Start model training with increasing learning rate."""

        epochs = kw_fit['epochs']
        num_batches = epochs * len(generator)
        self.lr_mult = (end_lr / start_lr)**(1 / num_batches)

        # Set the initial learning rate
        ks.backend.set_value(self.model.optimizer.lr, start_lr)

        callback = ks.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs),
                                               on_train_end=lambda logs: self.on_train_end(logs))

        # merge callbacks
        callbacks = callbacks + [callback]

        self.model.fit_generator(generator=generator, callbacks=callbacks, **kw_fit)


class StepDecay:
    """Learning rate scheduler with exponential decay.

    Learning rate scheduler for use in :class:`keras.callbacks.LearningRateScheduler`.

    Parameters::
        lr:
            Float, specifying the initial learning rate.
        decay_rate:
            Float, specifying the decay rate.
        steps:
            Integer, specifying the number of epochs at which the learning rate is reduced by
            ``decay_rate``.

    Returns:
        Float:
            Learning rate at given epoch.
    """
    def __init__(self, lr, decay_rate, steps):

        self.lr = lr
        self.decay_rate = decay_rate
        self.steps = steps

    def __call__(self, epoch):
        return self.lr * self.decay_rate**(epoch / self.steps)


class ExponentialDecay:
    """A learning rate scheduler that uses an exponential decay schedule.

    Parameters:
        initial_learning_rate : float
            Initial learning rate.
        decay_steps : int
            Number of steps after which the learning rate decays by the decay rate specified in `decay_rate`.
        decay_rate : float
            Decay rate at which the learning rate decays.
        warmup_steps : int
            Number of steps for warmup in which the learning rate increases.
        staircase : bool
            If `True`, the learning rate decays at discrete intervals.

    Returns:
        float

    """
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, warmup_steps=0, staircase=False):
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.staircase = staircase

    def __call__(self, epoch, *args):
        # catch additional unused arguments (TF1: epoch. TF2 epoch and lr)
        if epoch < self.warmup_steps:
            warmup_ratio = (epoch + 1) / (self.warmup_steps + 1)
            lr = self.initial_learning_rate * warmup_ratio
        else:
            decay_ratio = (epoch - self.warmup_steps) / self.decay_steps
            if self.staircase:
                decay_ratio = np.floor(decay_ratio)
            lr = self.initial_learning_rate * self.decay_rate**decay_ratio

        return float(lr)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    lr_schedule = ExponentialDecay(initial_learning_rate=0.001,
                                   decay_steps=10,
                                   decay_rate=0.5,
                                   warmup_steps=10,
                                   staircase=False)

    lr = [lr_schedule(epoch) for epoch in range(50)]
    plt.plot(lr, '.-')
    plt.grid()
