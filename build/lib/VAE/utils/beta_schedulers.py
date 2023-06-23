# -*- coding: utf-8 -*-
"""Collection of beta schedulers.

@author: Andreas Groth
"""

import abc

import numpy as np
from scipy.stats import loguniform


class BetaScheduler(abc.ABC):
    """Abstract class for beta schedulers."""
    def __init__(self, dtype='float'):
        """Constructor.

        Parameters:
            dtype :
                Numpy dtype of returned value.

        """
        self.dtype = dtype

    @abc.abstractmethod
    def __call__(self, epoch: int, shape: tuple[int, ...] = (1, )):
        """Return beta values.

        Parameters:
            epoch : int
                Training epoch for with the beta values will be return.
            shape : tuple of int
                Output shape for the beta values that will be returned.

        Returns:
            NumPy array of shape `shape` filled with beta values.
        """
        pass

    def get_config(self):
        """Return the configuration."""
        return {'dtype': self.dtype}

    def summary(self):
        config = self.get_config()
        cols = len(max(config.keys(), key=len))
        print(f'Summary of "{self.__class__.__name__}" (BetaScheduler)')
        for key, value in config.items():
            print(f'  {key:{cols}} : {value}')


class Constant(BetaScheduler):
    """Return constant beta value.

    Parameters:
        value : float
            Value of beta.
        kwargs :
            Additional arguments for :class:`BetaScheduler`.
    """
    def __init__(self, value=1., **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __call__(self, epoch, shape=(1, )):
        return np.full(shape=shape, fill_value=self.value, dtype=self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'value': self.value})
        return config


class Linear(BetaScheduler):
    """Linear change of beta.

    Parameters:
        lower: float
            Lower (left) bound of beta.
        upper: float
            Upper (right) bound of beta.
        epochs: int
            Number of epochs to reach the upper bound.
        kwargs :
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower=0., upper=1., epochs=10, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.epochs = epochs
        self.values = np.linspace(lower, upper, epochs)

    def __call__(self, epoch, shape=(1, )):
        if epoch < self.epochs:
            beta = self.values[epoch]
        else:
            beta = self.values[-1]
        return np.full(shape=shape, fill_value=beta, dtype=self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'lower': self.lower, 'upper': self.upper, 'epochs': self.epochs})
        return config


class LogisticGrowth(BetaScheduler):
    """Increase beta to maximum value at given rate.

    Parameters:
        lower : float
            Lower (left) asymptote of beta.
        upper : float
            Upper (right) asymptote of beta.
        midpoint : float
            Epoch at which beta equals the mean of the upper and lower asymptote.
        rate : float
            Growth rate at which beta increases.
        kwargs :
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower=0., upper=1., midpoint=5., rate=1., **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.midpoint = midpoint
        self.rate = rate

    def __call__(self, epoch, shape=(1, )):
        beta = self.lower + (self.upper - self.lower) / (1 + np.exp(-self.rate * (epoch - self.midpoint)))
        return np.full(shape=shape, fill_value=beta, dtype=self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            'lower': self.lower,
            'upper': self.upper,
            'midpoint': self.midpoint,
            'rate': self.rate,
        })
        return config


class LogUniform(BetaScheduler):
    """Draw beta values from log-uniform distribution.

    Parameters:
        lower : float
            Lower (minimum) value of the distribution.
        upper : float
            Upper (maximum) value of the distribution.
        kwargs :
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower=0.01, upper=1., **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
        self.fcn = loguniform(lower, upper)

    def __call__(self, epoch, shape=(1, )):
        beta = self.fcn.rvs(size=shape)
        return beta.astype(self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'lower': self.lower, 'upper': self.upper})
        return config


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    bs1 = LogisticGrowth(lower=0, upper=0.2, midpoint=5, rate=1)
    bs2 = LogisticGrowth(lower=10, upper=1, midpoint=5, rate=0.5)

    # bs1 = LogisticGrowth(lower=1, upper=0.1, midpoint=5, rate=1)
    # bs2 = LogisticGrowth(lower=1, upper=2, midpoint=8, rate=5)

    epochs = np.arange(0, 21)
    beta1 = np.array([bs1(epoch) for epoch in epochs])
    beta2 = np.array([bs2(epoch) for epoch in epochs])

    plt.plot(epochs, beta1, '.-')
    plt.plot(epochs, beta2, '.-')
    plt.plot(epochs, beta1 * beta2, 'o-')
    plt.grid(linestyle=':')
    plt.xticks(epochs[::2])
    plt.ylim([0, 3.35])

    bs1.summary()
