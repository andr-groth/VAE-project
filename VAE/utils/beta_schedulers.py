# -*- coding: utf-8 -*-
"""Collection of beta schedulers.

Beta schedulers are used to schedule the beta value of the KL divergence during training. Beta schedulers are
used in the :class:`generators`.

"""

import abc

import numpy as np
from scipy.stats import loguniform


class BetaScheduler(abc.ABC):
    """Abstract class for beta schedulers.

    This is the abstract base class for all beta schedulers.

    Beta schedulers should implement the method :func:`__call__` and can optionally overwrite the method
    :func:`get_config`

    Parameters:
        dtype:
            Data type of the returned beta values.

    """
    def __init__(self, dtype: str = 'float'):
        self.dtype = dtype

    @abc.abstractmethod
    def __call__(self, epoch: int, shape: tuple[int, ...] = (1, )) -> np.ndarray:
        """Return beta value.

        Abstract method that has to be implemented by all beta schedulers. This method is called during training to
        obtain the beta values for the current `epoch`. The `shape` parameter defines the shape of the returned array
        of constant beta values.

        Parameters:
            epoch:
                Training epoch for with the beta values will be return.
            shape:
                Output shape for the beta values that will be returned.

        Returns:
            Array of shape `shape` filled with beta values.
        """
        pass

    def get_config(self) -> dict:
        """Get configuration.

        Returns:
            Dictionary with the configuration of the beta scheduler.

        """
        return {'dtype': self.dtype}

    def summary(self):
        """Print a summary of the beta scheduler."""
        config = self.get_config()
        cols = len(max(config.keys(), key=len))
        print(f'Summary of "{self.__class__.__name__}" (BetaScheduler)')
        for key, value in config.items():
            print(f'  {key:{cols}} : {value}')


class Constant(BetaScheduler):
    """Return constant beta value.

    This beta scheduler returns a constant beta value for all epochs.

    Parameters:
        value:
            Value of beta.
        **kwargs:
            Additional arguments for :class:`BetaScheduler`.
    """
    def __init__(self, value: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def __call__(self, epoch, shape=(1, )):
        return np.full(shape=shape, fill_value=self.value, dtype=self.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({'value': self.value})
        return config


class Linear(BetaScheduler):
    """Linearly increase beta value.

    This beta scheduler returns a linearly increasing beta value.

    Parameters:
        lower:
            Lower (left) bound of beta.
        upper:
            Upper (right) bound of beta.
        epochs:
            Number of epochs for which beta will be increased. If the number of epochs is reached, beta will be
            constant at the upper bound.
        **kwargs :
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower: float = 0., upper: float = 1., epochs: float = 10, **kwargs):
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

    This beta scheduler returns a beta value that increases to a maximum value at a given rate. The beta value follows
    a logistic growth function.

    Parameters:
        lower:
            Lower (left) asymptote of beta.
        upper:
            Upper (right) asymptote of beta.
        midpoint:
            Epoch at which beta equals the mean of the upper and lower asymptote.
        rate:
            Growth rate at which beta increases.
        **kwargs:
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower: float = 0., upper: float = 1., midpoint: float = 5., rate: float = 1., **kwargs):
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

    This beta scheduler draws beta values from a log-uniform distribution. The log-uniform distribution is a uniform
    distribution in log-space.

    Parameters:
        lower:
            Lower (minimum) value of the distribution.
        upper:
            Upper (maximum) value of the distribution.
        **kwargs:
            Additional arguments for :class:`BetaScheduler`.

    """
    def __init__(self, lower: float = 0.01, upper: float = 1., **kwargs):
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
