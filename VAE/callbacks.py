# -*- coding: utf-8 -*-
"""
Collection of callbacks for model training.

"""

from tensorflow.keras import callbacks


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Callback to save the Keras model or model weights at some frequency.

    Prevents deprecation warning in super class when period is used.

    Parameters:
        period:
            Number of epochs between checkpoints.
        **kwargs:
            Additional keyword arguments to be passed to `keras.callbacks.ModelCheckpoint`.
    """
    def __init__(self, filepath: str, period: int = 1, **kwargs):
        super().__init__(filepath, **kwargs)
        self.period = period


class Evaluate(callbacks.Callback):
    """Callback to evaluate the model.

    This callbacks evaluates the model on the given data at the end of each epoch. For a detailed description of
    the input and target data see the Keras documentation for :func:`keras.Model.evaluate`.

    With this callback, further validation datasets can be evaluated during training. The prefix is used to identify
    the metrics in the logs and should be unique for each validation dataset. The prefix cannot be empty of `val_`.
    This would cause training metrics to be overwritten.

    Parameters:
        x:
            Input data.
        y:
            Target data.
        batch_size:
            Number of samples per batch.
        verbose:
            Verbosity mode.
        sample_weight:
            Sample weights.
        prefix:
            Prefix for the metric names. Defaults to 'val2_'.
        **kwargs :
            Additional keyword arguments to be passed to `keras.callbacks.Callback`.

    raises:
        ValueError: If the prefix is empty or `val_`.

    """
    def __init__(self,
                 x=None,
                 y=None,
                 batch_size: int = None,
                 verbose: int = 0,
                 sample_weight=None,
                 prefix: str = 'val2_',
                 **kwargs):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        self.sample_weight = sample_weight
        self.prefix = prefix
        if prefix in {None, '', 'val_'}:
            raise ValueError('prefix cannot be empty or "val_". This will cause the metrics to be overwritten.')

        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = {} if logs is None else logs
        scores = self.model.evaluate(self.x,
                                     self.y,
                                     batch_size=self.batch_size,
                                     verbose=self.verbose,
                                     sample_weight=self.sample_weight)

        scores = {self.prefix + name: score for name, score in zip(self.model.metrics_names, scores)}
        logs.update(scores)
