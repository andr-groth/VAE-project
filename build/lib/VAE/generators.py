# -*- coding: utf-8 -*-
"""Collection of generators for VAE model training.

- Generator class for data given as a Numpy array(s).
- Only zonal waves from spherical harmonics used for condition.

@author: Andreas Groth, Imperial College London
"""

from itertools import chain

import numpy as np
import tensorflow.keras as ks
from scipy.special import sph_harm

__version__ = '2022-06-04'

# Changelog:
# 2022-08-16 : support of sample weights
# 2022-08-16 : added two types of random ensemble sampling
# 2022-07-29 : added optional argument 'input_channels'
# 2022-07-27 : added optional argument 'filter_length'


class FitGenerator(ks.utils.Sequence):
    """Generator class for model training.

    Given an Numpy array of shape `(set_size, data_length, channels)`, the generator prepares the `inputs and `targets`
    for the model training in :func:`keras.Model.fit_generator()`.

    Parameters:
        datasets : Numpy array of list of Numpy arrays
            Numpy arrays of shape `(set_size, data_length, channels)`. In case of a list of Numpy arrays, `set_size` and
            `channels` must be the same, while `data_length` can vary. Missing (non-finite) values will be excluded from
            the samples.
        input_length : int
            Length of input to the encoder.
        batch_size : int
            Batch size. Note that the effective batch size is `batch_size * repeat_samples`.
        beta_scheduler : BetaScheduler
            Instance of :class:`BetaScheduler` that returns the `beta` parameters for the KL
            loss in each epoch.
        ensemble_size : int
            Size for the one-hot encoded ensemble condition.
        ensemble_range : tuple of two ints
            Range of the one-hot encoded ensemble condition. Must be a subrange of `(0, ensemble_size)`. Defaults to
            `None` and is set to `(0, ensemble_size)`.
        ensemble_replace : bool
            Whether to sample the one-hot encoded ensemble condition with replacement if `repeat_samples > 1`. Defaults
            to `False`.
        ensemble_sync : bool
            If `True`, the ensemble conditions of the encoder and decoder are the same. Defaults to `False`, i.e. the
            ensemble conditions of the encoder an decoder are different. Note that the ensemble conditions of the
            decoder and prediction are always the same.
        ensemble_type : str
            Whether to use the dataset index (`index`) or random ensemble condition (`random`, 'random_full'). If
            `index`, the one-hot encoded ensemble condition corresponds to the dataset index. If 'random' or
            `random_full`, the one-hot encoded ensemble condition is sampled from a uniform distribution in the range
            `ensemble_range`. The samples are the same for all samples in a batch ('random') or different for each
            sample in a batch ('random_full'). Defaults to `random`.
        filter_length : int or tuple of two ints
            Length of the temporal filter for the inputs and targets.  A centered moving average filter of length `2 *
            filter_length + 1` is applied to the inputs and targets. If a tuple of two ints is given, the first int is
            the length of the filter for the input to the encoder and the target to the decoder. The second int is the
            length of the filter for the target to the prediction. Defaults to `None`, i.e. no filter.
        initial_epoch : int
            Initial epoch at which the generator will start. This will affect the `beta` parameter.
        input_channels : tuple of ints
            Range of channels used as input. The items in the tuple refer to start, stop and step in slice notation.
            Defaults to `None` means all channels are used.
        latitude : float
            Latitude in degree if the spherical harmonics are used for spatial condition.
        longitude : array_like
            Longitude of the data in degree if the spherical harmonics are used for spatial condition. Length of
            `longitude` must be equal to `set_size`. Defaults to `None` and is set to `np.arange(0, 360, 360/set_size)`.
        prediction_channels : tuple of ints
            Range of channels used for prediction. The items in the tuple refer to start, stop and step in slice
            notation. Defaults to `None` means all channels are used.
        prediction_length : int
            Length of prediction. Defaults to `None` means no prediction.
        repeat_samples : int
            Number of times the same sample is repeated in the batch. This will augment the batch size. This options is
            useful in combination with the ensemble condition, in which the same samples is presented multiple times
            with different random samples of the ensemble conditions. Defaults to 1.
        sample_weights : array_like
            Sample weights of shape `(nr_datasets, set_size)`. Defaults to `None`.
        sph_degree : int
            Number of spherical degrees if the spherical harmonics are used for spatial condition.
        strides : int
            Sample strides along second dimension of `data` of size `data_length`.
        time : array_like
            Time of the data if the time-periodic harmonics are used for temporal condition.
        tp_period : float
            Maximal period for the temporal harmonics. See :func:`get_tp_harmonics`.
        dtype :
            Dtype of the data that will be returned.
        shuffle : bool
            Shuffle samples order.
    """
    def __init__(self,
                 datasets,
                 input_length,
                 batch_size=32,
                 beta_scheduler=None,
                 ensemble_size=None,
                 ensemble_range=None,
                 ensemble_replace=False,
                 ensemble_sync=False,
                 ensemble_type='random',
                 filter_length=None,
                 initial_epoch=0,
                 input_channels=None,
                 latitude=0,
                 longitude=None,
                 prediction_channels=None,
                 prediction_length=None,
                 repeat_samples=1,
                 sample_weights=None,
                 shuffle=True,
                 sph_degree=None,
                 strides=1,
                 time=None,
                 tp_period=None,
                 dtype='float32',
                 **kwargs):
        """Instantiate generator."""
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]

        shapes = {dataset[:, 0, :].shape for dataset in datasets}
        if len(shapes) > 1:
            raise ValueError('all datasets must have the same set_size and number of channels')

        set_size, channels = shapes.pop()
        self.channels = channels
        self.set_size = set_size

        self.batch_size = batch_size
        self.beta_scheduler = beta_scheduler
        self.datasets = datasets
        self.dtype = dtype
        self.ensemble_replace = ensemble_replace
        self.ensemble_size = ensemble_size
        self.ensemble_sync = ensemble_sync
        self.epoch = initial_epoch
        self.filter_length = tuple(np.broadcast_to(filter_length, (2, )).tolist())
        self.input_length = input_length
        self.repeat_samples = repeat_samples
        self.shuffle = shuffle
        self.strides = strides
        self.prediction_length = prediction_length if prediction_length is not None else 0

        self.input_channels = slice(*input_channels) if input_channels is not None else slice(None)
        self.prediction_channels = slice(*prediction_channels) if prediction_channels is not None else slice(None)

        if ensemble_size is not None:
            if ensemble_range is None:
                ensemble_range = (0, ensemble_size)

            if repeat_samples > len(range(*ensemble_range)) and not ensemble_replace:
                raise ValueError(
                    '`repeat_samples` must not be larger than `ensemble_range` if sampling without replacement')

            if ensemble_type not in ['index', 'random', 'random_full']:
                raise ValueError('`ensemble_type` must be either `index`, `random` or `random_full`')

            if ensemble_type == 'index' and len(datasets) > ensemble_size:
                raise ValueError(
                    '`ensemble_type` is `index` but `ensemble_size` is smaller than the number of datasets')

        self.ensemble_type = ensemble_type
        self.ensemble_range = ensemble_range

        self._prepare_embedding()
        if self.shuffle:
            self.shuffle_data()

        if sph_degree is not None:
            if longitude is None:
                longitude = np.arange(0, 360, 360 / set_size)

            self.sph_harmonics = self.get_sph_harmonics(latitude, longitude, sph_degree)
        else:
            self.sph_harmonics = None

        if tp_period is not None:
            if time is None:
                raise ValueError('time must be given if tp_period is given')

            self.tp_harmonics = self.get_tp_harmonics(time, tp_period)
        else:
            self.tp_harmonics = None

        if sample_weights is not None:
            assert len(sample_weights) == len(datasets), 'sample_weights must have the same length as datasets'
        self.sample_weights = sample_weights

    def __len__(self):
        """Return length of generator."""
        return -(-self.nr_samples // self.batch_size)

    def __getitem__(self, idx):
        """Return batch of data.

        Note that the effective batch size is `batch_size * repeat_samples`.

        Parameters:
            idx : int
                Batch index.

        Returns:
            tuple of two dictionaries:
                The first dictionary contains the inputs and the second dictionary the targets for the model training.
        """

        if idx >= self.__len__():
            raise IndexError('batch index out of range')

        inputs = dict()
        targets = dict()

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, ..., self.input_channels]
        batch_size = len(batch_x)
        batch_x = self._repeat(batch_x)
        inputs['encoder_input'] = batch_x
        targets['decoder'] = batch_x

        encoder_cond = []
        decoder_cond = []

        if self.sph_harmonics is not None:
            sph_cond = self.sph_harmonics[None, :, :]
            sph_cond = np.repeat(sph_cond, batch_size * self.repeat_samples, axis=0)
            encoder_cond.append(sph_cond)
            decoder_cond.append(sph_cond)

        if self.tp_harmonics is not None:
            tp_cond = self.tp_harmonics[self.get_index(idx)[:, 1], None, :]
            tp_cond = np.repeat(tp_cond, self.set_size, axis=1)
            encoder_cond.append(tp_cond)
            decoder_cond.append(tp_cond)

        if self.ensemble_size is not None:
            if self.ensemble_sync:
                ens_cond = self.get_ensemble_condition(batch_size, idx)
                encoder_cond.append(ens_cond)
                decoder_cond.append(ens_cond)
            else:
                encoder_cond.append(self.get_ensemble_condition(batch_size, idx))
                decoder_cond.append(self.get_ensemble_condition(batch_size, idx))

        if encoder_cond:
            inputs['encoder_cond'] = np.concatenate(encoder_cond, axis=-1)
            inputs['decoder_cond'] = np.concatenate(decoder_cond, axis=-1)

        if self.y is not None:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, ..., self.prediction_channels]
            batch_y = self._repeat(batch_y)
            targets['prediction'] = batch_y
            if 'decoder_cond' in inputs:
                inputs['prediction_cond'] = inputs['decoder_cond']

        if self.beta_scheduler is not None:
            inputs['beta'] = self.beta_scheduler(self.epoch, shape=(batch_size * self.repeat_samples, 1))

        if self.sample_weights is not None:
            indices = self.get_index(idx)[:, 0]
            sw = [self.sample_weights[i] for i in indices]
            sw = np.stack(sw, axis=0)
            samples_weights = {'decoder': sw, 'prediction': sw}
            return inputs, targets, samples_weights

        else:
            return inputs, targets

    def _filter(self, x, length):
        """Filter batch of inputs."""
        if length is not None and length > 0:
            xp = np.pad(x, pad_width=((0, 0), (0, 0), (length + 1, length), (0, 0)))
            xc = np.cumsum(xp, axis=-2)
            w = 2 * length + 1
            xd = xc[..., w:, :] - xc[..., :-w, :]

            # denominator
            d = np.ones(x.shape[-2])
            dp = np.pad(d, pad_width=(length + 1, length))
            dc = np.cumsum(dp)
            dd = dc[w:] - dc[:-w]

            return xd / dd[:, None]
        else:
            return x

    def _repeat(self, x):
        """Repeat samples."""
        if self.repeat_samples > 1:
            x = np.repeat(x, self.repeat_samples, axis=0)
        return x

    def _prepare_embedding(self):
        """Prepare the data embedding."""
        x = []
        y = []
        index = []
        nx, ny = self.filter_length

        for n, dataset in enumerate(self.datasets):
            _, data_length, _ = dataset.shape
            for sample_start in range(self.input_length, data_length - self.prediction_length, self.strides):
                # pick a new sample
                sample = dataset[:, sample_start - self.input_length:sample_start + self.prediction_length, :]

                # split into input and target
                if np.all(np.isfinite(sample)):
                    x.append(sample[:, :self.input_length, :])
                    index.append((n, sample_start))
                    if self.prediction_length:
                        y.append(sample[:, self.input_length:, :])

        self.x = self._filter(np.stack(x), nx).astype(self.dtype)
        self.index = np.stack(index)
        if y:
            self.y = self._filter(np.stack(y), ny).astype(self.dtype)
        else:
            self.y = None

    @property
    def nr_samples(self):
        """Return number of samples."""
        return len(self.x)

    def get_ensemble_condition(self, batch_size, idx=None):
        """Return ensemble condition for given batch.

        A random one-hot encoded ensemble index is return that is the same for all samples in the batch. The code is
        broadcasted along the second dimension of size `set_size`.

        In case of `repeat_samples > 1`, the actual batch size is `batch_size * repeat_samples` and a set of
        `repeat_samples`random indices is sampled.

        To alter between sampling with and without replacement, the `ensemble_replace` flag can be set.

        Parameters:
            batch_size : int
                Batch size.

        Returns:
            3D Numpy array of shape `(batch_size * repeat_samples, set_size, ensemble_size)`
        """
        if self.ensemble_type == 'random':
            ensemble_idx = np.random.choice(np.arange(*self.ensemble_range),
                                            size=self.repeat_samples,
                                            replace=self.ensemble_replace)
            condition = np.zeros((self.repeat_samples, self.set_size, self.ensemble_size), dtype=self.dtype)
            condition[np.arange(self.repeat_samples), :, ensemble_idx] = 1
            condition = np.tile(condition, (batch_size, 1, 1))

        elif self.ensemble_type == 'random_full':
            ensemble_idx = [
                np.random.choice(np.arange(*self.ensemble_range),
                                 size=self.repeat_samples,
                                 replace=self.ensemble_replace) for _ in range(batch_size)
            ]
            ensemble_idx = np.stack(ensemble_idx, axis=0)
            condition = np.zeros((batch_size * self.repeat_samples, self.set_size, self.ensemble_size),
                                 dtype=self.dtype)
            condition[np.arange(batch_size * self.repeat_samples), :, ensemble_idx.flat] = 1

        else:
            if idx >= self.__len__():
                raise IndexError('batch index out of range')
            if idx is None:
                raise ValueError('idx must be given')

            ensemble_idx = self.get_index(idx)[:, 0]
            condition = np.zeros((len(ensemble_idx), self.set_size, self.ensemble_size), dtype=self.dtype)
            condition[np.arange(len(ensemble_idx)), :, ensemble_idx] = 1
        return condition

    def get_index(self, idx):
        """Return array of dataset and time index of samples in batch.

        The returned array is of shape `(batch_size * repeat_samples, 2)` with the first column containing the dataset
        index and the second column the time index of the sample. The time index refers to the first sample of the
        target sequence for the prediction.

        Parameters:
            idx : int
                Batch index.

        Returns:
            Numpy array of shape `(batch_size * repeat_samples, 2)`.
        """
        if idx >= self.__len__():
            raise IndexError('batch index out of range')

        indices = self.index[idx * self.batch_size:(idx + 1) * self.batch_size, ...]
        indices = self._repeat(indices)
        return indices

    def get_sph_harmonics(self, latitude, longitude, sph_degree):
        """Get spherical harmonics.

        The returned array is of shape `(set_size, 2 * sph_degree + 1)` with the rows containing the spherical harmonics
        for the given latitude and longitude values.

        Parameters:
            latitude : float
                Latitude value in degree.
            longitude : array_like
                Array of longitude values in degree of shape `(set_size,)`.

        Returns:
            Numpy array of shape `(set_size, 2 * sph_degree + 1)`.
        """
        colat = np.pi / 2 - np.deg2rad(latitude)  # [0, pi]
        lon = np.deg2rad(longitude) % (2 * np.pi)  # [0, 2*pi]

        sph = []
        for n in range(sph_degree + 1):
            s = sph_harm(n, n, lon, colat)
            sph.append(np.real(s))
            if n > 0:
                s = sph_harm(-n, n, lon, colat)
                sph.append(np.imag(s))

        return np.stack(sph, axis=1).astype(self.dtype)

    def get_tp_harmonics(self, time, tp_period):
        """Get temporal harmonics.

        Parameters:
            time : array_like
                Array of time values for which the harmonics are calculated.
            tp_period : int
                Maximal period for the temporal harmonics in time units.

        Returns:
            2D Numy array of shape `(len(times), tp_period)`.
        """
        f = np.fft.rfftfreq(tp_period)[None, 1:]  # omit DC
        time = np.array(time)[:, None]
        harmonics = np.concatenate([np.sin(2 * np.pi * f * time), np.cos(2 * np.pi * f * time)], axis=1)

        return harmonics

    def on_epoch_end(self):
        """On epoch end.

        This method is called at the end of every epoch and is used to increment the internal epoch counter and shuffle
        the data.
        """
        self.epoch += 1
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        """Shuffle data."""
        idx = np.random.permutation(len(self.x))
        self.x = self.x[idx, ...]
        self.index = self.index[idx, ...]

        if self.y is not None:
            self.y = self.y[idx, ...]

    def summary(self):
        """Print summary."""
        total_length = sum([dataset.shape[1] for dataset in self.datasets])
        act_batch_size = self.batch_size * self.repeat_samples
        print(f'Number of datasets : {len(self.datasets):,}')
        print(f'Total data length  : {total_length:,}')
        print(f'Strides            : {self.strides:,}')
        print(f'Number of samples  : {self.nr_samples:,}')
        print(f'Batch size         : {self.batch_size:,}')
        print(f'Number of batches  : {len(self):,}')

        if self.repeat_samples > 1:
            print(f'Sample repetitions : {self.repeat_samples:,}')
            print(f'Actual batch size  : {self.batch_size:,} * {self.repeat_samples} = {act_batch_size:,}')

        print(f'Shuffle            : {self.shuffle}')
        print(f'Filter length      : {self.filter_length:}')

        if self.ensemble_size is not None:
            print('Ensemble condition')
            print(f'  size             : {self.ensemble_size}')
            print(f'  type             : {self.ensemble_type}')
            if self.ensemble_type in ['random', 'random_full']:
                print(f'  range            : {self.ensemble_range}')
                print(f'  sync             : {self.ensemble_sync}')
                print(f'  replace          : {self.ensemble_replace}')

        # unpack items
        items = self.__getitem__(0)
        inputs, targets, sample_weights, *_ = chain(items, [None] * 3)

        print('Inputs (key: shape)')
        for key, value in inputs.items():
            print(f'  {key:<16.16} : {value.shape}')

        channels = tuple(range(self.channels))[self.input_channels]
        if len(channels) == self.channels:
            print(f'Input channels     : all')
        else:
            print(f'Input channels     : {channels}')

        if targets is not None:
            print('Targets (key: shape)')
            for key, value in targets.items():
                print(f'  {key:<16.16} : {value.shape}')

            if self.prediction_length:
                channels = tuple(range(self.channels))[self.prediction_channels]
                if len(channels) == self.channels:
                    print(f'Predicted channels : all')
                else:
                    print(f'Predicted channels : {channels}')

        if sample_weights is not None:
            print('Sample weight shapes')
            for key, value in sample_weights.items():
                print(f'  {key:<16.16} : {value.shape}')


class PredictGenerator(FitGenerator):
    """Generator class for model prediction.

    The generator prepares the inputs for the model prediction with :func:`ks.Model.predict`.

    Parameters : multiple
        See :class:`FitGenerator`.

    Returns:
        dict containing the inputs for the model prediction

    See also:
        :class:`FitGenerator`: Generator class for model training.
    """
    def __getitem__(self, idx) -> dict:
        """Return inputs to the model for given batch."""
        inputs, _ = super().__getitem__(idx)
        return inputs


def _example_fit_generator():
    import matplotlib.pyplot as plt
    shape = (64, 32, 12)
    datasets = np.split(np.arange(np.prod(shape)).reshape(shape), shape[-1] // 3, axis=-1)
    fit_gen = FitGenerator(
        datasets,
        input_length=10,
        batch_size=64,
        ensemble_size=len(datasets),
        ensemble_type='index',
        filter_length=(0, 1),
        input_channels=(1, None),
        repeat_samples=1,
        strides=1,
        prediction_length=4,
        prediction_channels=(0, 1),
        tp_period=16,
        sph_degree=None,
        time=range(shape[1]),
        sample_weights=[[i] for i in range(len(datasets))],
        shuffle=False,
    )
    fit_gen.summary()
    inputs, targets, sample_weights = fit_gen[0]
    plt.figure()
    plt.pcolormesh(inputs['encoder_cond'][:, 0, :])
    # plt.figure()
    # plt.pcolormesh(inputs['encoder_cond'][0, :, :])
    # plt.figure()
    # plt.pcolormesh(fit_gen.sph_harmonics @ fit_gen.sph_harmonics.T, vmin=-1, vmax=1, cmap='bwr')

    plt.figure()
    plt.pcolormesh(sample_weights['decoder'])

    return fit_gen


if __name__ == '__main__':
    fit_gen = _example_fit_generator()
