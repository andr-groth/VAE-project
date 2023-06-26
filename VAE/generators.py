# -*- coding: utf-8 -*-
"""Collection of generators for VAE model training.

Generator class for data given as a Numpy array(s).

"""

from itertools import chain
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as ks
from scipy.special import sph_harm

__version__ = '2022-06-04'


class FitGenerator(ks.utils.Sequence):
    """Generator class for model training.

    Given an Numpy array of shape `(set_size, data_length, channels)`, the generator prepares the `inputs and `targets`
    for the model training in :func:`keras.Model.fit_generator()`.

    Parameters:
        datasets:
            Dataset used for training. The dataset can be either a single mumpy array of shape `(set_size, data_length,
            channels)` or a list of Numpy arrays. In case of a list of Numpy arrays, `set_size` and `channels` must be
            the same, while `data_length` can vary. Missing (non-finite) values will be excluded from the samples.
        input_length:
            Length of input to the encoder.
        batch_size:
            Batch size. Note that the effective batch size is `batch_size * repeat_samples`.
        beta_scheduler:
            Instance of :class:`BetaScheduler` that returns the `beta` parameters for the KL loss in each epoch.
        condition:
            Additional data used as condition. The Numpy arrays must be of length `data_length` matching the Numpy
            array(s)  in `dataset`. If a list is provided, the length of the list must match the length of `datasets`.
            If a dict is provided, the keys must match `encoder` and `decoder`. This allows to pass different conditions
            to the encoder and decoder, provided as the corresponding dict values.
        ensemble_size:
            Size for the one-hot encoded ensemble condition.
        ensemble_type:
            Whether to use the dataset index (`index`) or random ensemble condition (`random`, 'random_full'). If
            `index`, the ensemble condition corresponds to the dataset index. If 'random' or `random_full`, the ensemble
            condition is sampled from a uniform distribution in the range `ensemble_range`. The samples are the same for
            all samples in a batch ('random') or different for each sample in a batch ('random_full'). Defaults to
            `random`.
        ensemble_index:
            Array of indices used as ensemble condition if `ensemble_type` is `index`. Must match the length of
            `dataset` and must be in range `(0, ensemble_size)`. Defaults to `None` meaning the dataset index is used.
        ensemble_range:
            Range of the random ensemble condition. Must be a subrange of `(0, ensemble_size)`. Defaults to `None` and
            is set to `(0, ensemble_size)`.
        ensemble_replace:
            Whether to sample the random ensemble condition with replacement if `repeat_samples > 1`. Defaults to
            `False`.
        ensemble_sync:
            Sychronization random ensemble conditions between encoder and decoder. If `True` , the random ensemble
            conditions of the encoder and decoder are the same. Defaults to `False`, i.e. the random ensemble conditions
            of the encoder an decoder are different random samples. Note that the ensemble conditions of the decoder and
            prediction are always the same.
        filter_length:
            Length of the temporal filter for the inputs and targets.  A centered moving average filter of length `2 *
            filter_length + 1` is applied to the inputs and targets. If a tuple of two ints is given, the first int is
            the length of the filter for the input to the encoder and the target to the decoder. The second int is the
            length of the filter for the target to the prediction. Defaults to `None`, i.e. no filter.
        initial_epoch:
            Initial epoch at which the generator will start. This will affect the `beta` parameter.
        input_channels:
            Range of channels used as input. The items in the tuple refer to start, stop and step in slice notation.
            Defaults to `None` means all channels are used.
        latitude:
            Latitude in degree if the spherical harmonics are used for spatial condition.
        longitude:
            Longitude of the data in degree if the spherical harmonics are used for spatial condition. Length of
            `longitude` must be equal to `set_size`. Defaults to `None` and is set to `np.arange(0, 360, 360/set_size)`.
        prediction_channels:
            Range of channels used for prediction. The items in the tuple refer to start, stop and step in slice
            notation. Defaults to `None` means all channels are used.
        prediction_length:
            Length of prediction. Defaults to `None` means no prediction.
        repeat_samples:
            Number of times the same sample is repeated in the batch. This will augment the batch size. This options is
            useful in combination with the ensemble condition, in which the same samples is presented multiple times
            with different random samples of the ensemble conditions. Defaults to 1.
        sample_weights:
            Sample weights of shape `(nr_datasets, set_size)`. Defaults to `None`.
        shuffle:
            Shuffle samples order.
        sph_degree:
            Number of spherical degrees if the spherical harmonics are used for spatial condition.
        strides:
            Sample strides along second dimension of `data` of size `data_length`.
        time:
            Time of the data if the time-periodic harmonics are used for temporal condition. Must be of length
            `data_length`. If a list is provided, the length of the list must match the length of `datasets`.
        tp_period:
            Maximal period for the temporal harmonics. See :func:`get_tp_harmonics`.
        dtype:
            Dtype of the data that will be returned.
    """
    def __init__(self,
                 datasets: Union[np.ndarray, list[np.ndarray]],
                 input_length: int,
                 batch_size: int = 32,
                 beta_scheduler=None,
                 condition: Union[np.ndarray, list[np.ndarray], dict] = None,
                 ensemble_size: int = None,
                 ensemble_type: str = 'random',
                 ensemble_index: int = None,
                 ensemble_range: tuple[int, int] = None,
                 ensemble_replace: bool = False,
                 ensemble_sync: bool = False,
                 filter_length: Union[int, tuple[int, int]] = None,
                 initial_epoch: int = 0,
                 input_channels: list[int] = None,
                 latitude: np.ndarray = 0,
                 longitude: np.ndarray = None,
                 prediction_channels: list[int] = None,
                 prediction_length: int = None,
                 repeat_samples: int = 1,
                 sample_weights: np.ndarray = None,
                 shuffle: bool = True,
                 sph_degree: int = None,
                 strides: int = 1,
                 time: Union[np.ndarray, list[np.ndarray]] = None,
                 tp_period: float = None,
                 dtype: str = 'float32',
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
        self.filter_length = np.broadcast_to(filter_length, (2, ))
        self.input_length = input_length
        self.repeat_samples = repeat_samples
        self.shuffle = shuffle
        self.strides = strides
        self.prediction_length = prediction_length if prediction_length is not None else 0

        self.input_channels = slice(*input_channels) if input_channels is not None else slice(None)
        self.prediction_channels = slice(*prediction_channels) if prediction_channels is not None else slice(None)

        if condition is not None:
            if not isinstance(condition, dict):
                # same condition for encoder and decoder
                condition = {'encoder': condition}
            else:
                if 'encoder' not in condition.keys():
                    raise KeyError('Require at least `encoder` item in `condition`.')

            # prepare condition
            for key, value in condition.items():
                if isinstance(value, (list, tuple)):
                    condition.update({key: [self._prepare_condition(v) for v in value]})
                else:
                    condition.update({key: self._prepare_condition(value)})

        self.condition = condition

        if ensemble_size is not None:
            if ensemble_range is None:
                ensemble_range = (0, ensemble_size)

            if repeat_samples > len(range(*ensemble_range)) and not ensemble_replace:
                error_msg = f'{repeat_samples=} must not be larger than {ensemble_range=}' \
                    f' if sampling without replacement ({ensemble_replace=})'
                raise ValueError(error_msg)

            val_ensemble_type = {'index', 'random', 'random_full'}
            if ensemble_type not in val_ensemble_type:
                raise ValueError(f'{ensemble_type=} must be in {val_ensemble_type}')

        self.ensemble_index = np.array(ensemble_index) if ensemble_index is not None else None
        self.ensemble_type = ensemble_type
        self.ensemble_range = ensemble_range

        self._prepare_embedding()
        if self.shuffle:
            self._shuffle_data()

        if tp_period is not None:
            if time is None:
                raise ValueError('time must be given if tp_period is given')

            if isinstance(time, (list, tuple)):
                if len(time) == len(datasets):
                    self.tp_harmonics = [self.get_tp_harmonics(t, tp_period) for t in time]
                else:
                    raise ValueError('Length of `time` must match length of `datasets`')
            else:
                self.tp_harmonics = self.get_tp_harmonics(time, tp_period)

        else:
            self.tp_harmonics = None

        if sph_degree is not None:
            if longitude is None:
                longitude = np.arange(0, 360, 360 / set_size)

            self.sph_harmonics = self.get_sph_harmonics(latitude, longitude, sph_degree)
        else:
            self.sph_harmonics = None

        if sample_weights is not None:
            if len(sample_weights) != len(datasets):
                raise ValueError('`sample_weights` must have the same length as `datasets`')

        self.sample_weights = sample_weights

    def __len__(self):
        return -(-self.nr_samples // self.batch_size)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        """Return batch of data.

        Note that the effective batch size is `batch_size * repeat_samples`.

        Parameters:
            idx:
                Batch index.

        Returns:
            Two dicts, one for the inputs and one for the targets.
        """

        if idx >= self.__len__():
            raise IndexError('batch index out of range')

        inputs = dict()
        targets = dict()

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, ..., self.input_channels]
        batch_size = len(batch_x)
        batch_x = self._repeat_samples(batch_x)
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
            tp_cond = self._get_condition(self.tp_harmonics, idx)
            encoder_cond.append(tp_cond)
            decoder_cond.append(tp_cond)

        if self.condition is not None:
            ex_cond = self._get_condition(self.condition['encoder'], idx)
            if 'decoder' in self.condition.keys():
                dx_cond = self._get_condition(self.condition['decoder'], idx)
            else:
                dx_cond = ex_cond
            encoder_cond.append(ex_cond)
            decoder_cond.append(dx_cond)

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
            batch_y = self._repeat_samples(batch_y)
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

    def _filter(self, x: np.ndarray, length: int) -> np.ndarray:
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

    def _get_condition(self, condition: Union[np.ndarray, list[np.ndarray]], idx: int) -> np.ndarray:
        if isinstance(condition, list):
            output = np.stack([condition[ni][ti, :] for ni, ti in self.get_index(idx)], axis=0)
        else:
            output = condition[self.get_index(idx)[:, 1], :]
        output = np.repeat(output[:, None, :], self.set_size, axis=1)
        return output

    def _prepare_condition(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim > 2:
            raise ('`condition` must be either 1D or 2D Numpy array.')
        return x

    def _prepare_embedding(self):
        x = []
        y = []
        index = []
        nx, ny = self.filter_length

        for n, dataset in enumerate(self.datasets):
            _, data_length, _ = dataset.shape
            for sample_start in range(self.input_length, data_length - self.prediction_length + 1, self.strides):
                # pick a new sample
                sample = dataset[:, sample_start - self.input_length:sample_start + self.prediction_length, :]

                # split into input and prediction target
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

    def _repeat_samples(self, x: np.ndarray) -> np.ndarray:
        if self.repeat_samples > 1:
            x = np.repeat(x, self.repeat_samples, axis=0)
        return x

    @property
    def nr_samples(self):
        """Return number of samples."""
        return len(self.x)

    def get_ensemble_condition(self, batch_size: int, idx: int = None) -> np.ndarray:
        """Return ensemble condition for given batch.

        A one-hot encoded ensemble index is return that is the same for all samples in the batch. The code is
        broadcasted along the second dimension of size `set_size`.

        In case of `repeat_samples > 1`, the actual batch size is `batch_size * repeat_samples` and a set of
        `repeat_samples`random indices is sampled.

        To alter between sampling with and without replacement, the `ensemble_replace` flag can be set.

        Parameters:
            batch_size:
                Batch size.
            idx:
                Required if `ensemble_type='index`. Returns the ensemble condition corresponding to the batch with index
                `idx`.

        Returns:
            Array of shape `(batch_size * repeat_samples, set_size, ensemble_size)`
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

        else:  # `index`
            if idx >= self.__len__():
                raise IndexError('batch index out of range')
            if idx is None:
                raise ValueError('idx must be given')

            ensemble_idx = self.get_index(idx)[:, 0]
            if self.ensemble_index is not None:
                ensemble_idx = self.ensemble_index[ensemble_idx]

            condition = np.zeros((len(ensemble_idx), self.set_size, self.ensemble_size), dtype=self.dtype)
            condition[np.arange(len(ensemble_idx)), :, ensemble_idx] = 1
        return condition

    def get_index(self, idx: int) -> np.ndarray:
        """Return array of dataset and time index of samples in batch.

        The returned array is of shape `(batch_size * repeat_samples, 2)` with the first column containing the dataset
        index and the second column the time index of the sample. The time index refers to the first sample of the
        target sequence for the prediction.

        Parameters:
            idx:
                Batch index.

        Returns:
            Array of shape `(batch_size * repeat_samples, 2)`.
        """
        if idx >= self.__len__():
            raise IndexError('batch index out of range')

        indices = self.index[idx * self.batch_size:(idx + 1) * self.batch_size, ...]
        indices = self._repeat_samples(indices)
        return indices

    def get_sph_harmonics(self, latitude: np.ndarray, longitude: np.ndarray, sph_degree: int) -> np.ndarray:
        """Get spherical harmonics.

        The returned array is of shape `(set_size, 2 * sph_degree + 1)` with the rows containing the spherical harmonics
        for the given latitude and longitude values.

        Parameters:
            latitude : float
                Latitude value in degree.
            longitude : array_like
                Array of longitude values in degree of shape `(set_size,)`.

        Returns:
            Array of shape `(set_size, 2 * sph_degree + 1)`.
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

    def get_tp_harmonics(self, time: np.ndarray, tp_period: int) -> np.ndarray:
        """Get temporal harmonics.

        Parameters:
            time:
                Array of time values for which the harmonics are calculated.
            tp_period:
                Maximal period for the temporal harmonics in time units.

        Returns:
            Array of shape `(len(times), tp_period)`.
        """
        f = np.fft.rfftfreq(tp_period)[None, 1:]  # omit DC
        time = np.array(time)[:, None]
        harmonics = np.concatenate([np.sin(2 * np.pi * f * time), np.cos(2 * np.pi * f * time)], axis=1)

        return harmonics

    def on_epoch_end(self):
        """Shuffle data after each epoch.

        This method is called after each epoch and shuffles the data if `shuffle=True`.

        """
        self.epoch += 1
        if self.shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        idx = np.random.permutation(len(self.x))
        self.x = self.x[idx, ...]
        self.index = self.index[idx, ...]

        if self.y is not None:
            self.y = self.y[idx, ...]

    def summary(self):
        """Print summary."""
        total_size = sum([dataset.size for dataset in self.datasets])
        total_length = sum([dataset.shape[1] for dataset in self.datasets])
        print(f'Number of datasets : {len(self.datasets):,}')
        print(f'Total data size    : {total_size:,}')
        print(f'Total data length  : {total_length:,}')
        print(f'Strides            : {self.strides:,}')
        print(f'Number of samples  : {self.nr_samples:,}')
        print(f'Batch size         : {self.batch_size:,}')
        print(f'Number of batches  : {len(self):,}')

        act_batch_size = self.batch_size * self.repeat_samples
        print(f'Sample repetitions : {self.repeat_samples:,}')
        print(f'Actual batch size  : {self.batch_size:,} * {self.repeat_samples} = {act_batch_size:,}')

        print(f'Shuffle            : {self.shuffle}')

        nx, ny = self.filter_length
        if (nx is not None) or (ny is not None):
            print('Filter length')
            print(f'  input      : {nx}')
            print(f'  prediction : {ny}')

        if self.ensemble_size is not None:
            print('Ensemble condition')
            print(f'  size : {self.ensemble_size}')
            print(f'  type : {self.ensemble_type}')
            if self.ensemble_type in ['random', 'random_full']:
                print(f'  range   : {self.ensemble_range}')
                print(f'  sync    : {self.ensemble_sync}')
                print(f'  replace : {self.ensemble_replace}')

        channels = tuple(range(self.channels))[self.input_channels]
        if len(channels) == self.channels:
            print(f'Input channels     : all')
        else:
            print(f'Input channels     : {channels}')

        if self.prediction_length:
            channels = tuple(range(self.channels))[self.prediction_channels]
            if len(channels) == self.channels:
                print(f'Predicted channels : all')
            else:
                print(f'Predicted channels : {channels}')

        # get samples
        items = self.__getitem__(0)
        # unpack items
        inputs, targets, sample_weights, *_ = chain(items, [None] * 3)

        print('Output shapes')
        print('  inputs')
        for key, value in inputs.items():
            print(f'    {key:<16.16} : {value.shape}')

        if targets is not None:
            print('  targets')
            for key, value in targets.items():
                print(f'    {key:<16.16} : {value.shape}')

        if sample_weights is not None:
            print('  sample_weights')
            for key, value in sample_weights.items():
                print(f'    {key:<16.16} : {value.shape}')


class PredictGenerator(FitGenerator):
    """Generator class for model prediction.

    The generator prepares the inputs for the model prediction with :func:`ks.Model.predict`.

    Parameters:
        **kwargs:
            See :class:`FitGenerator` for parameters.

    Returns:
        Dictionary containing the inputs for the model prediction.

    """
    def __getitem__(self, idx: int) -> dict:
        """Return inputs to the model for given batch."""
        inputs, _ = super().__getitem__(idx)
        return inputs


def example_FitGenerator():
    """Example of :class:`FitGenerator`.

    This example shows how to use the :class:`FitGenerator` class.

    """

    # first we create some dummy data
    shape = (1, 32, 3)  # (set_size, data_length, channels)
    dataset = np.reshape(np.arange(np.prod(shape)), shape)
    datasets = [dataset] * 3

    # the corresponding time values
    time = range(shape[1])

    # and the corresponding conditions
    # the encoder and decoder conditions are different
    encoder_cond = np.linspace(-1, 1, 32)
    decoder_cond = np.linspace(1, -1, 32)

    # then we create the generator
    fit_gen = FitGenerator(datasets,
                           condition={
                               'encoder': encoder_cond,
                               'decoder': decoder_cond
                           },
                           input_length=1,
                           prediction_length=4,
                           batch_size=128,
                           ensemble_size=len(datasets),
                           ensemble_type='index',
                           tp_period=12,
                           time=time,
                           shuffle=False)

    # we can see the summary of the generator
    fit_gen.summary()

    # we can now use the generator to get the inputs for the model
    inputs, *_ = fit_gen[0]

    # we can plot the inputs, to see what the model will get
    # we show the encoder and decoder conditions
    fig, (lax, rax) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16, 5))
    lax.pcolormesh(inputs['encoder_cond'][:, 0, :])
    lax.set_title("inputs['encoder_cond']")
    mp = rax.pcolormesh(inputs['decoder_cond'][:, 0, :])
    rax.set_title("inputs['decoder_cond']")

    fig.colorbar(mp, ax=(lax, rax))
