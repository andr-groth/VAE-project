# -*- coding: utf-8 -*-
"""Collection of generators for many-to-many version of VAE.

Generator class for netCDF data given as instance of :class:`iowrapper.DataHandler`.

Generator class for model training of models in :mod:`models_multi`.

@author: Andreas Groth
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorflow.keras as ks

if TYPE_CHECKING:
    from VAE._iowrapper import DataHandler
    from VAE.utils.beta_schedulers import BetaScheduler

__version__ = '2021-11-15'

# 2021-12-15 : option ensemble_size added
# 2021-12-16 : code clean-up. Removed options scale_augmentation, offset_augmentation, combinations, overlap.
# 2021-12-28: refactor __getitem__ and provide separate getters for parts of inputs, targets and sample_weights
# 2021-12-30: encoder and decoder(s) get separate random samples from get_ensemble_condition


# TODO: use different decoder and prediction ensemble condition
class FitGenerator(ks.utils.Sequence):
    """Generator class for model training.

    Given an instance of :class:`DataHandler`, the generator prepares the `inputs`, `targets`, and `sample_weights` for
    the model training with :func:`ks.Model.fit`.

    Parameters:
        data : DataHandler
            Instance of :class:`DataHandler`
        input_length : int
            Length of input to the encoder.
        input_names : list of str
            Names of input variables as str. If `None`, all variables from `data` are taken.
        time_range : tuple of two str
            Temporal limits.
        lat_range : tuple of two float
            Latitute limits.
        lon_range : tuple of two float
            Longitude limits.
        strides : tuple of three int
            Tuple of length 3, specifying the sample strides for time, latitude, and longitude.
        prediction_names : list of str
            Names of target variables for prediction as str. Leave ``None`` for pure auto-encoders
            without prediction.
        prediction_length : int
            Length of target for prediction.
        cond_name : str
            Name of variable for spatial condition.
        tp_name : str
            Name of variable for temporal condition.
        tp_fix : str
            String specifying a datetime to which the temporal condition is fixed.
        latent_dim : int
            Number of latent variables for which the generator will return a random sample. Note
            that the random variable will depend on time only and be independent of latitude and
            longitude. The random variable will change between batches. Leave this as `None` will use the built-in
            random sampling in `latent_sampling`.
        batch_size : int
            Batch size.
        set_size : int
            Set size. Number of samples in a set. This is the leading dimenions after the batch dimensions.
        ensemble_size : int
            Ensemble size for which the generator will return additional values for the condition. The ensemble members
            will be one-hot encoded.
        initial_epoch : int
            Initial epoch at which the generator will start. This will effect the `beta` parameter.
        beta_scheduler : BetaScheduler
            Instance of :class:`BetaScheduler` that returns the `beta` parameters for the KL
            loss in each epoch.
        beta_cond_scheduler : BetaScheduler
            Instance of :class:`BetaScheduler` that returns additional values for the
            condition. These values are also multiplied with the `beta` parameters of the `beta_scheduler` before
            applied to the KL loss.
        shuffle : bool
            Shuffle samples order.

    Returns:
        tuple `(inputs, targets, sample_weights)`.

    """
    def __init__(self,
                 data: DataHandler,
                 input_length: int,
                 input_names: tuple[str, ...] = None,
                 time_range: tuple[str, str] = None,
                 lat_range: tuple[float, float] = None,
                 lon_range: tuple[float, float] = None,
                 strides: tuple[int, int, int] = (1, 1, 1),
                 prediction_length: int = 0,
                 prediction_names: tuple[str, ...] = None,
                 cond_name: str = None,
                 tp_name: str = None,
                 tp_fix: str = None,
                 latent_dim: int = None,
                 shuffle: bool = False,
                 batch_size: int = 32,
                 set_size: int = 1,
                 ensemble_size: int = None,
                 initial_epoch: int = 0,
                 beta_scheduler: BetaScheduler = None,
                 beta_cond_scheduler: BetaScheduler = None,
                 **kwargs):
        """Init class."""
        if data.variables is None:
            raise ValueError("Load data first with read().")

        if input_names is None:
            input_names = data.var_names
        else:
            if set(input_names) - set(data.var_names):
                raise ValueError(
                    "Unknown variable name in `input_names`: {}".format(set(input_names) - set(data.var_names)))

        if prediction_names is not None:
            if set(prediction_names) - set(data.var_names):
                raise ValueError("Unknown variable name in `prediction_names`: {}".format(
                    set(prediction_names) - set(data.var_names)))
            if prediction_length <= 0:
                raise ValueError("Parameter `prediction_length` must be larger than 0.")

        if cond_name is not None:
            if cond_name not in data.variables:
                raise KeyError("Unknown variable in `cond_name`: {}".format(cond_name))

        if tp_name is not None:
            if tp_name not in data.variables:
                raise KeyError("Unknown variable in `tp_name`: {}".format(tp_name))

        self.data = data
        self.input_length = input_length
        self.input_names = input_names
        self.prediction_length = prediction_length
        self.prediction_names = prediction_names
        self.strides = strides
        self.cond_name = cond_name
        self.tp_name = tp_name
        self.latent_dim = latent_dim
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.set_size = set_size
        self.ensemble_size = ensemble_size
        self.epoch = initial_epoch
        self.beta_scheduler = beta_scheduler
        self.beta_cond_scheduler = beta_cond_scheduler

        if tp_fix is not None:
            tp_fix = data.get_time_index(tp_fix, tp_fix)[0]
        self.tp_fix = tp_fix

        # get valid time indices
        if time_range is None:
            time_idx = np.arange(len(self.data.datetime))
        else:
            time_idx = self.data.get_time_index(time_range[0], time_range[1])
        time_idx = time_idx[input_length:len(time_idx) - prediction_length]

        # get latitude indices
        if lat_range is None:
            lat_idx = np.arange(len(self.data.lat))
        else:
            lat_idx = self.data.get_lat_index(lat_range[0], lat_range[1])

        # get longitude indices
        if lon_range is None:
            lon_idx = np.arange(len(self.data.lon))
        else:
            lon_idx = self.data.get_lon_index(lon_range[0], lon_range[1])

        # strides
        self.time_idx = time_idx[::strides[0]]
        self.lat_idx = lat_idx[::strides[1]]
        self.lon_idx = lon_idx[::strides[2]]

        # get spatial combinations
        latlat, lonlon = np.broadcast_arrays(self.lat_idx[:, None], self.lon_idx[None, :])
        self.samples = dict(
            lat_idx=latlat.flatten(),
            lon_idx=lonlon.flatten(),
        )

        # combine spatial samples into (random) sets
        self.sets = dict()
        for key, value in self.samples.items():
            self.sets[key] = np.split(value[:self.nr_sets * self.set_size], self.nr_sets)

        # combine spatial sets and time indices into batches
        a, b = np.broadcast_arrays(self.time_idx[:, None], np.arange(self.nr_sets)[None, :])
        batches = np.stack([a.flatten(), b.flatten()], axis=0).T
        self.batch_list = np.split(batches, range(self.batch_size, len(batches), self.batch_size))

        # optionally shuffle sets and batches
        if self.shuffle:
            self.shuffle_sets()
            self.shuffle_batches()

    def __getitem__(self, idx):
        """Return inputs, targets, and sample weights to the model for given batch.

        Parameters:
            idx:
                Batch index.

        Returns:
            Tuple of (inputs, targets, sample_weights).
                inputs : dict of Numpy arrays.
                targets : dict of Numpy arrays.
                sample_weights : dict of Numpy arrays.
        """
        if idx >= self.__len__():
            raise IndexError('batch index out of range')

        inputs = dict()
        targets = dict()
        sample_weights = dict()

        # input to encoder = target for decoder
        encoder_input = self.get_encoder_input(idx)
        inputs['encoder_input'] = encoder_input
        targets['decoder'] = encoder_input

        # sample weights
        sample_weights['decoder'] = self.get_sample_weights(idx)

        # beta parameter
        beta = self.get_beta_values(idx)
        if beta is not None:
            inputs['beta'] = beta

        # special latent sample
        latent_sigma = self.get_latent_sample(idx)
        if latent_sigma is not None:
            inputs['latent_sigma'] = latent_sigma

        # collectors of conditions for encoder and decoder
        encoder_cond_list = []
        decoder_cond_list = []

        # spatial condition
        spatial_condition = self.get_spatial_condition(idx)
        if spatial_condition is not None:
            encoder_cond_list += [spatial_condition]
            decoder_cond_list += [spatial_condition]

        # temporal condition
        temporal_condition = self.get_temporal_condition(idx)
        if temporal_condition is not None:
            encoder_cond_list += [temporal_condition]
            decoder_cond_list += [temporal_condition]

        # beta condition
        beta_condition = self.get_beta_condition(idx)
        if beta_condition is not None:
            encoder_cond_list += [beta_condition]
            decoder_cond_list += [beta_condition]

            #  scales the KL loss accordingly
            if 'beta' in inputs:
                inputs['beta'] *= beta_condition[:, 0, :]
            else:
                inputs['beta'] = beta_condition[:, 0, :]

        # separate ensemble conditions for encoder and decoder
        ensemble_condition_enc = self.get_ensemble_condition(idx)
        if ensemble_condition_enc is not None:
            encoder_cond_list += [ensemble_condition_enc]

        ensemble_condition_dec = self.get_ensemble_condition(idx)
        if ensemble_condition_dec is not None:
            decoder_cond_list += [ensemble_condition_dec]

        # concatenate list of conditions for encoder
        if encoder_cond_list:
            encoder_cond = np.concatenate(encoder_cond_list, axis=-1)
            inputs['encoder_cond'] = encoder_cond

        # concatenate list of conditions for decoder
        if decoder_cond_list:
            decoder_cond = np.concatenate(decoder_cond_list, axis=-1)
            inputs['decoder_cond'] = decoder_cond

        # target, condition and sample weights for prediction
        prediction_target = self.get_prediction_target(idx)
        if prediction_target is not None:
            targets['prediction'] = prediction_target
            sample_weights['prediction'] = sample_weights['decoder']

            if decoder_cond_list:
                inputs['prediction_cond'] = decoder_cond

        return inputs, targets, sample_weights

    def __len__(self):
        """Return number of batches."""
        return len(self.batch_list)

    def get_beta_condition(self, idx: int):
        """Return beta condition for given batch.

        The function returns samples from `beta_cond_scheduler`. Returns `None` if `beta_cond_scheduler` is `None`.

        Parameters:
            idx: Batch index

        Returns:
            3D Numpy array of shape `(batch_size, set_size, 1)`.
        """
        if self.beta_cond_scheduler is not None:
            batch = self.batch_list[idx]
            batch_size = len(batch)
            condition = self.beta_cond_scheduler(self.epoch, shape=(batch_size, 1, 1))
            condition = np.broadcast_to(condition, shape=(batch_size, self.set_size, 1))
        else:
            condition = None

        return condition

    def get_beta_values(self, idx: int):
        """Return beta values for given batch.

        The function returns samples from `beta_scheduler`. Returns `None` if `beta_scheduler` is `None`.

        Parameters:
            idx : int
                Batch index

        Returns:
            2D Numpy array of shape `(batch_size, 1)`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        if self.beta_scheduler is not None:
            beta = self.beta_scheduler(self.epoch, shape=(batch_size, 1))
        else:
            beta = None

        return beta

    def get_config(self):
        """Return dictionary of configurations."""
        config = {
            'strides': self.strides,
            'set_size': self.set_size,
            'tp_fix': str(self.data.datetime[self.tp_fix]) if self.tp_fix is not None else None,
        }
        return config

    def get_encoder_input(self, idx: int):
        """Returns input to encoder.

        Parameters:
            idx : int
                Batch index.

        Returns:
            4D Numpy array of shape `(batch_size, set_size, input_length, len(input_names))`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        input_shape = (batch_size, self.set_size, self.input_length, len(self.input_names))
        encoder_input = np.zeros(input_shape, dtype=self.data.dtype)

        for batch_idx, (time_idx, set_idx) in enumerate(batch):
            for channel, name in enumerate(self.input_names):
                # data dimensions are time, lat, lon
                x = self.data.variables[name][time_idx - self.input_length:time_idx, self.sets['lat_idx'][set_idx],
                                              self.sets['lon_idx'][set_idx]]
                encoder_input[batch_idx, :, :, channel] = x.T

        return encoder_input

    def get_ensemble_condition(self, idx: int):
        """Return ensemble condition for given batch.

        For each sample in the batch, a random one-hot encoded ensemble index is return. The code is the same along
        the second dimension of size `set_size`.

        Parameters:
            idx : int
                Batch index.

        Returns:
            3D Numpy array of shape `(batch_size, set_size, ensemble_size)`
        """
        #
        batch = self.batch_list[idx]
        batch_size = len(batch)
        if self.ensemble_size:
            ensemble_idx = np.random.randint(low=0, high=self.ensemble_size, size=batch_size)
            condition = np.zeros((batch_size, self.set_size, self.ensemble_size), dtype=self.data.dtype)
            condition[np.arange(batch_size), :, ensemble_idx] = 1
        else:
            condition = None

        return condition

    def get_indices(self, idx: int):
        """Return array of time, lat, lon indices.

        Array of time, lat, lon indices of given batch. The array has shape `(batch_size, set_size, 3)`, with time, lat,
        lon indices arranged along the last dimension.

        Parameters:
            idx : int
                Batch index.

        Returns:
            3D Numpy array of shape `(batch_size, set_size, 3)`.
        """
        batch = self.batch_list[idx]
        indices = np.zeros((len(batch), self.set_size, 3), dtype=int)
        for batch_idx, (time_idx, set_idx) in enumerate(batch):
            indices[batch_idx, :, 0] = time_idx
            indices[batch_idx, :, 1] = self.sets['lat_idx'][set_idx]
            indices[batch_idx, :, 2] = self.sets['lon_idx'][set_idx]

        return indices

    def get_latent_sample(self, idx: int):
        """Draw optional random sample for latent sampling.

        The optional latent sample is only time-dependent and otherwise kept spatially fixed within the batch.

        Parameters:
            idx : int
                Batch index.

        Returns:
            2D Numpy array of shape `(batch_size, latent_dim)`.
        """
        if self.latent_dim is not None:
            batch = self.batch_list[idx]
            batch_size = len(batch)
            sigma = np.random.randn(len(self.data.datetime), self.latent_dim).astype(self.data.dtype)

            latent_shape = (batch_size, self.latent_dim)
            latent_sigma = np.zeros(latent_shape, dtype=self.data.dtype)
            for batch_idx, (time_idx, _) in enumerate(batch):
                latent_sigma[batch_idx, :] = sigma[time_idx, :]
        else:
            latent_sigma = None

        return latent_sigma

    def get_prediction_target(self, idx: int):
        """Return target for prediction for given batch.

        Parameters:
            idx : int
                Batch index.

        Returns:
            4D Numpy array of shape `(batch_size, set_size, prediction_length, len(prediction_names))`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        if (self.prediction_names is not None) and (self.prediction_length > 0):
            prediction_shape = (batch_size, self.set_size, self.prediction_length, len(self.prediction_names))
            prediction_target = np.zeros(prediction_shape, dtype=self.data.dtype)

            for batch_idx, (time_idx, set_idx) in enumerate(batch):
                for channel, name in enumerate(self.prediction_names):
                    x = self.data.variables[name][time_idx:time_idx + self.prediction_length,
                                                  self.sets['lat_idx'][set_idx], self.sets['lon_idx'][set_idx]]
                    prediction_target[batch_idx, :, :, channel] = x.T
        else:
            prediction_target = None

        return prediction_target

    def get_sample_weights(self, idx: int):
        """Return sample weights for given batch.

        The sample weights are defined as the cosine of the latitude.

        Parameters:
            idx : int
                Batch index.

        Returns:
            2D Numpy array of shape `(batch_size, set_size)`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        sample_weights = np.zeros((batch_size, self.set_size), dtype=self.data.dtype)

        for batch_idx, (_, set_idx) in enumerate(batch):
            for sample_idx, lat_idx in enumerate(self.sets['lat_idx'][set_idx]):
                sample_weights[batch_idx, sample_idx] = self.data.get_lat_weight(lat_idx)

        return sample_weights

    def get_spatial_condition(self, idx: int):
        """Return spatial condition for given batch.

        Parameters:
            idx : int
                Batch index.

        Returns:
            3D Numpy array of shape `(batch_size, set_size, cond_size)`
                `cond_size` refers to the size of the leading dimension of the spatial condition array in `cond_name`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        if self.cond_name is not None:
            cond_size = self.data.variables[self.cond_name].shape[0]
            shape = (batch_size, self.set_size, cond_size)
            condition = np.zeros(shape, dtype=self.data.dtype)

            for batch_idx, (_, set_idx) in enumerate(batch):
                x = self.data.variables[self.cond_name][:, self.sets['lat_idx'][set_idx], self.sets['lon_idx'][set_idx]]
                condition[batch_idx, :, :] = x.T

        else:
            condition = None

        return condition

    def get_temporal_condition(self, idx: int):
        """Return temporal condition for given batch.

        Parameters:
            idx : int
                Batch index.

        Returns:
            3D Numpy array of shape `(batch_size, set_size, cond_size)`
                `cond_size` refers to the size of the last dimension of the temporal condition array in `tp_name`.
        """
        batch = self.batch_list[idx]
        batch_size = len(batch)
        if self.tp_name is not None:
            tp_size = self.data.variables[self.tp_name].shape[1]
            shape = (batch_size, self.set_size, tp_size)
            condition = np.zeros(shape, dtype=self.data.dtype)

            if self.tp_fix is None:
                for sample_idx, (time_idx, _) in enumerate(batch):
                    condition[sample_idx, ...] = self.data.variables[self.tp_name][time_idx, :]
            else:
                condition[...] = self.data.variables[self.tp_name][self.tp_fix, :]

        else:
            condition = None

        return condition

    @property
    def nr_samples(self):
        """Number of spatial samples."""
        return len(self.samples['lat_idx'])

    @property
    def nr_sets(self):
        """Number of spatial sets."""
        return self.nr_samples // self.set_size

    def on_epoch_end(self):
        """Run on epoch end of model training."""
        # increment epoch number
        self.epoch += 1
        if self.shuffle:
            self.shuffle_sets()
            self.shuffle_batches()

    def shuffle_sets(self):
        """Shuffle items in sets."""
        idx = np.random.permutation(self.nr_samples)
        for key, value in self.samples.items():
            value_shuffled = value[idx]
            self.sets[key] = np.split(value_shuffled[:self.nr_sets * self.set_size], self.nr_sets)

    def shuffle_batches(self):
        """"Shuffle items in batch list."""
        batches = np.concatenate(self.batch_list)
        idx = np.random.permutation(len(batches))
        batches_shuffled = batches[idx, :]
        self.batch_list = np.split(batches_shuffled, range(self.batch_size, len(batches_shuffled), self.batch_size))

    def summary(self, line_length: int = 80):
        """Print summary."""
        print("=" * line_length)
        print("Summary of {}".format(type(self).__name__))
        print('-' * line_length)

        print("Time      samples/range      : {:,} / ({!s}, {!s}, {!s})".format(
            len(self.time_idx),
            self.data.datetime[self.time_idx[0]].astype('datetime64[D]'),
            self.data.datetime[self.time_idx[-1]].astype('datetime64[D]'),
            (self.data.datetime[self.time_idx[1]] - self.data.datetime[self.time_idx[0]]).astype('timedelta64[D]'),
        ))
        print("Latitude  samples/range      : {:,} / ({:}, {:}, {:})".format(
            len(self.lat_idx),
            self.data.lat[self.lat_idx[0]],
            self.data.lat[self.lat_idx[-1]],
            self.data.lat[self.lat_idx[1]] - self.data.lat[self.lat_idx[0]],
        ))
        print("Longitude samples/range      : {:,} / ({:}, {:}, {:})".format(
            len(self.lon_idx),
            self.data.lon[self.lon_idx[0]],
            self.data.lon[self.lon_idx[-1]],
            self.data.lon[self.lon_idx[1]] - self.data.lon[self.lon_idx[0]],
        ))

        print("Spatial combinations         : {:,} x {:,} = {:,}".format(len(self.lat_idx), len(self.lon_idx),
                                                                         self.nr_samples))

        print("Spatial set size             : {:,}".format(self.set_size))
        print("Spatial sets                 : {:,}".format(self.nr_sets))
        print("Spatio-temporal combinations : {:,} x {:,} = {:,}".format(
            self.nr_sets,
            len(self.time_idx),
            self.nr_sets * len(self.time_idx),
        ))
        print("Batch size                   : {:,}".format(self.batch_size))
        print("Number of batches            : {:,}".format(len(self.batch_list)))
        print("Shuffle                      : {}".format(self.shuffle))
        print("Input names                  : {}".format(self.input_names))
        print("Input length                 : {:,}".format(self.input_length))

        if self.prediction_names is not None:
            print("Prediction names             : {}".format(self.prediction_names))
            print("Prediction length            : {:,}".format(self.prediction_length))

        if self.tp_fix is not None:
            print("Temporal encoding fixed to   : {}".format(self.data.datetime[self.tp_fix]))


class PredictGenerator(FitGenerator):
    """Generator class for model prediction.

    Given an instance of :class:`DataHandler`, the generator prepares the inputs for the model prediction with
    :func:`ks.Model.predict`.

    Parameters : multiple
        See :class:`FitGenerator`.

    Returns:
        dict containing the inputs for the model prediction

    See also:
        :class:`FitGenerator`: Generator class for model training.

    """
    def __getitem__(self, idx) -> dict:
        """Return inputs to the model for given batch."""
        inputs, _, _ = super().__getitem__(idx)
        return inputs


class ModelOutputGenerator(PredictGenerator):
    """Generic class for model prediction."""
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return output of the model for given batch.

        Parameters:
            idx : int
                Batch index.
        Returns:
            tuple of two ndarrays containing the indices and model output
             indices : 2D Numpy array
                Array of indices of shape `(batch_size * set_size, 3)`, where the columns correspond
                to the time index, latitude index, and longitude index, respectively.
            outputs : Numpy array
                Corresponding model outputs of shape `(batch_size * set_size,) + output_shape`, where
                `output_shape` is the shape of either the decoder or the prediction output.
        """

        raise NotImplementedError('Subclasses should override __getitem__.')

    def _get_inputs(self, idx: int) -> dict:
        return super().__getitem__(idx)


class ModelPredictGenerator(ModelOutputGenerator):
    """Generator class for model prediction.

    Given an instance of :class:`DataHandler` and :class:`ks.Model`, the generator feeds the inputs to the `model` and
    returns the `model` output. The model output can be merged with :func:`DataHandler.merge_generator`.

    Parameters:
        data : DataHandler
            Instance of :class:`DataHandler`.
        model : ks.Model
            Instance of :class:`ks.Model`, specifying the model.
        model_output_name : str
            Name of the model output that will be returned. Must be one of `{'decoder', 'prediction'}`.
        **kwargs : multiple
            Parameters from :class:`FitGenerator`.

    Returns:
        see :class:`ModelPredictGenerator` for more information

    See also:
        :class:`DataHandler` : Class to handle the input data.
        :class:`ModelPredictGenerator` : Generator class for model training.

    """
    def __init__(self, data: DataHandler, model: ks.Model, model_output_name: str = 'decoder', **kwargs):
        """Init self."""
        super().__init__(data, **kwargs)
        self.model = model

        if model_output_name not in set(model.output_names):
            raise ValueError("Model has no output with name `{}`".format(model_output_name))

        self.model_output_name = model_output_name
        self.model_output_index = self.model.output_names.index(self.model_output_name)

    def __getitem__(self, idx):
        inputs = super()._get_inputs(idx)

        # get model output
        outputs = self.model.predict_on_batch(inputs)
        if isinstance(outputs, list):
            outputs = outputs[self.model_output_index]

        # get indices
        indices = self.get_indices(idx)

        # flatten batch and set dimensions, i.e. axis=(0, 1)
        outputs = np.reshape(outputs, (-1, ) + outputs.shape[2:])
        indices = np.reshape(indices, (-1, 3))

        return indices, outputs

    def get_config(self):
        config = {
            'model_output_name': self.model_output_name,
        }
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

    def summary(self, line_length=80):
        super().summary(line_length=line_length)

        print("Model output name            : {}".format(self.model_output_name))


class ModelResponseGenerator(ModelOutputGenerator):
    """Generator class for model response.

    Given an instance of :class:`DataHandler` and three instances of :class:`ks.Model`, the generator feeds an input to
    the `encoder`, applies the transformation to the `encoder` output, draws random samples from `latent_sampling`, and
    returns the `decoder` output. The finally decoder output can be merged with :func:`DataHandler.merge_generator`.

    Given the latent variables `z_mean` and `z_log_var from the encoder output, the following affin transformation is
    applied to the elements in `latent_dims_pert`:
    >>> for i in latent_dims_pert:
    >>>     z_mean_pert[i] = z_mean[i] * z_mean_scale + z_mean_offset
    >>>     z_log_var_pert[i] = z_log_var[i] * z_log_var_scale + z_log_var_offset

    The (partly) modified latent variables `z_mean_pert` and `z_log_var_pert` are then fed to `latent_sampling` to draw
    modified random latent samples, which are then fed to the `decoder` to finally obtain the modified output.

    Parameters:
        data : DataHandler
            Instance of :class:`DataHandler`.
        encoder : ks.Model
            Instance of :class:`ks.Model`, specifying the encoder.
        decoder : ks.Model
            Instance of :class:`ks.Model`, specifying the decoder.
        latent_sampling : keras.Model
            Instance of :class:`keras.Model`, specifying the latent sampling.
        latent_dims_pert : list of int
            List of latent dimensions to which the pertubation will be applied.
        z_mean_scale : float
            Scales the mean of the latent dimensions in `latent_dims_pert`. This specifies the relative strength of
            the pertubation on `z_mean`. Defaults to 1; i.e. no pertubation.
        z_log_var_scale : float
            Scales the log variance of the latent dimensions in `latent_dims_pert`. This specifies the
            relative strength of the pertubation on `z_log_var`. Defaults to 1; i.e. no pertubation.
        z_mean_offset : float
            Shifts the mean of the latent dimensions in `latent_dims_pert`. Defaults to 0.
        z_log_var_offset : float
            Shifts the log variance of the latent dimensions in `latent_dims_pert`. Defaults to 0.
        **kwargs : multiple
            Parameters from :class:`FitGenerator`.

   Returns:
        see :class:`ModelPredictGenerator` for more information

    See also:
        :class:`DataHandler`: Class to handle the input data.
        :class:`FitGenerator`: Generator class for model training.

    """
    def __init__(self,
                 data: DataHandler,
                 encoder: ks.Model,
                 decoder: ks.Model,
                 latent_sampling: ks.Model,
                 latent_dims_pert: tuple[int, ...],
                 z_mean_scale=1.,
                 z_log_var_scale=1.,
                 z_mean_offset=0.,
                 z_log_var_offset=0.,
                 **kwargs):
        """Init self."""
        super().__init__(data, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sampling = latent_sampling
        self.latent_dims_pert = latent_dims_pert if latent_dims_pert else []
        self.z_mean_scale = z_mean_scale
        self.z_log_var_scale = z_log_var_scale
        self.z_mean_offset = z_mean_offset
        self.z_log_var_offset = z_log_var_offset
        self.model_output_name = decoder.output_names[0]

    def __getitem__(self, idx):
        inputs = super()._get_inputs(idx)

        # get latent variables
        z_mean, z_log_var = self.encoder.predict_on_batch(inputs)

        if self.latent_dims_pert:
            # apply pertubation to mean
            z_mean[:, self.latent_dims_pert] *= self.z_mean_scale
            z_mean[:, self.latent_dims_pert] += self.z_mean_offset

            # apply pertubation to log variance
            z_log_var[:, self.latent_dims_pert] *= self.z_log_var_scale
            z_log_var[:, self.latent_dims_pert] += self.z_log_var_offset

        z_inputs = [z_mean, z_log_var]
        if len(self.latent_sampling.inputs) == 3:
            z_inputs += inputs['latent_sigma']

        # draw random sample from pertubation
        zs = self.latent_sampling.predict_on_batch(z_inputs)

        if len(self.decoder.inputs) == 2:
            zs = [zs, inputs['decoder_cond']]

        # get corresponding decoder response = pertubation
        outputs = self.decoder.predict_on_batch(zs)

        # get indices
        indices = self.get_indices(idx)

        # flatten batch and set dimensions, i.e. axis=(0, 1)
        outputs = np.reshape(outputs, (-1, ) + outputs.shape[2:])
        indices = np.reshape(indices, (-1, 3))

        return indices, outputs

    def get_config(self) -> dict:
        config = {
            'model_output_name': self.model_output_name,
            'latent_dims_pert': self.latent_dims_pert,
            'z_mean_scale': self.z_mean_scale,
            'z_mean_offset': self.z_mean_offset,
            'z_log_var_scale': self.z_log_var_scale,
            'z_log_var_offset': self.z_log_var_offset,
        }
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

    def summary(self, line_length=80):
        super().summary(line_length=line_length)

        print("Model output name            : {}".format(self.model_output_name))
        print("Pertubated dimensions        : {}".format(self.latent_dims_pert))
        print("  Scale of mean              : {}".format(self.z_mean_scale))
        print("  Offset of mean             : {}".format(self.z_mean_offset))
        print("  Scale of log variance      : {}".format(self.z_log_var_scale))
        print("  Offset of log variance     : {}".format(self.z_log_var_offset))


class ModelCompositeGenerator(ModelOutputGenerator):
    """Generator class for model composite.

    Given an instance of :class:`iowrapper.DataHandler` and :class:`keras.Model`, the generator prepares the
    `inputs` for the model prediction and returns then model output. The model output can be merged with
    :func:`iowrapper.DataHandler.merge_generator`.

    Parameters:
        data : :class:`DataHandler`
            Instance of :class:`DataHandler`.
        encoder : :class:`ks.Model`
            Instance of :class:`ks.Model`, specifying the encoder.
        decoder : :class:`ks.Model`
            Instance of :class:`ks.Model`, specifying the decoder.
        latent_sampling : :class:`ks.Model`
            Instance of :class:`ks.Model`, specifying the latent sampling.
        latent_dims_comp : list of int
            List of latent dimensions for which the composite will be returned.
        latent_dims_mult : float
            Multiplier applied to the latent dimensions in `latent_dims_comp`. This specifies the relative
            strength of the pertubation for which the composite is returned. Defaults to 0; i.e. full pertubation.
        **kwargs : multiple
            Parameters from :class:`.FitGenerator`.

   Returns:
        see :class:`ModelPredictGenerator` for more information

    See also:
        :class:`iowrapper.DataHandler`: Class to handle the input data.
        :class:`.FitGenerator`: Generator class for model training.

    """
    def __init__(self, data, encoder, decoder, latent_sampling, latent_dims_comp, latent_dims_mult=0, **kwargs):
        """Init self."""
        super().__init__(data, **kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_sampling = latent_sampling
        self.latent_dims_comp = latent_dims_comp
        self.latent_dims_mult = latent_dims_mult
        self.model_output_name = decoder.output_names[0]

    def __getitem__(self, idx):
        inputs = super()._get_inputs(idx)

        # get latent variables
        z_mean, z_log_var = self.encoder.predict_on_batch(inputs)

        # apply pertubation to mean
        z_mean_pert = z_mean.copy()
        z_mean_pert[..., self.latent_dims_comp] *= self.latent_dims_mult

        # draw random samples
        if len(self.latent_sampling.inputs) == 2:
            z = self.latent_sampling.predict_on_batch([z_mean, z_log_var])
            z_pert = self.latent_sampling.predict_on_batch([z_mean_pert, z_log_var])
        else:
            latent_sigma = inputs['latent_sigma']
            z = self.latent_sampling.predict_on_batch([z_mean, z_log_var, latent_sigma])
            z_pert = self.latent_sampling.predict_on_batch([z_mean_pert, z_log_var, latent_sigma])

        # random permutation of latent dimensions that are not in latent_dims_comp
        order = np.random.permutation(len(z))
        z_perm = z[order, ...].copy()
        z_perm[..., self.latent_dims_comp] = z[..., self.latent_dims_comp]

        # apply same permutation to pertubation
        z_pert_perm = z_pert[order, ...].copy()
        z_pert_perm[..., self.latent_dims_comp] = z_pert[..., self.latent_dims_comp]

        # get corresponding decoder outputs
        if len(self.decoder.inputs) == 1:
            outputs = self.decoder.predict_on_batch(z_perm)
            outputs_pert = self.decoder.predict_on_batch(z_pert_perm)
        else:
            outputs = self.decoder.predict_on_batch([z_perm, inputs['decoder_cond']])
            outputs_pert = self.decoder.predict_on_batch([z_pert_perm, inputs['decoder_cond']])

        # get difference between the decoder outputs
        outputs -= outputs_pert

        # get indices
        indices = self.get_indices(idx)

        # flatten batch and set dimensions, i.e. axis=(0, 1)
        outputs = np.reshape(outputs, (-1, ) + outputs.shape[2:])
        indices = np.reshape(indices, (-1, 3))

        return indices, outputs

    def get_config(self) -> dict:
        config = {
            'model_output_name': self.model_output_name,
            'latent_dims_comp': self.latent_dims_comp,
            'latent_dims_mult': self.latent_dims_mult,
        }
        base_config = super().get_config()
        return dict(list(config.items()) + list(base_config.items()))

    def summary(self, line_length=80):
        super().summary(line_length=line_length)

        print("Model output name            : {}".format(self.model_output_name))
        print("Latent dimensions composite  : {}".format(self.latent_dims_comp))
        print("Pertubation multiplier       : {}".format(self.latent_dims_mult))
