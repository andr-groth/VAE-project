# -*- coding: utf-8 -*-
"""Input-output wrapper.

@author: Andreas Groth
"""

import glob
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import tensorflow.keras as ks
from scipy.special import sph_harm
from tensorflow.python.lib.io import file_io

__version__ = '2021-09-16'


class DataHandler:
    """Class to handle the netCDF data.

    Parameters:
            verbose : bool
                Verbose output.
            line_length : int
                Line length of verbose output.
            dtype : Data type
                Data type of the variables. Defaults to `np.float32`.

    Attributes:
        variables : dict
            Dictionary of numpy arrays that contain the data from the netCDF files.
        attributes : dict
            Dictionary of attributes corresponding to the variables.

    """
    def __init__(self, verbose: bool = True, line_length: int = 80, dtype=np.float32):
        """Initialize class."""
        self.verbose = verbose
        self.line_length = line_length
        self.dtype = dtype
        self.variables = dict()
        self.var_names = []
        self.attributes = dict()
        self.map_data = None
        self.map_params = None
        self.time_index = None
        self.lat = None
        self.lon = None

    def _verbose_msg(self, *args, **kwargs):
        """Print a message if verbose."""
        if self.verbose:
            print(*args, **kwargs)

    def read(self, files, prepend=None, append=None):
        """Read netCDF data.

        Parameters:
            files : str or iterable of str
                The glob pattern(s) specifying the netCDF file(s).
            prepend/append : callable, optional
                Callable that takes the filename as argument and returns a string. The returned string will be
                prepended/appended to the variable name.

        Note:
            The netCDF files are assumed to be in CF-1 convention (time, lat, lon) or (time, level, lat, lon). In the
            latter case, levels are split into separate variables.

        """
        self.filenames = file_io.get_matching_files(files)
        if not self.filenames:
            raise FileNotFoundError(f'No files found that match {os.path.normpath(files)}.')

        # Read data into dict of numpy arrays

        self._verbose_msg('=' * self.line_length)
        self._verbose_msg('Read netCDF files')
        self._verbose_msg('-' * self.line_length)

        for filename in self.filenames:
            self._verbose_msg(filename)

            try:
                # access GCS bucket
                memory = file_io.FileIO(filename, mode='rb')
                dataset = netCDF4.Dataset('name', memory=memory.read(), mode='r')

                for var_key, variable in dataset.variables.items():
                    if prepend is not None:
                        var_key = prepend(filename) + var_key
                    if append is not None:
                        var_key += append(filename)

                    # assume  CF-1 convention (time, lat, lon)
                    if variable.ndim == 3:
                        value = variable[:].filled(fill_value=np.nan).astype(self.dtype)

                        self.variables[var_key] = value
                        self.var_names.append(var_key)

                        self.attributes[var_key] = dict([(nc_attr, variable.getncattr(nc_attr))
                                                         for nc_attr in variable.ncattrs()])

                        time_key, lat_key, lon_key = variable.dimensions

                    # assume  CF-1 convention (time, level, lat, lon)
                    if variable.ndim == 4:
                        value = variable[:].filled(fill_value=np.nan).astype(self.dtype)

                        time_key, level_key, lat_key, lon_key = variable.dimensions
                        levels = dataset.variables[level_key][:]
                        nc_attrs = dict([(nc_attr, variable.getncattr(nc_attr)) for nc_attr in variable.ncattrs()])

                        # split levels into separate variables
                        for level_idx, level in enumerate(levels):
                            _var_key = var_key + '/{:g}'.format(level)
                            self.variables[_var_key] = value[:, level_idx, :, :]
                            self.var_names.append(_var_key)
                            cfg = {level_key: level}
                            self.attributes[_var_key] = dict(list(nc_attrs.items()) + list(cfg.items()))

                    # get time, lat, lon variables
                    if variable.ndim in {3, 4}:

                        time_index = dataset.variables[time_key][:].filled().astype(self.dtype)
                        lat = dataset.variables[lat_key][:].filled().astype(self.dtype)
                        lon = dataset.variables[lon_key][:].filled().astype(self.dtype)

                        if self.time_index is None:
                            self.time_index = time_index
                        else:
                            if not np.all(self.time_index == time_index):
                                raise ValueError("Mismatch in time dimension in {}: {}.".format(filename, var_key))

                        if self.lat is None:
                            self.lat = lat
                        else:
                            if not np.all(self.lat == lat):
                                raise ValueError("Mismatch in latitude dimension in {}: {}.".format(filename, var_key))

                        if self.lon is None:
                            self.lon = lon
                        else:
                            if not np.all(self.lon == lon):
                                raise ValueError("Mismatch in longitude dimension in {}: {}.".format(filename, var_key))

                        # use standardized keys for time, lat, and lon
                        for trg_key, src_key in [('time', time_key), ('lat', lat_key), ('lon', lon_key)]:
                            nc_attrs = dataset.variables[src_key].ncattrs()
                            self.attributes[trg_key] = dict([(nc_attr, dataset.variables[src_key].getncattr(nc_attr))
                                                             for nc_attr in nc_attrs])

            finally:
                dataset.close()
                memory.close()

        if self.var_names:
            # convert time index to numpy datetime64
            units = self.attributes['time']['units']
            if units.startswith('year'):
                self.time_index *= 12
                units = units.replace('year', 'month')
                self.attributes['time']['units'] = units

            if units.startswith('month'):
                calendar = '360_day'
            else:
                calendar = self.attributes['time'].get('calendar', 'standard')

            try:
                self.datetime = netCDF4.num2date(self.time_index,
                                                 units=units,
                                                 calendar=calendar,
                                                 only_use_cftime_datetimes=False,
                                                 only_use_python_datetimes=True)
            except ValueError:
                date = netCDF4.num2date(self.time_index, units=units, calendar=calendar)
                date = [np.datetime64(d.strftime('%Y-%m-%d')) for d in date]
                self.datetime = np.array(date)

            self.datetime = self.datetime.astype('datetime64[D]')
        else:
            self._verbose_msg("Not data found!")

    def prepare(self,
                anomalies_stride: int = None,
                anomalies_range: tuple[str, str] = None,
                normalize: bool = False,
                sph_degree: int = None,
                tp_period: int = None,
                fill_value: float = 0.):
        """Prepare data.

        Preprocess the data.

        Parameters:
            anomalies_stride : int, optional
                Variables are converted to anomalies. The parameter defines the stride of the average. Defaults to None.
            anomalies_range : Tuple[str, str], optional
                Time range that is used to obtain the anomalies. Defaults to None.
            normalize : bool, optional
                Normalize the data by removing the mean value and deviding by the standard deviation. Defaults to False.
            sph_degree : int, optional
                Create variable `sph` with spherical harmonics. See :func:`get_sph_harmonics`. Defaults to None.
            tp_period : int, optional
                Crreate variable `tph` with temporal harmonics. See :func:`get_tp_harmonics`. Defaults to None.
            fill_value : float, optional
                Value used to fill missing values. Defaults to 0.
        """
        if not self.var_names:
            print("No data available.")
            return

        self.anomalies_stride = anomalies_stride
        self.anomalies_range = anomalies_range
        self.normalize = normalize
        self.sph_degree = sph_degree
        self.tp_period = tp_period

        self._verbose_msg("=" * self.line_length)
        self._verbose_msg("Prepare data")
        self._verbose_msg("-" * self.line_length)

        # get anomalies
        self.anomalies = dict()
        if anomalies_stride is not None:
            self._verbose_msg("Convert data to anomalies.")

            if anomalies_range is not None:
                idx = self.get_time_index(*anomalies_range)
            for var_name in self.var_names:
                mean_fields = []
                for r in range(anomalies_stride):
                    mean_pos = np.arange(r, len(self.datetime), anomalies_stride)
                    if anomalies_range is not None:
                        mean_pos = mean_pos[np.searchsorted(mean_pos, idx[0]):np.
                                            searchsorted(mean_pos, idx[-1], side='right')]

                    mean = np.nanmean(self.variables[var_name][mean_pos, ...], axis=0)
                    self.variables[var_name][r::anomalies_stride, ...] -= mean
                    mean_fields.append(mean)

                self.anomalies[var_name] = mean_fields

        # normalize data
        if normalize:
            self._verbose_msg("Normalize data.")

            for var_name in self.var_names:
                offset = np.nanmean(self.variables[var_name])
                scale = np.nanstd(self.variables[var_name])
                self.variables[var_name] -= offset
                self.variables[var_name] /= scale
                self.attributes[var_name]['offset'] = offset
                self.attributes[var_name]['scale'] = scale
        else:
            for var_name in self.var_names:
                self.attributes[var_name]['offset'] = 0
                self.attributes[var_name]['scale'] = 1

        # fill values that has been set to nan in read()
        for var_name in self.var_names:
            idx = np.isnan(self.variables[var_name])
            self.variables[var_name][idx] = fill_value
            if np.sum(idx):
                self._verbose_msg("Fill {} missing values in {} with {}".format(
                    np.sum(idx),
                    var_name,
                    fill_value,
                ))

        # spherical harmonics
        if sph_degree is not None:
            self.variables['sph'] = self.get_sph_harmonics(sph_degree).astype(self.dtype)
            self.attributes['sph'] = dict(long_name='Spherical harmonics', degree=sph_degree)

        # Temporal harmonics
        if tp_period is not None:
            self.variables['tph'] = self.get_tp_harmonics(tp_period).astype(self.dtype)
            self.attributes['tph'] = dict(long_name='Temporal harmonics', period=tp_period)

    def merge_generator(self,
                        generator,
                        epochs: int = 1,
                        fill_value: float = np.nan,
                        taper: str = None,
                        start: int = None,
                        stop: int = None):
        """Merge model output.

        Merge model output from generator. Overlapping parts are averaged. To obtain the generator output
        without averaging overlapping parts, use :func:`stack_generator` instead.

        Parameters:
            generator : :class:`VAE.generator.ModelOutputGenerator`
                Instance of :class:`VAE.generator.ModelOutputGenerator`, returning tuples of `indices` and
                `outputs`.
            epochs : int
                Number of generator epochs that will be merged. This will average over different random sets if
                the generator has `shuffle=True` or over different realizations of the latent variables `z`.
            fill_value : float
                Fill value for missing values. Defaults to nan.
            taper : str
                Name of window function in :mod:`numpy` that is used to taper the model output.
            start : int
                Integer between 0 and the length of the model output specifying the start index of the model output that
                will be used.
            stop : int
                Similar to `model_start`, but specifying the end index of the model output that will be used.

        See also:
            :class:`VAE.generator.ModelPredictGenerator`: Generator class for model prediction.

            :func:`.stack_generator`: Stack generator output.

        """

        # get start and end of time slices, corresponding to generator.FitGenerator.__getitem__()
        if generator.model_output_name == 'decoder':
            # time_idx - generator.input_length : time_idx
            start_idx = -generator.input_length
            end_idx = 0
            names = generator.input_names
        elif generator.model_output_name == 'prediction':
            # time_idx : time_idx + generator.prediction_length
            start_idx = 0
            end_idx = generator.prediction_length
            names = generator.prediction_names
        else:
            raise ValueError("Unknown model output name: %s" % generator.model_output_name)

        if taper is not None:
            taper_fcn = getattr(np, taper)
            output_length = end_idx - start_idx
            taper_values = taper_fcn(2 * output_length + 2)  # symmetric taper of even length
            taper_values = taper_values[1:-1]  # exclude zero values at either end
            taper_values = taper_values[output_length + start_idx:output_length + end_idx]
            taper_values = taper_values[start:stop]  # restrict to start:stop interval
            taper_values = taper_values[:, None]  # add channel dimension
        else:
            taper_values = 1

        # allocate memory
        shape = (len(self.datetime), len(generator.lat_idx), len(generator.lon_idx), len(names))
        map_data = np.full(shape, fill_value=0, dtype=self.dtype)
        denom = np.full(shape, fill_value=0, dtype=self.dtype)

        # map original indices in self.variables to strided indices in map_data
        invert_lat_idx = np.zeros_like(self.lat, dtype=np.int32)
        invert_lon_idx = np.zeros_like(self.lon, dtype=np.int32)
        invert_lat_idx[generator.lat_idx] = range(len(generator.lat_idx))
        invert_lon_idx[generator.lon_idx] = range(len(generator.lon_idx))

        self._verbose_msg("=" * self.line_length)
        self._verbose_msg("Merge output of `{}` with {}".format(generator.model_output_name, type(generator).__name__))
        self._verbose_msg("-" * self.line_length)

        for epoch in range(epochs):
            self._verbose_msg("Epoch {}/{}".format(epoch + 1, epochs))
            if self.verbose:
                pbar = ks.utils.Progbar(len(generator), unit_name='batch', verbose=1)

            for indices, outputs in generator:
                for index, output in zip(indices, outputs):
                    time_idx, lat_idx, lon_idx = index

                    # map original indices to strided indices in map_data
                    _lat_idx = invert_lat_idx[lat_idx]
                    _lon_idx = invert_lon_idx[lon_idx]

                    # add output
                    map_data_target = map_data[time_idx + start_idx:time_idx + end_idx, _lat_idx, _lon_idx, :]
                    map_data_target[start:stop, :] += output[start:stop, :] * taper_values

                    # increase corresponding denominator
                    denom_target = denom[time_idx + start_idx:time_idx + end_idx, _lat_idx, _lon_idx, :]
                    denom_target[start:stop, :] += taper_values

                if self.verbose:
                    pbar.add(1)

            generator.on_epoch_end()

        # divide map_data by danominator
        is_valid = denom > 0
        map_data[is_valid] /= denom[is_valid]
        # fill unknown values with fill_value
        map_data[~is_valid] = fill_value

        # map to dict
        self.map_data = dict()
        for channel, name in enumerate(names):
            self.map_data[name] = map_data[..., channel]

        generator_config = generator.get_config()
        config = {
            'epochs': epochs,
            'taper': taper,
            'start': start,
            'stop': stop,
        }

        generator_config = dict(list(generator_config.items()) + list(config.items()))

        self.map_params = MapParams(time=self.datetime[generator.time_idx],
                                    latitude=self.lat[generator.lat_idx],
                                    longitude=self.lon[generator.lon_idx],
                                    model_output_name=generator.model_output_name,
                                    generator_name=type(generator).__name__,
                                    generator_config=generator_config)

    def stack_generator(self,
                        generator,
                        epochs: int = 1,
                        fill_value: float = np.nan,
                        start: int = None,
                        stop: int = None):
        """Stack generator output.

        Stack model output from generator. In contrast to :func:`merge_generator`, overlapping parts are not
        averaged, but stacked along the second dimensions. The size of the second axis corresponds to the length of
        the model output, optionally reduced by `start` and `stop`. Note that the generator output is still averaged
        over different epochs.

        Parameters:
            generator : :class:`generator.ModelOutputGenerator`
                Instance of :class:`generator.ModelOutputGenerator`, returning tuples of ``indices`` and
                ``outputs``.
            epochs : int
                Number of generator epochs that will be merged. This will average over different random sets if
                the generator has `shuffle=True` or over different realizations of the latent variables `z`.
            fill_value : float
                Fill value for missing values. Defaults to nan.
            start : int
                Integer between 0 and the length of the model output specifying the start index of the model output that
                will be used.
            stop : int
                Similar to `model_start`, but specifying the end index of the model output that will be used.

        See also:
            :class:`VAE.generator.ModelPredictGenerator`: Generator class for model prediction.
            :func:`merge_generator`: Merge generator output.

        """

        # probe generator output shape
        indices, outputs = generator[0]
        _, output_length, channels = outputs[:, start:stop, :].shape

        # output names
        if generator.model_output_name == 'decoder':
            start_idx = -generator.input_length
            end_idx = 0
            names = generator.input_names
        elif generator.model_output_name == 'prediction':
            start_idx = 0
            end_idx = generator.prediction_length
            names = generator.prediction_names
        else:
            raise ValueError("Unknown model output name: %s" % generator.model_output_name)

        # allocate memory
        shape = (len(generator.time_idx), output_length, len(generator.lat_idx), len(generator.lon_idx), channels)
        map_data = np.full(shape, fill_value=0, dtype=self.dtype)
        denom = np.full(shape, fill_value=0, dtype=self.dtype)

        # map original indices in self.variables to strided indices in map_data
        invert_time_idx = np.zeros_like(self.datetime, dtype=np.int32)
        invert_lat_idx = np.zeros_like(self.lat, dtype=np.int32)
        invert_lon_idx = np.zeros_like(self.lon, dtype=np.int32)
        invert_time_idx[generator.time_idx] = range(len(generator.time_idx))
        invert_lat_idx[generator.lat_idx] = range(len(generator.lat_idx))
        invert_lon_idx[generator.lon_idx] = range(len(generator.lon_idx))

        self._verbose_msg("=" * self.line_length)
        self._verbose_msg("Stack output of `{}` with {}".format(generator.model_output_name, type(generator).__name__))
        self._verbose_msg("-" * self.line_length)

        for epoch in range(epochs):
            self._verbose_msg("Epoch {}/{}".format(epoch + 1, epochs))
            if self.verbose:
                pbar = ks.utils.Progbar(len(generator), unit_name='batch', verbose=1)

            for indices, outputs in generator:
                for index, output in zip(indices, outputs):
                    time_idx, lat_idx, lon_idx = index

                    _time_idx = invert_time_idx[time_idx]
                    _lat_idx = invert_lat_idx[lat_idx]
                    _lon_idx = invert_lon_idx[lon_idx]

                    # add output
                    map_data[_time_idx, :, _lat_idx, _lon_idx, :] += output[start:stop, :]

                    # increase corresponding denominator
                    denom[_time_idx, :, _lat_idx, _lon_idx, :] += 1

                if self.verbose:
                    pbar.add(1)

            generator.on_epoch_end()

        # divide map_data by denominator
        is_valid = denom > 0
        map_data[is_valid] /= denom[is_valid]
        # fill unknown values with fill_value
        map_data[~is_valid] = fill_value

        # map to dict
        self.map_data = dict()
        for channel, name in enumerate(names):
            self.map_data[name] = map_data[..., channel]

        generator_config = generator.get_config()
        config = {
            'epochs': epochs,
        }
        generator_config = dict(list(generator_config.items()) + list(config.items()))

        self.map_params = MapParams(time=self.datetime[generator.time_idx],
                                    step=range(start_idx, end_idx)[start:stop],
                                    latitude=self.lat[generator.lat_idx],
                                    longitude=self.lon[generator.lon_idx],
                                    model_output_name=generator.model_output_name,
                                    generator_name=type(generator).__name__,
                                    generator_config=generator_config)

    def summary(self):
        """Print summary."""
        if self.variables is None:
            print("No data. Read data first with `read`.")
            return

        w = (self.line_length - 3 * 3) // 5
        s = '{!s:' + str(w) + '.' + str(w) + '}'
        s5 = ' | '.join([s, s, s, s, s])
        print('=' * self.line_length)
        print(s5.format('Variable', 'min', 'max', 'mean', 'std'))
        print('-' * self.line_length)

        for var_name in self.var_names:
            mi = np.amin(self.variables[var_name])
            ma = np.amax(self.variables[var_name])
            mean = np.mean(self.variables[var_name])
            std = np.std(self.variables[var_name])

            print(s5.format(var_name, mi, ma, mean, std))

        s4 = ' | '.join([s, s, s, s])
        print('=' * self.line_length)
        print(s4.format('Dimension', 'min', 'max', 'delta'))
        print('-' * self.line_length)
        print(
            s4.format(
                'time',
                np.amin(self.datetime),
                np.amax(self.datetime),
                (self.datetime[1] - self.datetime[0]).astype('timedelta64[D]'),
            ))
        print(s4.format('latitude', np.amin(self.lat), np.amax(self.lat), self.lat[1] - self.lat[0]))
        print(s4.format('longitude', np.amin(self.lon), np.amax(self.lon), self.lon[1] - self.lon[0]))

    def to_netcdf(self, filename: str, denormalize: bool = True):
        """Export map data to netCDF.

        Exports the map data from :func:`merge_generator` or :func:`stack_generator`.

        Parameters:
            filename : str
                Filename of the netCDF file.
            denormalize : bool, optional
                Whether to denormalize the map data. Defaults to True.

        """

        # TODO: invert anomalies

        if self.map_data is None:
            print("No map data found that can be merged. Run first `merge_generator` or `stack_generator`.")
            return

        self._verbose_msg('=' * self.line_length)
        self._verbose_msg('Export map to netCDF')

        # create file
        dataset = netCDF4.Dataset(filename, 'w')
        dataset.createDimension('time', len(self.map_params.time))
        dataset.createDimension('lat', len(self.map_params.latitude))
        dataset.createDimension('lon', len(self.map_params.longitude))

        # write time variable
        time = dataset.createVariable('time', int, dimensions=('time', ), zlib=True)
        t0 = np.datetime64(0, 'D')
        td = self.map_params.time - t0
        td = td.astype('timedelta64[D]').astype(int)
        time[:] = td
        time.long_name = self.attributes['time'].get('long_name', '')
        time.units = 'days since ' + np.datetime_as_string(t0, 'D')

        # write latitude variable
        lat = dataset.createVariable('lat', float, dimensions=('lat', ), zlib=True)
        lat[:] = self.map_params.latitude
        lat.long_name = self.attributes['lat'].get('long_name', '')
        lat.units = self.attributes['lat'].get('units', '')

        # write longitude variable
        lon = dataset.createVariable('lon', float, dimensions=('lon', ), zlib=True)
        lon[:] = self.map_params.longitude
        lon.long_name = self.attributes['lon'].get('long_name', '')
        lon.units = self.attributes['lon'].get('units', '')

        # write step variable
        if self.map_params.step is not None:
            dataset.createDimension('step', len(self.map_params.step))
            step = dataset.createVariable('step', int, dimensions=('step', ), zlib=True)
            step[:] = self.map_params.step
            step.long_name = 'Step'
            step.units = ''

        # write map data
        for key, value in self.map_data.items():
            if self.map_params.step is not None:
                data = dataset.createVariable(key, self.dtype, ('time', 'step', 'lat', 'lon'), zlib=True)
            else:
                data = dataset.createVariable(key, self.dtype, ('time', 'lat', 'lon'), zlib=True)

            data.long_name = self.attributes[key].get('long_name', '')
            data.dataset = self.attributes[key].get('dataset', '')

            if denormalize:
                value_denormalized = value.copy()
                value_denormalized *= self.attributes[key].get('scale')
                value_denormalized += self.attributes[key].get('offset')
                data[:] = value_denormalized
                data.units = self.attributes[key].get('units', '')
            else:
                data[:] = value
                data.factor = self.attributes[key].get('scale')
                data.offset = self.attributes[key].get('offset')
                data.units = self.attributes[key].get('units', '') + ' normalized'

        # write global attributes
        dataset.source = "VAE output of " + self.map_params.model_output_name
        dataset.description = "Merged with " + self.map_params.generator_name
        dataset.history = "Created " + str(np.datetime64('today'))

        generator_config = self.map_params.generator_config
        for key, value in generator_config.items():
            if value is not None:
                dataset.setncattr('generator_' + key, value)

        self._verbose_msg('-' * self.line_length)
        self._verbose_msg('Map saved in `{}`'.format(filename))

        dataset.close()

    def get_lat_index(self, minimum: float, maximum: float):
        """Return indices of the latitude within the given minimum and maximum range.

        Parameters:
            minimum : float
                Minimum value for the latitude.
            maximum : float
                Maximum value for the latitude.

        Returns:
            Numpy array of indices.
        """
        idx = np.flatnonzero(np.logical_and(minimum <= self.lat, self.lat <= maximum))

        return idx

    @lru_cache(maxsize=None)
    def get_lat_weight(self, index: int):
        """Return the weight of the given latitude index.

        The weight is defined as the cosine of the latitude.

        Parameters:
            index : int
                The index to get the latitude weight for. See `self.lat[index]` for corresponding latitude value.

        Returns:
            float
        """
        lat = self.lat[index]
        dlat = np.abs(self.lat[1] - self.lat[0]) / 2
        if lat > 0:
            lat -= dlat
        elif lat < 0:
            lat += dlat

        return np.cos(np.deg2rad(lat))

    def get_lon_index(self, minimum: float, maximum: float):
        """Return indices of the longitude within the given minimum and maximum.

        Parameters:
            minimum : float
                Minimum value for the longitude.
            maximum : float
                Maximum value for the longitude.

        Returns:
            Numpy array of indices.
        """

        # convert into range [0, 2*pi]
        minimum = np.deg2rad(minimum) % (2 * np.pi)
        maximum = np.deg2rad(maximum) % (2 * np.pi)
        lon = np.deg2rad(self.lon) % (2 * np.pi)

        if minimum < maximum:
            idx = np.flatnonzero(np.logical_and(minimum <= lon, lon <= maximum))
        elif minimum >= maximum:
            idx = np.flatnonzero(np.logical_or(minimum <= lon, lon <= maximum))

        return idx

    def get_time_index(self, minimum: str, maximum: str):
        """Return indices of the time within the given minimum and maximum.

        Parameters:
            minimum : str
                Minimum datetime.
            maximum : str
                Maximum datetime.

        Returns:
            Numpy array of indices.
        """
        a = np.searchsorted(self.datetime, np.datetime64(minimum), side='left')
        b = np.searchsorted(self.datetime, np.datetime64(maximum), side='right')
        return np.arange(a, b)

    def get_sph_harmonics(self, sph_degree: int, real_valued: bool = True):
        """Return spherical harmonics.

        Parameters:
            sph_degree : int
                Degree of spherical harmonics.
            real_valued : bool
                Whether real valued (True) or complex valued (False) spherical harmonics are returned. Defaults to True.

        Returns:
            3D Numpy array of shape `(nr_of_sph, nr_latitudes, nr_longitudes)
        """
        self._verbose_msg("Get spherical harmonics...", end='')

        dlat = (self.lat[1] - self.lat[0]) / 2
        lat = self.lat + dlat  # center pixels
        colat = np.pi / 2 - np.deg2rad(lat)  # [0, pi]
        lon = np.deg2rad(self.lon) % (2 * np.pi)  # [0, 2*pi]

        colat = colat[:, None]
        lon = lon[None, :]
        colat, lon = np.broadcast_arrays(colat, lon)

        sph = []
        for n in range(sph_degree + 1):
            self._verbose_msg('{:3}'.format(n), end='')

            if real_valued:
                for m in range(-n, n + 1):
                    s = sph_harm(m, n, lon, colat)
                    if m >= 0:
                        s = np.real(s)
                    else:
                        s = np.imag(s)
                    sph.append(s)
            else:
                for m in range(n + 1):
                    s = sph_harm(m, n, lon, colat)
                    sph.append(s)

        self._verbose_msg(".")

        return np.stack(sph, axis=0)

    def get_tp_harmonics(self, tp_period: int):
        """Return temporal harmonics.

        Parameters:
            tp_period : int
                Maximal period for the temporal harmonics.

        Returns:
            2D Numy array of shape `(nr_times, nr_harmonics)`.
        """
        self._verbose_msg("Get temporal harmonics.")

        f = np.fft.fftfreq(tp_period)[None, :]
        t_index = np.arange(len(self.datetime))[:, None]
        harmonics = np.concatenate([
            np.sin(2 * np.pi * f * t_index),
            np.cos(2 * np.pi * f * t_index),
        ], axis=1)

        return harmonics


class CopyToGS(ks.callbacks.Callback):
    """Copy remote machine output to bucket.

    Parameters:
        job_dir : str
            Job dir.
        log_dir : str
            VM log dir.
        filemask:
            List of strings, specifying one or more filename masks that will be copied; e.g.
            ``filemask = ['model.*', 'events.*']``.
        update_freq:
            'batch' or 'epoch' or integer.

    """
    def __init__(self, job_dir='.', log_dir='.', filemask=['*'], update_freq='epoch'):
        """Init class."""
        self.job_dir = job_dir
        self.log_dir = log_dir
        self.filemask = filemask
        self.update_freq = update_freq

        self.write()

    def on_epoch_end(self, epoch, logs):
        """Write on epoch end."""
        if self.update_freq == 'epoch':
            self.write()

    def on_batch_end(self, batch, logs):
        """Write on batch end."""
        if self.update_freq == 'batch':
            self.write()

        if isinstance(self.update_freq, (int, np.integer)):
            if batch % self.update_freq == 0:
                self.write()

    def write(self):
        """Write."""
        def gcscopy(job_dir, file_path):
            with file_io.FileIO(file_path, mode='rb') as input_f:
                with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
                    output_f.write(input_f.read())

        if self.job_dir.startswith('gs://') and len(self.filemask) > 0:
            for fmask in self.filemask:
                files = glob.glob(os.path.join(self.log_dir, fmask))
                if len(files) > 0:
                    # sort files by creation time
                    files.sort(key=lambda x: os.path.getmtime(x))
                    # copy latest file
                    gcscopy(self.job_dir, files[-1])


@dataclass
class MapParams:
    """Class to handle the map parameters."""
    time: np.datetime64
    latitude: np.ndarray
    longitude: np.ndarray
    model_output_name: str
    generator_name: str
    generator_config: dict
    step: np.ndarray = None


if __name__ == '__main__':
    # filename = os.path.expanduser(r'~\Documents\ClimateData\ERA5\*.nc')
    filename = os.path.expanduser('~/Documents/ClimateData/20CRv3/20CRv3.air.2m.mon.mean.nc')

    data = DataHandler(dtype=np.float32)
    data.read(filename)
    # data.prepare(anomalies=12, normalize=True, sph_degree=9, tp_period=12)
    # data.prepare(anomalies=12, anomalies_range=['1980', '2000'])
    # data.summary()
