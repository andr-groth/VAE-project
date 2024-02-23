"""Collection of file readers.

"""

import glob
import json
import os
import warnings
from pprint import pprint

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tensorflow.keras.utils import Progbar

from VAE.utils import plot as vplt


def _check_arg(filename, arg):

    # convert arg to list of same length as filename, if:
    # 1. arg is not a list
    if not isinstance(arg, (tuple, list)):
        arg = [arg] * len(filename)
    # 2. arg is not list of lists
    elif not all(isinstance(el, (tuple, list)) for el in arg):
        arg = [arg] * len(filename)

    if len(arg) != len(filename):
        raise ValueError(f'Length of `{arg}` must match the length of `{filename}`.')
    return arg


def _get_climexp_model_names():
    """Get models names from Climate Explorer."""

    url = 'http://climexp.knmi.nl/selectmember.cgi?i=%d&field=cmip5_tos_Omon_one_rcp45'
    values = range(0, 42)
    div = "container-fluid content"
    line = 4

    for value in values:
        _url = url % value
        with requests.get(_url) as r:
            soup = BeautifulSoup(r.content, "html.parser")
            lines = soup.find("div", {"class": div}).stripped_strings
            content = list(lines)[line]
            # print(value, content)
            print(content)


def read_netcdf(filename: str,
                level_range: tuple[int, int] = None,
                time_interval: tuple[str, str] = None,
                time_range: tuple[int, int] = None,
                num2date: bool = False,
                datetime_unit: str = 'D',
                scale: float = 1.,
                dtype: str = None) -> tuple[dict, dict, dict]:
    """Read netCDF file.

    The function follows the CF-1 convention and assumes data of the form:
        2D: `(time, level)`
        3D: `(time, lat, lon)`
        4D: `(time, level, lat, lon)`

    Parameters:
        filename:
            Name of the netCDF file.
        level_range:
            Start and stop indices of levels to be read (in slice notation). Defaults to `None`.
        time_interval:
            Start and stop time to be read (in datetime notation). Default is 'None'.
        time_range:
            Start and stop indices along time dimensions to be read (in slice notation). Default is 'None'.
        num2date:
            Convert time to Numpy datetime. Defaults to 'True' if 'time_interval' is given otherwise default is 'False'.
        datetime_unit:
            Unit of datetime if `num2date` is set to `True`. Defaults to `D`, meaning day. For a list of units see
            https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units.
        scale:
            Scale all variables by this factor. Defaults to 1.
        dtype:
            Data type of the variables that will be returned. Default is None, meaning dtype of netCDF file.

    Returns:
        tuple of three dicts containing the variables, dimensions, and attributes.
    """

    variables = {}
    dimensions = {}
    attributes = {}

    with netCDF4.Dataset(filename) as dataset:
        # read global attributes
        attributes['.'] = {attr: dataset.getncattr(attr) for attr in dataset.ncattrs()}

        # read variables
        for var_name, variable in dataset.variables.items():
            if variable.ndim in (2, 3, 4):
                # extract dimensions
                var_dims = [dataset.variables.get(name) for name in variable.dimensions]

                if None not in var_dims:
                    # extract variable
                    variables[var_name] = variable[:].filled(fill_value=np.nan)
                    if dtype is not None:
                        variables[var_name] = variables[var_name].astype(dtype)

                    attributes[var_name] = {attr: variable.getncattr(attr) for attr in variable.ncattrs()}

                    # scale variable
                    variables[var_name] *= scale
                    attributes[var_name]['scale'] = scale

                    # follow CF-1 convention
                    if variable.ndim == 2:
                        dim_keys = ('time', 'level')
                    elif variable.ndim == 3:
                        dim_keys = ('time', 'lat', 'lon')
                    elif variable.ndim == 4:
                        dim_keys = ('time', 'level', 'lat', 'lon')

                    for dim_name, var_dim in zip(dim_keys, var_dims):
                        dimensions[dim_name] = var_dim[:].filled(fill_value=np.nan)
                        attributes[dim_name] = {
                            attr: var_dim.getncattr(attr)
                            for attr in var_dim.ncattrs() if attr not in {'bounds'}
                        }

        # replace time with Numpy datetime
        if num2date | (time_interval is not None):
            datetime = netCDF4.num2date(dimensions['time'],
                                        units=attributes['time']['units'],
                                        calendar=attributes['time']['calendar'])
            datetime = [np.datetime64(d.strftime('%Y-%m-%d %H:%M:%S')) for d in datetime]
            datetime = np.array(datetime)
            datetime = datetime.astype(f'datetime64[{datetime_unit}]')
            dimensions['time'] = datetime

        # restrict time interval
        if time_interval is not None:
            ti0, ti1 = time_interval
            idx = (np.datetime64(ti0) <= dimensions['time']) & (dimensions['time'] <= np.datetime64(ti1))
            dimensions['time'] = dimensions['time'][idx]
            variables = {key: val[idx, ...] for key, val in variables.items()}

        # restrict time range
        if time_range is not None:
            idx = slice(*time_range)
            dimensions['time'] = dimensions['time'][idx]
            variables = {key: val[idx, ...] for key, val in variables.items()}

        # restrict level range
        if level_range is not None:
            idx = slice(*level_range)
            if 'level' in dimensions:
                dimensions['level'] = dimensions['level'][idx]
                variables = {key: val[:, idx, ...] for key, val in variables.items() if val.ndim in (2, 4)}

    return variables, dimensions, attributes


def write_netcdf(filename: str,
                 variables: dict[str, np.ndarray],
                 dimensions: dict[str, np.ndarray],
                 attributes: dict[str, dict],
                 scale: float = None,
                 dtype: str = 'f4',
                 compression: str = 'zlib',
                 verbose: bool = True):
    """Write netCDF file.

    The function follows the CF-1 convention and assumes data of the form:
        2D: `(time, level)`
        3D: `(time, lat, lon)`
        4D: `(time, level, lat, lon)`

    The structure of the dictionaries `variables`, `dimensions`, and `attributes` follows that of :func:`read_netcdf`.

    Parameters:
        filename:
            Name of the netCDF file.
        variables:
            Dictionary of variables with the items defining the variable name and values.
        dimensions:
            Dictionary of dimensions with the items defining the dimension name and values. Dimension names must be
            either `time`, `level`, `lat`, or `lon`.
        attributes:
            Dictionary of attributes for the variables and the dimensions with the items defining the variable/dimension
            name and a dict of attributes. Global attributes will be taken from `attributes['.']`.
        scale:
            Scale all variables. If scale is `None`, scale is obtained from `attributes[variable_name]['scale']`.
        dtype:
            Data type of the variables that will be written. Default is 'float'.
        compression:
            Data will be compressed in the netCDF file using the specified compression algorithm.

    """

    valid_dims = {'time', 'level', 'lat', 'lon'}
    if set(dimensions.keys()) > valid_dims:
        raise KeyError(f'Dimension keys must be subset of {valid_dims}')

    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as dataset:
        if verbose:
            print('Write:', os.path.normpath(filename))

        # write dimensions
        for name, value in dimensions.items():
            dataset.createDimension(name, len(value))
            variable = dataset.createVariable(name, dtype, (name, ))
            atts = {k: v for k, v in attributes[name].items() if k not in {'bounds'}}
            variable.setncatts(atts)

            if name == 'time':
                value = netCDF4.date2num([pd.to_datetime(v) for v in value],
                                         units=attributes['time']['units'],
                                         calendar=attributes['time']['calendar'])

            variable[:] = value

        # write variables
        for name, value in variables.items():
            atts = attributes[name].copy()
            if scale is None:
                scale = 1 / atts.pop('scale', 1.)

            value *= scale

            # follow CF-1 convention
            if value.ndim == 2:
                dims = ('time', 'level')
            elif value.ndim == 3:
                dims = ('time', 'lat', 'lon')
            elif value.ndim == 4:
                dims = ('time', 'level', 'lat', 'lon')

            variable = dataset.createVariable(name,
                                              dtype,
                                              dimensions=dims,
                                              compression=compression,
                                              fill_value=atts.pop('_FillValue', None))
            variable.setncatts(atts)
            variable[:] = value

        # write global attributes
        dataset.setncatts(attributes.get('.', {}))


def read_netcdf_multi(filename: list[str],
                      level_range: tuple[int, int] = None,
                      time_interval: tuple[str, str] = None,
                      time_range: tuple[int, int] = None,
                      scale: float = 1.,
                      recursive: bool = False,
                      num2date: bool = False,
                      datetime_unit: str = 'D',
                      dtype: str = None,
                      verbose: bool = True) -> tuple[dict, dict, dict]:
    """Read multiple files of netCDF data.

    Parameters:
        filename:
            Name of file(s). Glob patterns can be used.
        level_range:
            Start and stop indices of levels to be read (in slice notation). If a list of tuples, length of list must
            match the length of `filename`. Defaults to `None`.
        time_interval:
            Time intervaL to be read. If a list of tuples, length of list must match the length of `filename`. Defaults
            to `None`.
        time_range:
            Start and stop indices along time dimensions to be read (in slice notation). If a list of tuples, length of
            list must match the length of `filename`. Defaults to `None`.
        recursive:
            If recursive is true, the pattern '**' will match any files and zero or more directories and subdirectories.
        num2date:
            Convert time to Numpy datetime. Defaults to 'True' if 'time_interval' is given otherwise default is 'False'.
        datetime_unit:
            Unit of datetime if `num2date` is set to `True`. Defaults to `D`, meaning day. For a list of units see
            https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-units.
        dtype:
            Data type of the variables that will be returned. Default is None, meaning dtype of netCDF file.

    Returns:
        tuple of three dicts containing the variables, dimensions, and attributes of each file.

    """

    if isinstance(filename, str):
        filename = [filename]

    filename = [os.path.normpath(fn) for fn in filename]
    time_interval = _check_arg(filename, time_interval)
    time_range = _check_arg(filename, time_range)
    level_range = _check_arg(filename, level_range)
    scale = np.broadcast_to(scale, len(filename)).tolist()

    filenames = []
    time_intervals = []
    time_ranges = []
    level_ranges = []
    scales = []
    ml = max(len(fn) for fn in filename)
    for fn, ti, tr, lr, sc in zip(filename, time_interval, time_range, level_range, scale):
        new_filenames = sorted(glob.glob(fn, recursive=recursive))
        if new_filenames:
            filenames += new_filenames
            time_intervals += [ti] * len(new_filenames)
            time_ranges += [tr] * len(new_filenames)
            level_ranges += [lr] * len(new_filenames)
            scales += [sc] * len(new_filenames)

            if verbose:
                print(f'{fn:{ml}.{ml}} : {len(new_filenames)} file(s) found.')
        else:
            raise FileNotFoundError(f'No files found for: {fn}')

    if not filenames:
        raise FileNotFoundError('No files found.')

    variables = {}
    dimensions = {}
    attributes = {}
    # commonpath = os.path.commonpath(filenames)

    pbar = Progbar(len(filenames), verbose=verbose, unit_name='file')
    for fn, ti, tr, lr, sc in zip(filenames, time_intervals, time_ranges, level_ranges, scales):
        try:
            variable, dimension, attribute = read_netcdf(fn,
                                                         level_range=lr,
                                                         time_interval=ti,
                                                         time_range=tr,
                                                         scale=sc,
                                                         num2date=num2date,
                                                         datetime_unit=datetime_unit,
                                                         dtype=dtype)
        except Exception as error:
            print('Error reading', os.path.normpath(fn), ':', error)
            raise

        # key = os.path.relpath(fn, commonpath)
        # key, _ = os.path.splitext(key)
        key = os.path.normpath(fn)
        variables[key] = variable
        dimensions[key] = dimension
        attributes[key] = attribute
        pbar.add(1)

    return variables, dimensions, attributes


def read_climexp_raw_data(filename: str,
                          ensemble_members: list[int] = None,
                          time_interval: tuple[str, str] = None,
                          dtype: str = 'float') -> tuple[pd.DataFrame, dict]:
    """Read single file of raw data from climexp.

    Read a single file of raw data downloaded from https://climexp.knmi.nl.

    Parameters:
        filename:
            File name.
        ensemble_members:
            Ensemble members that will be returned. Defaults to `None`, meaning all members are returned.
        time_interval:
            Time interval to be read.
        dtype:
            Data type of the returned DataFrame. Default is 'float'.

    Returns:
        tuple of DataFrame and dict, where the dataframe contains the data and the dict contains the metadata.
    """
    # split file into datasets
    with open(filename) as file:
        datasets = file.read().split('# ensemble member')

    # split datasets into list of lines
    datasets = [dataset.splitlines() for dataset in datasets]

    if len(datasets) == 1:
        # single dataset
        comments = [line for line in datasets[0] if line.startswith('#')]
        data = [line for line in datasets[0] if not line.startswith('#')]
        datasets = [comments, ['0'] + data]

    # remove whitespace and comment symbol from each line
    datasets = [[line.strip('# ') for line in dataset] for dataset in datasets]

    # remove empty lines, inplace
    datasets[:] = [[line for line in dataset if line] for dataset in datasets]

    # extract metadata
    metadata = datasets.pop(0)

    # convert header into dict
    metadata = [line.partition('::') for line in metadata]
    variable_name = metadata.pop()[0]
    metadata = {key.strip(): val.strip() for key, sep, val in metadata if sep}
    metadata['variable'] = variable_name

    # first line contains ensemble number
    datasets_dict = {dataset[0]: [line.split() for line in dataset[1:]] for dataset in datasets if len(dataset) > 1}

    # filter ensemble members
    if ensemble_members is not None:
        ensemble_members = set(ensemble_members)
        datasets_dict = {key: val for key, val in datasets_dict.items() if int(key) in ensemble_members}

    # convert list of datasets into list of dataframes
    dataframes = []
    for key, dataset in datasets_dict.items():
        dataframe = pd.DataFrame(dataset, columns=['time', key], dtype=dtype)
        dataframe.set_index('time', inplace=True)
        dataframes.append(dataframe)

    # concatenate dataframes in one dataframe
    df = pd.concat(dataframes, axis=1)

    # convert time into datetime
    year = df.index.astype(np.int32)
    month = np.rint((df.index % 1) * 12 + 1).astype(np.int32)
    date = pd.to_datetime({'year': year, 'month': month, 'day': 1}, unit='D')
    df.set_index(date, inplace=True)

    if time_interval is not None:
        df = df.loc[time_interval[0]:time_interval[1]]

    return df, metadata


def read_climexp_raw_data_multi(filename: list[str],
                                ensemble_members: list[int] = None,
                                time_interval: tuple[str, str] = None,
                                recursive: bool = False,
                                join: str = 'inner',
                                dtype: str = 'float') -> tuple[pd.DataFrame, dict[str, dict]]:
    """Read multiple files of raw data from climexp.

    Read multiples files of raw data downloaded from https://climexp.knmi.nl.

    Parameters:
        filename:
            Name of file(s). Glob patterns can be used.
        ensemble_members:
            Ensemble members that will be returned. Defaults to `None`, meaning all members are returned.
        time_interval:
            Time interval to be read. If a list, it must match the length of `filename`.
        recursive:
            If recursive is true, the pattern '**' will match any files and zero or more directories and subdirectories.
        join:
            How to join the different files. 'inner' means that only the common rows will be kept. 'outer' means that
            all rows will be kept and missing values will be filled with NaN. Default is 'inner'.
        dtype:
            Data type of the returned DataFrame. Default is 'float'.

    Returns:
        tuple of DataFrame and dict of dict. The dataframe contains the data and the dict contains the metadata. In the
        DataFrame, the filename is used as level-zero index in a multi-index. In the dict, the filename is used as key.
    """
    if isinstance(filename, str):
        filename = [filename]

    time_interval = _check_arg(filename, time_interval)

    filenames = []
    time_ranges = []
    for file, tr in zip(filename, time_interval):
        new_filenames = sorted(glob.glob(file, recursive=recursive))
        if new_filenames:
            filenames += new_filenames
            time_ranges += [tr] * len(new_filenames)
        else:
            warnings.warn(f'No files found for pattern: {os.path.normpath(file)}')

    if not filenames:
        raise ValueError('No files found.')

    commonpath = os.path.commonpath(filenames)
    df = {}
    metadata = {}
    for name, tr in zip(filenames, time_ranges):
        try:
            new_df, new_metadata = read_climexp_raw_data(name,
                                                         ensemble_members=ensemble_members,
                                                         time_interval=tr,
                                                         dtype=dtype)
        except ValueError:
            raise ValueError('Error reading file {}.'.format(name))

        key = os.path.relpath(name, commonpath)
        key, _ = os.path.splitext(key)
        df[key] = new_df
        metadata[key] = new_metadata

    df = pd.concat(df, axis=1, join=join)

    return df, metadata


def _read_iri_enso_plume(filename: str,
                         join: str = 'outer',
                         model_type: str = None,
                         time_range: tuple[str, str] = None,
                         dtype: str = 'float') -> pd.DataFrame:
    """Read ENSO forecast plume from IRI.

    Read ENSO forecast plume in json format downloaded from https://iri.columbia.edu/~forecast/ensofcst/Data.

    Parameters:
        filename:
            Name of json file.
        join :
            How to join the different models. `inner` means that only the common years will be kept. `outer` means that
            all years will be kept and missing values will be filled with NaN. Default is `outer`.
        model_type:
            Restrict returned dataframe to given model type. Possible values are 'dynamical', 'statistical' and
            'combined'. Default is None, meaning no restriction.
        time_range: tuple of two str
            Time range to be read.
        dtype:
            Data type of the returned DataFrame. Default is 'float'.

    Returns:
        DataFrame with multi-index. The level-zero index refers to the model and the level-one index to the lead time.
    """
    try:
        with open(filename, "rb") as f:
            data = json.load(f)
    except IOError:
        print(f'error reading {filename}')

    model_type = str(model_type or '').lower()

    dataset = dict()
    for year in data['years']:
        for month in year['months']:
            for model in month['models']:
                if model_type in str(model['type'] or '').lower():
                    datestr = f"{year['year']}-{month['month'] + 1:02d}-01"
                    newentry = {datestr: model['data']}
                    dataset.setdefault(model['model'], {}).update(newentry)

    if not dataset:
        raise IOError("No entries found.")

    dfs = {key: pd.DataFrame.from_dict(val, orient='index', dtype=dtype) for key, val in dataset.items()}
    df = pd.concat(dfs, axis=1, join=join)
    df.replace(-999, np.nan, inplace=True)
    df.set_index(pd.to_datetime(df.index), inplace=True)
    df.sort_index(inplace=True)

    if time_range is not None:
        df = df.loc[time_range[0]:time_range[1]]

    return df


def example_read_climexp_raw_data_multi():
    """Example of how to use the function `read_climexp_raw_data_multi`.

    The function reads multiple files of raw data from the `example_data/` folder.

    ```
    example_data/icmip5_tos_Omon_one_rcp45_pc01.txt
    example_data/icmip5_tos_Omon_one_rcp45_pc02.txt
    ```

    """

    filename = [
        'example_data/icmip5_tos_Omon_one_rcp45_pc01.txt',
        'example_data/icmip5_tos_Omon_one_rcp45_pc02.txt',
    ]

    # read data
    df, metadata = read_climexp_raw_data_multi(filename, ensemble_members=[0, 1, 2, 3, 4, 5], join='outer')

    with pd.option_context('display.precision', 3):
        print(df)

    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    time = df.index.to_numpy()

    # access data by level-zero index
    for idx in df.columns.levels[0]:
        x = df[idx]
        x = x.to_numpy()
        x_mean = x.mean(axis=1)
        x_std = x.std(axis=1) * 3

        ax.plot(time, x_mean, label=metadata[idx].get('description'), zorder=2.2)
        ax.fill_between(time, x_mean - x_std, x_mean + x_std, alpha=0.5, zorder=2.1)

    ax.legend(loc='upper left')
    ax.grid(linestyle=':')


def example1_read_netcdf():
    """Example of how to use the function `read_netcdf`.

    This function reads EOFs in a netCDF file from the Climate explorer from the `example_data/` folder:

    ```
    example_data/eofs_icmip5_tos_Omon_one_rcp45.nc
    ```

    """

    filename = 'example_data/eofs_icmip5_tos_Omon_one_rcp45.nc'

    # read data
    variables, dimensions, attributes = read_netcdf(filename)

    print('variables')
    for key, value in variables.items():
        print('  ', key, value.shape)

    print('dimensions')
    for key, value in dimensions.items():
        print('  ', key, value.shape)

    print('attributes')
    pprint(attributes)

    # remove singleton dimensions
    squeeze_variables = {key: np.squeeze(value) for key, value in variables.items()}
    # plot variables
    vplt.map_plot(dimensions['lat'], dimensions['lon'], squeeze_variables, ncols=5, figwidth=20, cmap='seismic')


def example2_read_netcdf():
    """Example of how to use the function `read_netcdf`.

    This function reads EOFs in a netCDF file from the output of the climate data operators (CDO). The file is from the
    `example_data/` folder:

    ```
    example_data/eofs_anom_gpcc_v2020_1dgr.nc
    ```

    :material-github: For the calculation of the EOFs and PCs with CDO see the [CDO
        scripts](https://github.com/andr-groth/cdo-scripts).

    """

    filename = 'example_data/eofs_anom_gpcc_v2020_1dgr.nc'

    # read data
    variables, dimensions, attributes = read_netcdf(filename)

    print('variables')
    for key, value in variables.items():
        print('  ', key, value.shape)

    print('dimensions')
    for key, value in dimensions.items():
        print('  ', key, value.shape)

    print('attributes')
    pprint(attributes)

    # plot first variable
    key, *_ = list(variables)
    variable = variables[key]
    vplt.map_plot(dimensions['lat'], dimensions['lon'], variable, ncols=10, figwidth=20, cmap='seismic')


def example3_read_netcdf():
    """Example of how to use the function `read_netcdf`.

    This function reads PCs in a netCDF file from the output of the climate data operators (CDO). The file is from the
    `example_data/` folder:

    ```
    example_data/pcs_anom_gpcc_v2020_1dgr.nc
    ```

    :material-github: For the calculation of the EOFs and PCs with CDO see the [CDO
        scripts](https://github.com/andr-groth/cdo-scripts).

    """

    filename = 'example_data/pcs_anom_gpcc_v2020_1dgr.nc'

    # read data
    variables, dimensions, attributes = read_netcdf(filename, num2date=True)

    print('variables')
    for key, value in variables.items():
        print('  ', key, value.shape)

    print('dimensions')
    for key, value in dimensions.items():
        print('  ', key, value.shape)

    print('attribtutes')
    pprint(attributes)

    # plot first variable
    key, *_ = list(variables)
    variable = np.squeeze(variables[key])  # remove singleton spatial dimensions
    variable = variable.T
    variable = np.atleast_2d(variable)
    cols = min(len(variable), 5)
    rows = -(-len(variable) // cols)
    _, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2 * rows), sharex=True, sharey=True, squeeze=False)
    for n, (ax, value) in enumerate(zip(axs.flatten(), variable)):
        ax.plot(dimensions['time'], value)
        ax.set_title(n)


def example_read_netcdf_multi(filename: str):
    """Example of how to use the function `read_netcdf_multi`.

    This function reads PCs in multiple netCDF files from the output of the climate data operators (CDO). The files are
    from the `example_data/` folder:

    ```
    example_data/pcs_anom_pr_*.nc
    ```

    :material-github: For the calculation of the EOFs and PCs with CDO see the [CDO
        scripts](https://github.com/andr-groth/cdo-scripts).

    """

    filename = 'example_data/pcs_anom_pr_*.nc'

    # read data
    variables, dimensions, attributes = read_netcdf_multi(filename, num2date=True)

    print('variables')
    for key, value in variables.items():
        print('  ', key)
        for k, v in value.items():
            print('    ', k, v.shape)

    print('dimensions')
    for key, value in dimensions.items():
        print('  ', key)
        for k, v in value.items():
            print('    ', k, v.shape)

    rows = 5
    cols = 2
    _, axs = plt.subplots(rows, cols, figsize=(8 * cols, 3 * rows), sharex=True, sharey=True, squeeze=False)
    # cycle through different files
    for key, variable in variables.items():
        var_name, *_ = list(variable)  # extract first variable
        values = np.squeeze(variable[var_name])  # remove singleton spatial dimensions
        values = values.T
        values = np.atleast_2d(values)
        for n, (ax, value) in enumerate(zip(axs.flatten(), values)):
            ax.plot(dimensions[key]['time'], value, label=key)
            ax.set_title(n)

    axs.flat[0].legend()


def _example_read_iri_enso_plume(filename):
    """Example of how to use the function `read_iri_enso_plume`.

    This function reads the ENSO plume data from the IRI. The file is from the `example_data/` folder:

    ```
    example_data/enso_plumes.json
    ```

    Data provided by The International Research Institute for Climate and Society, Columbia University Climate School,
    at [iri.columbia.edu/ENSO](https://iri.columbia.edu/ENSO)

    """

    filename = 'example_data/_enso_plumes.json'

    df = _read_iri_enso_plume(filename)
    names = df.columns.levels[0]
    fig, axs = plt.subplots(len(names),
                            1,
                            figsize=(10, len(names) * 0.5),
                            sharex=True,
                            sharey=True,
                            gridspec_kw={'hspace': 0})

    for ax, name in zip(axs.flat, names):
        data = np.stack([np.roll(x, lead) for lead, x in enumerate(df[name].to_numpy().T)])
        data_valid = np.mean(np.isfinite(data)) * 100
        mp = ax.pcolormesh(df.index, df[name].columns, data, cmap='RdYlBu_r', vmin=-1, vmax=1, shading='nearest')
        ax.set_ylabel(f'{name} ({data_valid:.0f})', rotation=0, ha='right', va='center')

    fig.colorbar(mp, ax=axs, shrink=0.3, extend='both')
