"""Collection of file readers.

@author: Andreas Groth
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
from matplotlib import ticker


def get_climexp_model_names(url, values=(0, ), div="container-fluid content", line=4):
    """Get models names from Climate Explorer.

    Parameters:
        url : str
            URL to the ensemble member with the index argument indicated by a `%d` specifier. For example
            `'http://climexp.knmi.nl/selectmember.cgi?i=%d&field=cmip5_tos_Omon_one_rcp45'`.
        values : iterable, optional
            Values of the index. Defaults to (0, ).
        div : str, optional
            Class name of the div container. Defaults to "container-fluid content".
        line : int, optional
            Line number in the div container that contains the model name. Defaults to 4.
    """
    for value in values:
        _url = url % value
        with requests.get(_url) as r:
            soup = BeautifulSoup(r.content, "html.parser")
            lines = soup.find("div", {"class": div}).stripped_strings
            content = list(lines)[line]
            print(value, content)


def read_netcdf(filename: str, dtype='float'):
    """Read netCDF file.

    Parameters:
        filename : str
            Name of the file.
        dtype : str
            Data type of the variables. Default is 'float'.

    Returns:
        tuple of three dicts containing the variables, dimensions, and attributes.
    """

    variables = dict()
    dimensions = dict()
    attributes = dict()

    with netCDF4.Dataset(filename) as dataset:
        for var_key, variable in dataset.variables.items():
            # read all 3D, assume assume  CF-1 convention (time, lat, lon)
            # read all 4D, assume assume  CF-1 convention (time, level, lat, lon)
            if variable.ndim in (3, 4):
                # extract variable
                variables[var_key] = variable[:].filled(fill_value=np.nan).astype(dtype)
                attributes[var_key] = {attr: getattr(variable, attr) for attr in variable.ncattrs()}

                # extract dimensions
                var_dims = [dataset.variables[name] for name in variable.dimensions]

                if variable.ndim == 3:
                    dim_keys = ('time', 'lat', 'lon')
                else:
                    dim_keys = ('time', 'level', 'lat', 'lon')

                for dim_key, var_dim in zip(dim_keys, var_dims):
                    dimensions[dim_key] = var_dim[:].filled(fill_value=np.nan)
                    attributes[dim_key] = {attr: getattr(var_dim, attr) for attr in var_dim.ncattrs()}

    return variables, dimensions, attributes


def read_climexp_raw_data(filename: str,
                          ensemble_members: list[int] = None,
                          time_range: tuple[str, str] = None,
                          dtype='float'):
    """Read single file of raw data from climexp.

    Read a single file of raw data downloaded from https://climexp.knmi.nl.

    Parameters:
        filename : str
            File name.
        ensemble_members: list of int
            Ensemble members that will be returned. Defaults to `None`, meaning all members are returned.
        time_range : tuple of two str
            Time range to be read.
        dtype : str
            Data type of the returned DataFrame. Default is 'float'.

    Returns:
        tuple of DataFrame and dict.
            The dataframe contains the data and the dict contains the metadata.
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

    if time_range is not None:
        df = df.loc[time_range[0]:time_range[1]]

    return df, metadata


def read_climexp_raw_data_multi(filename: list[str],
                                ensemble_members: list[int] = None,
                                time_range: tuple[str, str] = None,
                                recursive=False,
                                join='inner',
                                dtype='float'):
    """Read multiple files of raw data from climexp.

    Read multiples files of raw data downloaded from https://climexp.knmi.nl.

    Parameters:
        filename : list of str
            Name of file(s). Glob patterns can be used.
        ensemble_members: list of int
            Ensemble members that will be returned. Defaults to `None`, meaning all members are returned.
        time_range : tuple of two str or list of tuples of two str
            Time range to be read. If a list, it must match the length of `files`.
        recursive : bool
            If recursive is true, the pattern '**' will match any files and zero or more directories and subdirectories.
        join : {'inner', 'outer'}
            How to join the different files. 'inner' means that only the common rows will be kept. 'outer' means that
            all rows will be kept and missing values will be filled with NaN. Default is 'inner'.
        dtype : str
            Data type of the returned DataFrame. Default is 'float'.

    Returns:
        tuple of DataFrame and dict of dict
            The dataframe contains the data and the dict contains the metadata. In the DataFrame, the filename is used
            as level-zero index in a multi-index. In the dict, the filename is used as key.
    """
    if isinstance(filename, str):
        filename = [filename]

    if time_range is None:
        time_range = [time_range] * len(filename)
    else:
        if all(isinstance(el, (list, tuple)) for el in time_range):
            lengths = {len(el) for el in time_range}
            if len(lengths) > 1:
                raise ValueError('time_range must be a list of tuples of length 2')
            if len(time_range) != len(filename):
                raise ValueError('list of time_range must match the length of the list of files')
        else:
            time_range = [time_range] * len(filename)

    filenames = []
    time_ranges = []
    for file, tr in zip(filename, time_range):
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
                                                         time_range=tr,
                                                         dtype=dtype)
        except ValueError:
            raise ValueError('Error reading file {}.'.format(name))

        key = os.path.relpath(name, commonpath)
        key = os.path.splitext(key)[0]
        df[key] = new_df
        metadata[key] = new_metadata

    df = pd.concat(df, axis=1, join=join)

    return df, metadata


def read_iri_enso_plume(filename, join='outer', model_type=None, time_range: tuple[str, str] = None, dtype='float'):
    """Read ENSO forecast plume from IRI.

    Read ENSO forecast plume in json format downloaded from https://iri.columbia.edu/~forecast/ensofcst/Data.

    Parameters:
        filename: str
            Name of json file.
        join : {'inner', 'outer'}
            How to join the different models. `inner` means that only the common years will be kept. `outer` means that
            all years will be kept and missing values will be filled with NaN. Default is `outer`.
        model_type : {None, 'Dynamical', 'Statistical', 'Other'}
            Restrict returned dataframe to given model type. Default is None, meaning no restriction.
        time_range : tuple of two str
            Time range to be read.
        dtype: str
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


def example_read_climexp_raw_data_multi(root_dir):
    """Example of how to use the function `read_climexp_raw_data_multi`."""
    filename = [
        'rcp45/nino34/*one*._txt',
        'rcp45/pcs_55S60N_5dgr_1865-2005/*one*.txt',
    ]
    ensemble_members = [0, 1, 2, 3, 4, 5]
    time_range = ['1865', '2005']
    filename = [os.path.join(root_dir, file) for file in filename]

    # read data
    df, metadata = read_climexp_raw_data_multi(filename,
                                               ensemble_members=ensemble_members,
                                               time_range=time_range,
                                               join='outer')

    print(df.head())

    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    time = df.index.to_numpy()
    # access data by level-zero index
    for idx in df.columns.levels[0]:
        x = df[idx]
        x = x.rolling(window=3, center=True).mean()
        x = x.to_numpy()
        x_mean = x.mean(axis=1)
        x_std = x.std(axis=1) * 3

        ax.plot(time, x_mean, label=metadata[idx].get('description'), zorder=2.2)
        ax.fill_between(time, x_mean - x_std, x_mean + x_std, alpha=0.5, zorder=2.1)

    ax.legend(loc='upper left')
    ax.grid(linestyle=':')


def example_read_netcdf(root_dir):
    """Example of how to use the function `read_netcdf`."""
    filename = 'rcp45/pcs_55S60N/icmip5_tos_Omon_one_rcp45_eofs.nc'
    filename = os.path.join(root_dir, filename)

    # read data
    variables, dimensions, attributes = read_netcdf(filename)

    for key, value in variables.items():
        print(key, value.shape)

    for key, value in dimensions.items():
        print(key, value.shape)

    pprint(attributes)

    _, axs = plt.subplots(4, 5, figsize=(30, 8), sharex=True, sharey=True)
    for ax, (key, value) in zip(axs.flatten(), variables.items()):
        ax.pcolormesh(dimensions['lon'],
                      dimensions['lat'],
                      np.squeeze(value),
                      shading='nearest',
                      vmin=-0.5,
                      vmax=0.5,
                      cmap='coolwarm')
        ax.set_title(key)
        ax.set_aspect('equal')


def example_read_iri_enso_plume(root_dir):
    """Example of how to use the function `read_iri_enso_plume`."""
    filename = 'enso_plumes.json'
    filename = os.path.join(root_dir, filename)
    df = read_iri_enso_plume(filename)
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
    return df


if __name__ == '__main__':
    root_dir = os.path.expanduser('~/GoogleDrive/Andreas/myPython/Tensorflow/VAE-ENSO/data')
    # example_read_climexp_raw_data_multi(root_dir)
    # example_read_netcdf(root_dir)
    # df = example_read_iri_enso_plume(root_dir)
    get_climexp_model_names(url='http://climexp.knmi.nl/selectmember.cgi?i=%d&field=cmip5_tos_Omon_one_rcp45',
                            values=range(0, 40))
