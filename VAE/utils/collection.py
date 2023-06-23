# -*- coding: utf-8 -*-
"""Collection of helper functions.

"""

import fnmatch
import glob
import os
from typing import List, Union

import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K
import yaml


def complete_shape(inputs: tf.Tensor, partial_shape: tuple[int, ...]) -> tf.Tensor:
    """Complete partially-known shape based on inputs shape.

    Input shape of `inputs` must be of the same length as `partial_shape`. `None` values in `partial_shape` will be
    replaced with the corresponding dimension of `inputs`. The resulting shape is returned as a tensor of the same
    length as the shape of `inputs`.

    Parameters:
        inputs:
            Input tensor that is used to complete shape.
        partial_shape:
            Partial shape to complete.

    Returns:
        Tensor with completed shape.
    """
    inputs_shape = tf.shape(inputs)

    if partial_shape is None:
        return inputs_shape

    complete_shape = []
    for n, value in enumerate(partial_shape):
        complete_shape.append(inputs_shape[n] if value is None else value)

    return tf.convert_to_tensor(complete_shape)


def summary_trainable(model: ks.Model, line_length: int = 80):
    """Print model summary of trainable parameters.

    This function prints a summary of trainable layers of a model.

    Parameters:
        model:
            Model to be summarized.
        line_length:
            Total length of each printed line.

    """
    def recursion(layer: ks.layers.Layer, level=0):
        trainable_weights = sum(K.count_params(p) for p in layer.trainable_weights)

        if trainable_weights:
            if level == 0:
                print('=' * line_length)
            elif level == 1:
                print('_' * line_length)

            layer_name = ' ' * level * 2 + layer.name
            layer_name = f'{layer_name:{c1}.{c1}s}'
            layer_type = f'{layer.__class__.__name__:{c2}.{c2}s}'
            layer_params = f'{trainable_weights:>{c3},d}'
            print(f'{layer_name} {layer_type} {layer_params}')

        for children in layer._flatten_layers(recursive=False, include_self=False):
            recursion(children, level + 1)

    c1 = line_length // 2
    c2 = line_length // 4
    c3 = line_length - c1 - c2 - 2

    print('_' * line_length)
    print(f'{"Layer":{c1}} {"Type":{c2}} {"# params":>{c3}}')
    recursion(model)
    print('_' * line_length)


def set_trainable(model: ks.Model, trainable: Union[str, List[str]], verbose: bool = False):
    """Set trainable layers of a Keras model.

    This function sets the trainable property of layers of a Keras model. The layers to be set trainable can be
    defined by their name. Unix shell-style wildcards can be used. If a layer is set to trainable, all its children
    will be set to trainable as well.

    Parameters:
        model:
            Model to be modified.
        trainable:
            Layer names to be set to trainable. Unix shell-style wildcards can be used.
        verbose:
            Print the names of the layers that are set to trainable if `True`.

    """
    def recursion(layer: ks.layers.Layer, layer_names: set):
        sublayers = list(layer._flatten_layers(recursive=False, include_self=False))
        if sublayers:
            for sublayer in sublayers:
                recursion(sublayer, layer_names)
        else:
            if layer.name not in layer_names:
                layer.trainable = False

    if isinstance(trainable, str):
        trainable = [trainable]

    model_layer_names = [layer.name for layer in model._flatten_layers()]
    trainable_layers = []
    for pattern in trainable:
        trainable_layers.extend(fnmatch.filter(model_layer_names, pattern))

    trainable_layers = set(trainable_layers)

    if verbose:
        names = '\n  '.join(sorted(trainable_layers))
        print(f'Setting trainable layers:\n  {names}')

    for layer in model._flatten_layers():
        layer.trainable = True

    recursion(model, trainable_layers)


def SubModel(model: ks.Model, layer_name: Union[str, list[str]], flatten: bool = False) -> ks.Model:
    """Get a submodel of a Keras model.

    This function returns a submodel of a Keras model. The submodel takes the same input as the original model, and
    covers all layers of the original model between the input layer and the layer(s) with the specified name.

    Parameters:
        model:
            Model to be submodeled.
        layer_name:
            If str, it specifies the name of the layer to be used as output. In this case, `layer_name` can be a
            substring of the layer name. If list of str, it specifies the names of the layers to be used as outputs.
        flatten:
            If True, the outputs of the submodel are flattened.
    Returns:
        Instance of :class:`keras.Model`.

    """
    if isinstance(layer_name, str):
        outputs = [layer.output for layer in model._flatten_layers() if layer_name in layer.name]
    else:
        outputs = [layer.output for layer in model._flatten_layers() if layer.name in layer_name]

    if not outputs:
        raise ValueError(f'No layers with name matching "{layer_name}" found.')

    if flatten:
        outputs = [ks.layers.Flatten()(output) for output in outputs]
    return ks.Model(model.inputs, outputs=outputs)


class TrainerConfigCollection:
    def __init__(self, path: str = '.', filemask: str = '*.yaml', recursive: bool = True, verbose: bool = True):
        """Collect the training configurations for a given path.

        Parameters:
            path:
                Path to the training configuration files. Defaults to `.`.
            filemask:
                Filename or filemask of the training configuration files. Defaults to `*.yaml`.
            recursive:
                Recursive search in subdirectories. Defaults to True.
            verbose:
                Verbose output. Defaults to True.
        """
        self.path = os.path.abspath(path)
        self.filemask = filemask
        self.recursive = recursive
        self.verbose = verbose
        self.__read__()

    def __getitem__(self, key) -> dict:
        """Get configurations for a given key.

        Parameters:
            key : str
                Key of the training configuration. For example 'fit_generator' or 'model'.
                For a list of keys see :func:`keys`.

        Returns:
            Dictionary of configuration parameters for the given key.
        """
        return self.configs[key]

    def __read__(self):
        """Read the configuration files."""
        if self.recursive:
            pathname = os.path.join(self.path, '**', self.filemask)
        else:
            pathname = os.path.join(self.path, self.filemask)

        self.filenames = glob.glob(pathname, recursive=self.recursive)

        self.filenames = [os.path.normpath(filename) for filename in self.filenames]

        self.indices = []
        for filename in self.filenames:
            tail = os.path.relpath(os.path.dirname(os.path.abspath(filename)), start=self.path)
            self.indices.append(tail)

        self.configs = dict()
        for index, filename in zip(self.indices, self.filenames):
            with open(filename, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                for key, values in config.items():
                    cfg = self.configs.setdefault(key, {})
                    cfg[index] = values

        if self.verbose:
            print(f'{len(self.filenames)} files found.')

    def keys(self):
        """Return list of keys.

        Returns:
            dictitems
        """
        return self.configs.keys()

    def to_dataframe(self, key, fillna=''):
        """Convert configurations to dataframe.

        Parameters:
            key : str
                Key of the training configuration.
            fillna : str, optional
                Fill value for missing values. Defaults to ''.

        Returns:
            pd.DataFrame
        """
        df = pd.DataFrame.from_records(self.__getitem__(key))
        df = df.transpose()
        df = df.fillna(fillna)
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)

        return df

    def to_excel(self, filename, key=None, fillna='', column_width=None):
        """Write configurations to excel file.

        Parameters:
            filename : str
                Filename of excel file to write.
            key : str, optional
                Key of the training configuration(s) to write, defaults to None, i.e. write all configurations.
            fillna : str, optional
                Fill value for missing values, defaults to ''.
            column_width : float, optional
                Fixed column width, defaults to None.
        """
        if key is None:
            keys = self.keys()
        else:
            keys = list(key)

        full_filename = os.path.abspath(os.path.join(self.path, filename))
        if self.verbose:
            print(f'Write output to {full_filename}')

        with pd.ExcelWriter(full_filename) as writer:
            for current_key in sorted(keys):
                df = self.to_dataframe(current_key, fillna=fillna)
                df.to_excel(writer, sheet_name=current_key)
                writer.sheets[current_key].freeze_panes = "B2"

                if column_width is None:
                    width = max([len(column) for column in df.columns])
                else:
                    width = column_width

                for column_cells in writer.sheets[current_key].columns:
                    header = column_cells[0]
                    writer.sheets[current_key].column_dimensions[header.column_letter].width = width
