# -*- coding: utf-8 -*-
"""
Collection of plot functions.

@author: Andreas Groth
"""

# TODO: add latent_sampling as argument to decoder functions. Manipulate z_mean, draw z and feed to decoder.

from types import SimpleNamespace

import cartopy.crs as ccrs
import numpy as np
from cartopy import feature as cfeature
from cartopy.mpl import ticker as cticker
from cartopy.util import add_cyclic_point
from matplotlib import cycler, dates
from matplotlib import pyplot as plt
from matplotlib import ticker
from scipy.signal import find_peaks
from scipy.stats import norm
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar

from VAE._iowrapper import DataHandler


def decoder_plot(decoder: Model,
                 z_values,
                 z_mean=0,
                 cond=None,
                 target=None,
                 order=None,
                 index=0,
                 channels=None,
                 labels=None,
                 highlight='lightgray',
                 name='Decoder plot',
                 batch_size=32,
                 verbose=1,
                 sharex=True,
                 sharey=True):
    """Plot decoder output for sampled latent variable. One variable at a time.

    Draw samples from latent space and plot decoder output. The function varies one latent variable at a time, while all
    others are set to `z_mean`.

    Parameters:
        decoder : Model
            Instance of the decoder model.
        z_values : Sequence
            Values by which the latent variables will be varied.
        z_mean : Scalar or Numpy array
            Values of the unchanged latent variables. If Numpy array, `z_mean` must be broadcastable to an array of
            shape `(len(z_values), set_size, latent_dim)`.
        cond : Numpy array
            Optional input condition.
        order: Sequence
            Plotting order of the latent dimensions.
        index : int
            Index of the decoder output that will be shown. Index refers to the second dimension of size `set_size`.
        channels : list of ints
            Channels to be shown. Channels refer to the last dimension of the decoder output. If None, all channels
            will be shown.
        labels : list of str
            Labels of the channels.
        name: String
            Name of the figure.
        verbose: int
            Verbosity level.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    _, set_size, latent_dim = decoder.inputs[0].get_shape().as_list()
    _, set_size, output_length, output_channels = decoder.outputs[0].get_shape().as_list()

    z_mean = np.broadcast_to(z_mean, (len(z_values), set_size, latent_dim))
    z_values = np.array(z_values)[:, None]
    idx_z0 = np.argmin(np.abs(z_values - z_mean[:, 0, :]), axis=0)

    if channels is None:
        channels = range(output_channels)

    if cond is not None:
        cond = np.broadcast_to(cond, (len(z_values), set_size, cond.shape[-1]))

    if order is None:
        order = np.arange(latent_dim)

    output_reverse = [layer.name for layer in decoder.layers if 'reverse' in layer.name]
    if output_reverse:
        x = range(-output_length, 0)
    else:
        x = range(output_length)

    linestyles = cycler(color=[plt.get_cmap('tab10')(n) for n in channels])

    fig, axs = _create_figure(fig=name, rows=len(z_values), columns=len(order), sharex=sharex, sharey=sharey)
    pbar = Progbar(len(order), unit_name='latent dimension', verbose=verbose, interval=1)
    for column, k in enumerate(order):
        z = z_mean.copy()
        z[..., k] = z_values

        # predict
        if cond is None:
            y = decoder.predict(z, batch_size=batch_size)
        else:
            y = decoder.predict([z, cond], batch_size=batch_size)

        y = y[:, index, ...][..., channels]

        axs[-1, column].set_title('$k={}$'.format(k))

        for row, yr in enumerate(y):
            ax = axs[row, column]
            ax.set_prop_cycle(linestyles)
            p = ax.plot(x, yr, zorder=1.1, linewidth=2)
            if target is not None:
                ax.plot(x, target, zorder=1, alpha=0.5)

            ax.grid(axis='both', linestyle=':')
            ax.axhline(0, color='k', linewidth=1.25, linestyle=':')
            if column == 0:
                ax.set_ylabel(z_values[row, 0])

            if row == idx_z0[k]:
                ax.set_facecolor(highlight)

            if (row == len(y) - 1) and (column == len(order) - 1):
                if labels is not None:
                    ax.legend(p, labels, loc='upper left', bbox_to_anchor=(1, 1.05))

        pbar.add(1)

    fig.text(0.01, 0.5, '$z_k$', va='center', rotation='vertical', fontsize='large')

    return fig, axs


def decoder_plot2d(decoder: Model,
                   latent_pair,
                   z_values,
                   z_mean=0,
                   cond=None,
                   target=None,
                   index=0,
                   channels=None,
                   labels=None,
                   highlight='lightgray',
                   batch_size=32,
                   name="Decoder plot 2D",
                   verbose=1,
                   sharex=True,
                   sharey=True):
    """Plot decoder output for sampled latent variable. Two variables at a time.

    Draw samples from latent space and plot decoder output. The function varies two latent variable at a time and sets
    all others to ``z0``.

    Note:
        For models in :mod:`models_single`.

    Parameters:
        decoder: Model
            Instance of the decoder model.
        latent_pair: Tuple/list of two integers
            Latent dimensions that will be varied.
        z_values: Sequence
            Values by which the latent variables will be varied.
        z_mean: Scalar or Numpy array
            Values of the unchanged latent variables. If Numpy array, `z_mean` must be briadcastable to an array of
            shape `(len(z_values), set_size, latent_dim)`.
        cond : Numpy array
            Optional input condition.
        index : int
            Index of the decoder output that will be shown. Index refers to the second dimension of size `set_size`.
        channels : list of ints
            Channels to be shown. Channels refer to the last dimension of the decoder output. If None, all channels
            will be shown.
        labels : list of str
            Labels of the channels.
        name: String
            Name of the figure.

    Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    _, set_size, latent_dim = decoder.inputs[0].get_shape().as_list()
    _, set_size, output_length, output_channels = decoder.outputs[0].get_shape().as_list()

    z_mean = np.broadcast_to(z_mean, (len(z_values), set_size, latent_dim))
    z_values = np.array(z_values)[:, None]
    k0, k1 = latent_pair
    idx_z0 = np.argmin(np.abs(z_values - z_mean[:, 0, :]), axis=0)

    if channels is None:
        channels = range(output_channels)

    if cond is not None:
        cond = np.broadcast_to(cond, (len(z_values), set_size, cond.shape[-1]))

    output_reverse = [layer.name for layer in decoder.layers if 'reverse' in layer.name]
    if output_reverse:
        x = range(-output_length, 0)
    else:
        x = range(output_length)

    linestyles = cycler(color=[plt.get_cmap('tab10')(n) for n in channels])

    fig, axs = _create_figure(fig=name, rows=len(z_values), columns=len(z_values), sharex=sharex, sharey=sharey)
    pbar = Progbar(len(z_values), unit_name='column', verbose=verbose, interval=1)
    for column in range(len(z_values)):
        # prepare latent samples
        z = z_mean.copy()
        z[..., k0] = z_values[column, ...]
        z[..., k1] = z_values

        # predict
        if cond is None:
            y = decoder.predict(z, batch_size=batch_size)
        else:
            y = decoder.predict([z, cond], batch_size=batch_size)

        y = y[:, index, ...][..., channels]

        axs[0, column].set_xlabel(z_values[column, 0])
        for row, yr in enumerate(y):
            ax = axs[row, column]
            ax.set_prop_cycle(linestyles)
            p = ax.plot(x, yr, zorder=1, linewidth=2)
            if target is not None:
                ax.plot(x, target, zorder=1, alpha=0.5)

            ax.grid(axis='both', linestyle=':')
            ax.axhline(0, color='k', linewidth=1.25, linestyle=':')
            if column == 0:
                ax.set_ylabel(z_values[row, 0])

            if (column == idx_z0[k0]) and (row == idx_z0[k1]):
                ax.set_facecolor(highlight)

            if (row == len(y) - 1) and (column == len(z_values) - 1):
                if labels is not None:
                    ax.legend(p, labels, loc='upper left', bbox_to_anchor=(1, 1.05))

        pbar.add(1)

    fig.text(0.01, 0.5, f'$z_{{{k1}}}$', va='center', rotation='vertical', fontsize='large')
    fig.text(0.5, 0.01, f'$z_{{{k0}}}$', ha='center', fontsize='large')

    return fig, axs


def decoder_pcolormesh(decoder: Model,
                       z_values,
                       z0=0,
                       cond=None,
                       channel=0,
                       order=None,
                       vmin=None,
                       vmax=None,
                       cmap=None,
                       highlight='#bbbbbb',
                       name='Decoder plot',
                       batch_size=32,
                       verbose=1):
    """Plot decoder output for sampled latent variable. One variable at a time.

    Draw samples from latent space and plot decoder output. The function varies one latent variable at a time, while all
    others are set to ``z0``.

    Parameters:
        decoder: Model
            Instance of the decoder model.
        z_values: Sequence
            Values by which the latent variables will be varied.
        z0: Scalar or Numpy array
            Values of the unchanged latent variables. If Numpy array, ``z0`` must be compatible
            with shape ``(len(z_values), latent_dim)``.
        cond : Numpy array
            Optional input condition.
        channel : int
            Channel to be shown.
        order: Sequence
            Plotting order of the latent dimensions.
        name: String
            Name of the figure.
        verbose: 0, 1, 2
            Verbosity mode.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    z_values = np.array(z_values)
    latent_dim = decoder.inputs[0].get_shape().as_list()[1]
    z0 = np.broadcast_to(z0, (1, latent_dim))
    z0 = np.repeat(z0, len(z_values), axis=0)

    if cond is not None:
        cond = np.repeat(cond, len(z_values), axis=0)

    idx_z0 = np.argmin(np.abs(z_values[:, None] - z0), axis=0)

    if order is None:
        order = np.arange(latent_dim)

    fig, axs = _create_figure(fig=name, rows=len(z_values), columns=len(order), subplots_adjust=False)

    pbar = Progbar(len(order), unit_name='latent dimension', verbose=verbose, interval=1)

    for column, column_k in enumerate(order):
        # sample from latent space
        # vary single value and set others to z0
        z = z0.copy()
        z[:, column_k] = z_values

        # predict
        if cond is None:
            y = decoder.predict(z, batch_size=batch_size)
        else:
            y = decoder.predict([z, cond], batch_size=batch_size)

        axs[-1, column].set_title('$k={}$'.format(column_k))

        for row, yr in enumerate(y):
            ax = axs[row, column]
            ax.pcolormesh(yr[..., channel], vmin=vmin, vmax=vmax, cmap=cmap, zorder=1)

            if column == 0:
                ax.set_ylabel('$z_k={:.1f}$'.format(z_values[row]))

            if row == idx_z0[column_k]:
                ax.patch.set_edgecolor(highlight)
                ax.patch.set_linewidth(10)

        pbar.add(1)

    return fig, axs


def decoder_response(decoder: Model,
                     inputs,
                     z_pertub_p=None,
                     z_pertub_n=None,
                     order=None,
                     index=0,
                     channels=None,
                     labels=None,
                     batch_size=32,
                     name='Decoder response',
                     verbose=1,
                     showmax=True,
                     sharex=True,
                     sharey=True):
    """Plot decoder response.

    Decoder response with respect to changes in the latent variables. The response is the difference between the decoder
    output of the positive response from `z_pertub_p` and the negative response from `z_pertub_n`.

    Parameters:
        decoder: Model
            Instance of the decoder model.
        inputs:
            Inputs to the decoder.
        z_pertub_p : callable, optional
            Callable applied to the latent variables that will be used to calculate the positive response.
        z_pertub_n : callable, optional
            Callable applied to the latent variables that will be used to calculate the negative/neutral response.
        order: list of int
            Plotting order of the latent dimensions.
        index : int
            Index of the decoder output that will be shown. Index refers to the first dimension after the batch
            dimension.
        channels : list of ints
            Channels to be shown. Channels refer to the last dimension of the decoder output. If None, all channels
            will be shown.
        labels : list of str
            Labels of the channels.
        name: str
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    if isinstance(inputs, list):
        z_mean, condition = inputs
        condition = [condition]
    else:
        z_mean = inputs
        condition = []

    _, set_size, output_length, output_channels = decoder.outputs[0].get_shape().as_list()
    latent_dim = decoder.inputs[0].get_shape().as_list()[-1]
    if channels is None:
        channels = range(output_channels)

    if z_mean.ndim == 2:
        z_mean = z_mean[:, None, :]

    z_mean = np.broadcast_to(z_mean, (len(z_mean), set_size, latent_dim))

    if order is None:
        order = range(latent_dim)

    pbar = Progbar(len(order), unit_name='latent dimension', verbose=verbose, interval=1)
    fig, axs = _create_figure(fig=name,
                              rows=len(channels),
                              columns=len(order),
                              flip_rows=False,
                              sharex=sharex,
                              sharey=sharey)

    output_reverse = [layer.name for layer in decoder.layers if 'reverse' in layer.name]
    if output_reverse:
        x = range(-output_length, 0)
    else:
        x = range(output_length)

    for column, k in enumerate(order):
        zp = z_mean.copy()
        zn = z_mean.copy()

        if z_pertub_p is not None:
            zp[..., k] = z_pertub_p(zp[..., k])

        if z_pertub_n is not None:
            zn[..., k] = z_pertub_n(zp[..., k])

        # predict
        yp = decoder.predict([zp] + condition, batch_size=batch_size)
        yn = decoder.predict([zn] + condition, batch_size=batch_size)
        y_diff = yp - yn
        y_diff = y_diff[:, index, ...][..., channels]

        # average over batches
        y_diff_mean = np.mean(y_diff, axis=0)
        y_diff_prcs = np.percentile(y_diff, [5, 95], axis=0)

        axs[0, column].set_title(f'$z_{{{k}}}$')
        for row in range(len(channels)):
            ax = axs[row, column]
            ax.plot(x, y_diff_mean[:, row], color='tab:blue', zorder=1.2)
            ax.fill_between(x,
                            y_diff_prcs[0, :, row],
                            y_diff_prcs[1, :, row],
                            facecolor='lightsteelblue',
                            edgecolor='tab:blue',
                            linewidth=1,
                            zorder=1.1)

            ax.grid(axis='both', linestyle=':')
            ax.axhline(0, color='k', linewidth=1.25, linestyle=':')
            if column == 0:
                if labels is not None:
                    ax.set_ylabel(labels[row])
                else:
                    ax.set_ylabel(row)
            if showmax:
                peaks, _ = find_peaks(np.abs(y_diff_mean[:, row]))
                if len(peaks) > 0:
                    plt.text(0.05,
                             0.95,
                             f'\u0394={x[peaks[0]]}',
                             transform=ax.transAxes,
                             ha='left',
                             va='top',
                             bbox=dict(facecolor='white', alpha=0.5))

        pbar.add(1)

    for ax_row in axs:
        ylims = [np.max(np.abs(ax.get_ylim())) for ax in ax_row]
        if sharey:
            ax_row[0].set_ylim(-ylims[0], ylims[0])
        else:
            for ax, ylim in zip(ax_row, ylims):
                ax.set_ylim(-ylim, ylim)

    return fig, axs


def decoder_composite(decoder: Model,
                      inputs,
                      order=None,
                      index=0,
                      channels=None,
                      adjusted=False,
                      labels=None,
                      batch_size=32,
                      name='Decoder composite',
                      verbose=1,
                      showmax=True,
                      sharex=True,
                      sharey=True):
    """Plot decoder composite.

    The decoder composite is obtained by setting all but one latent dimension to zero.

    Parameters:
        decoder: Model
            Instance of the decoder model.
        inputs:
            Inputs to the decoder.
        order: list of int
            Plotting order of the latent dimensions.
        index : int
            Index of the decoder output that will be shown. Index refers to the first dimension after the batch
            dimension.
        channels : list of ints
            Channels to be shown. Channels refer to the last dimension of the decoder output. If None, all channels
            will be shown.
        adjusted: bool
            Whether the composite is adjusted by removing the average decoder output sampled from the prior.
        labels : list of str
            Labels of the channels.
        name: str
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    if isinstance(inputs, list):
        z_mean, condition = inputs
        condition = [condition]
    else:
        z_mean = inputs
        condition = []

    _, set_size, output_length, output_channels = decoder.outputs[0].get_shape().as_list()
    latent_dim = decoder.inputs[0].get_shape().as_list()[-1]
    if channels is None:
        channels = range(output_channels)

    if z_mean.ndim == 2:
        z_mean = z_mean[:, None, :]

    z_mean = np.broadcast_to(z_mean, (len(z_mean), set_size, latent_dim))

    if order is None:
        order = range(latent_dim)

    pbar = Progbar(len(order), unit_name='latent dimension', verbose=verbose, interval=1)
    fig, axs = _create_figure(fig=name,
                              rows=len(channels),
                              columns=len(order),
                              flip_rows=False,
                              sharex=sharex,
                              sharey=sharey)

    output_reverse = [layer.name for layer in decoder.layers if 'reverse' in layer.name]
    if output_reverse:
        x = range(-output_length, 0)
    else:
        x = range(output_length)

    for column, k in enumerate(order):
        zc = np.zeros_like(z_mean)
        zc[..., k] = z_mean[..., k]

        # predict
        yc = decoder.predict([zc] + condition, batch_size=batch_size)

        if adjusted:
            yc -= decoder.predict([zc * 0] + condition, batch_size=batch_size)

        yc = yc[:, index, ...][..., channels]

        # average over batches
        # positive
        idx = z_mean[:, 0, k] > 0
        yp_mean = np.mean(yc[idx, ...], axis=0)
        yp_prcs = np.percentile(yc[idx, ...], [5, 95], axis=0)
        # negative
        idx = z_mean[:, 0, k] < 0
        yn_mean = np.mean(yc[idx, ...], axis=0)
        yn_prcs = np.percentile(yc[idx, ...], [5, 95], axis=0)

        axs[0, column].set_title(f'$z_{{{k}}}$')
        for row in range(len(channels)):
            ax = axs[row, column]
            hp1, = ax.plot(x, yp_mean[:, row], color='tab:red', zorder=1.2)
            hp2 = ax.fill_between(x,
                                  yp_prcs[0, :, row],
                                  yp_prcs[1, :, row],
                                  alpha=0.5,
                                  facecolor='tab:red',
                                  edgecolor='tab:red',
                                  linewidth=1,
                                  zorder=1.1)

            hn1, = ax.plot(x, yn_mean[:, row], color='tab:blue', zorder=1.2)
            hn2 = ax.fill_between(x,
                                  yn_prcs[0, :, row],
                                  yn_prcs[1, :, row],
                                  alpha=0.5,
                                  facecolor='tab:blue',
                                  edgecolor='tab:blue',
                                  linewidth=1,
                                  zorder=1.1)

            ax.grid(axis='both', linestyle=':')
            ax.axhline(0, color='k', linewidth=1.25, linestyle=':')
            if column == 0:
                if labels is not None:
                    ax.set_ylabel(labels[row])
                else:
                    ax.set_ylabel(row)

            if ax == axs[0, -1]:
                ax.legend([(hp1, hp2), (hn1, hn2)], ['$z>0$', '$z<0$'], loc='upper left', bbox_to_anchor=(1, 1))

            if showmax:
                peaks, _ = find_peaks(np.abs(yp_mean[:, row]))
                if len(peaks) > 0:
                    plt.text(0.05,
                             0.95,
                             f'\u0394={x[peaks[0]]}',
                             transform=ax.transAxes,
                             ha='left',
                             va='top',
                             bbox=dict(facecolor='white', alpha=0.5))

        pbar.add(1)

    for ax_row in axs:
        ylims = [np.max(np.abs(ax.get_ylim())) for ax in ax_row]
        if sharey:
            ax_row[0].set_ylim(-ylims[0], ylims[0])
        else:
            for ax, ylim in zip(ax_row, ylims):
                ax.set_ylim(-ylim, ylim)

    return fig, axs


def decoder_response_hist(decoder: Model,
                          inputs,
                          z_pertub_p=None,
                          z_pertub_n=None,
                          order=None,
                          index=0,
                          channels=None,
                          bins=10,
                          vmin=None,
                          vmax=None,
                          cmap=None,
                          norm=None,
                          batch_size=32,
                          name='Temporal response',
                          labels=None,
                          verbose=1,
                          sharex=True,
                          sharey=True):
    """Plot decoder response histogram.

    Decoder response with respect to changes in the latent variables.

    Parameters:
        decoder : Model
            Instance of the decoder.
        inputs :
            Inputs to the decoder.
        z_pertub_p : callable, optional
            Callable applied to the latent variables that will be used to calculate the positive response.
        z_pertub_n : callable, optional
            Callable applied to the latent variables that will be used to calculate the negative/neutral response.
        bins : int or Sequence
            See :func:`numpy.histogram`.
        order : iterable
            Plotting order of the latent dimensions.
        index : int
            Index of the decoder output that will be shown. Index refers to the first dimension after the batch
            dimension.
        channels : list of ints
            Channels to be shown. Channels refer to the last dimension of the decoder output. If None, all channels
            will be shown.
        labels : list of str
            Labels of the channels.
        name : str
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    def _hist(x, bins):
        pdf = []
        for x_row in x.T:
            new_pdf, bins = np.histogram(x_row, bins=bins, density=True)
            pdf.append(new_pdf)

        return np.stack(pdf, axis=1), bins

    if isinstance(inputs, list):
        z_mean, condition = inputs
        condition = [condition]
    else:
        z_mean = inputs
        condition = []

    _, set_size, output_length, output_channels = decoder.outputs[0].get_shape().as_list()
    latent_dim = decoder.inputs[0].get_shape().as_list()[-1]
    if channels is None:
        channels = range(output_channels)

    if z_mean.ndim == 2:
        z_mean = z_mean[:, None, :]

    z_mean = np.broadcast_to(z_mean, (len(z_mean), set_size, latent_dim))

    if order is None:
        order = np.arange(latent_dim)

    output_reverse = [layer.name for layer in decoder.layers if 'reverse' in layer.name]
    if output_reverse:
        x = range(-output_length, 0)
    else:
        x = range(output_length)

    pbar = Progbar(len(order), unit_name='latent dimension', verbose=verbose, interval=1)
    fig, axs = _create_figure(fig=name,
                              rows=len(channels),
                              columns=len(order),
                              flip_rows=False,
                              sharex=sharex,
                              sharey=sharey)

    for column, k in enumerate(order):
        zp = z_mean.copy()
        zn = z_mean.copy()

        if z_pertub_p is not None:
            zp[..., k] = z_pertub_p(zp[..., k])

        if z_pertub_n is not None:
            zn[..., k] = z_pertub_n(zp[..., k])

        # predict
        yp = decoder.predict([zp] + condition, batch_size=batch_size)
        yn = decoder.predict([zn] + condition, batch_size=batch_size)
        yp = yp[:, index, ...][..., channels]
        yn = yn[:, index, ...][..., channels]

        axs[0, column].set_title(f'$z_{{{k}}}$')
        bins = np.array(bins)
        for row in range(len(channels)):
            yp_pdf, bins = _hist(yp[:, :, row], bins=bins)
            yn_pdf, bins = _hist(yn[:, :, row], bins=bins)
            y_pdf = yp_pdf - yn_pdf

            ax = axs[row, column]
            ax.pcolormesh(x, bins, y_pdf, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, zorder=1.1)
            ax.grid(axis='x')
            if column == 0:
                if labels is not None:
                    ax.set_ylabel(labels[row])
                else:
                    ax.set_ylabel(channels[row])

        pbar.add(1)

    return fig, axs


def encoder_boxplot(encoder, inputs, plottype='kl', sort=True, batch_size=None, name="Encoder boxplot", verbose=1):
    """Boxplot of latent variables from encoder output.

    If `plottype` equals to `mean`, the function creates a boxplot of the latent variable `z_mean` that correspond
    to `inputs` given to the encoder. If `plottype` equals to `var`, the function creates a boxplot of the latent
    variable `z_log_var`. If `plottype` equals to `kl`, the function  creates a boxplot of the KL divergence.

    Parameters:
        encoder : class:`keras.Model`
            Instance of the encoder model.
        inputs : Numpy array or generator
            Input to the encoder.
        plottype : str
            Type of boxplot. One of `mean`, `var`, `kl`.
        sort : bool
            Plot latent dimension in descending order of mean Kullback-Leibler divergence.
        name: String
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: :class:`matplotlib.axes.Axes`
                Axes object.
            idx: Sequence
                Order of sorted latent dimensions.
            kl_loss : 2D Numpy array
                Values of the KL divergence corresponding to the input.

    """
    # get latent samples
    z_mean, z_log_var, *_ = encoder.predict(inputs, verbose=verbose, batch_size=batch_size)

    # z has shape (samples, latent_dim)
    latent_dim = z_mean.shape[1]

    fig = plt.figure(name)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)

    kl_loss = 1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl_loss *= -0.5
    kl_loss_mean = np.mean(kl_loss, axis=0)

    if sort:
        idx = np.argsort(kl_loss_mean)
        idx = idx[::-1]
    else:
        idx = np.arange(latent_dim)

    labels = ['{}'.format(i) for i in idx]

    if plottype.lower() == 'mean':
        variable = z_mean
        yscale = 'Linear'
        ylabel = 'Mean'
    elif plottype.lower() == 'var':
        variable = np.exp(z_log_var)
        yscale = 'log'
        ylabel = 'Variance'
    elif plottype.lower() == 'kl':
        variable = kl_loss
        yscale = 'linear'
        ylabel = 'KL divergence'
    else:
        assert False, "Unknown option in `plottype` argument."

    ax.boxplot(
        variable[:, idx],
        showfliers=False,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor='lightsteelblue'),
        medianprops=dict(color='tab:blue'),
        meanprops=dict(markerfacecolor='tab:blue'),
        zorder=2,
    )

    ax.set_xlabel('k')
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.grid(axis='both', linestyle=':')

    return fig, ax, idx, kl_loss


def encoder_hist(encoder,
                 inputs,
                 latent_sampling=None,
                 bins=30,
                 order=None,
                 show_null=True,
                 batch_size=None,
                 name="Encoder hist",
                 verbose=1):
    """Histogram of latent variables from encoder output.

    The function plots the histogram of the latent variable `z_mean` that correspond to `inputs` given to the encoder.
    If `latent_sampling` is specified, then the histogram of random samples `z` from `latent_sampling` is shown instead.

    Together with the histogram, the prior distribution is shown as a bold line.

    Parameters:
        encoder: class:`keras.Model`
            Instance of the encoder model.
        inputs: 4D Numpy array
            Input to the encoder. The first dimension is the batch dimension.
        latent_sampling : class:`keras.Model`
            Instance of the latent sampling model. Defaults to `None`.
        bins: Integer or sequence
            Number or edges of bins.
        order: Sequence
            Plotting order of the latent dimensions.
        show_null : boolean
            Whether to show the null hypothesis of a normal distribution with mean 0 and standard deviation 1.
        name: String
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    # get latent samples
    z_mean, z_log_var, *_ = encoder.predict(inputs, verbose=verbose, batch_size=batch_size)

    # draw random z
    if latent_sampling is not None:
        zs = latent_sampling.predict([z_mean, z_log_var], verbose=verbose, batch_size=batch_size)
        if zs.ndim == 3:
            zs = zs[:, 0, :]

    bins = np.array(bins)
    if bins.size == 1:
        bins = np.linspace(np.min(z_mean), np.max(z_mean), bins)

    norm_pdf = norm.pdf(bins, 0, 1)

    fig, axs = _create_figure(fig=name, rows=1, columns=len(order))

    for column, column_k in enumerate(order):
        ax = axs[0, column]
        if latent_sampling is None:
            ax.hist(z_mean[:, column_k],
                    bins=bins,
                    density=True,
                    color='red',
                    histtype='stepfilled',
                    edgecolor='k',
                    zorder=2.2)
        else:
            ax.hist(zs[:, column_k],
                    bins=bins,
                    density=True,
                    color='tab:cyan',
                    histtype='stepfilled',
                    edgecolor='k',
                    zorder=2.1)

        if show_null:
            ax.plot(bins, norm_pdf, color='k', zorder=2.3)

        ax.axvline(0, color='k', linewidth=1.25, linestyle=':', zorder=2.2)
        ax.set_xlabel('$z_{{{}}}$'.format(column_k))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    xl = np.max(np.abs(ax.get_xlim()))
    ax.set_xlim([-xl, xl])

    return fig, axs


def encoder_hist2d(encoder,
                   inputs,
                   latent_sampling=None,
                   bins=30,
                   order=None,
                   batch_size=None,
                   cmap=None,
                   norm=None,
                   name="Encoder hist2d",
                   verbose=1):
    """Histogram of pairs of latent variables from encoder output.

    The function creates pariwise histograms of the latent variable `z_mean` that correspond to ``inputs`` given to the
    encoder. If `latent_sampling` is specified, then pairwise histograms of random samples `z` from `latent_sampling`
    are shown instead.

    Parameters:
        encoder : :class:`keras.Model`
            Instance of the encoder model.
        inputs : 4D Numpy array
            Input to the encoder. The first dimension is the batch dimension.
        latent_sampling : class:`keras.Model`
            Instance of the latent sampling model. Defaults to `None`.
        bins : int or sequence
            Number or edges of bins.
        order : Sequence
            Plotting order of the latent dimensions.
        name : str
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    # get latent samples
    z_mean, z_log_var, *_ = encoder.predict(inputs, verbose=verbose, batch_size=batch_size)

    # draw random z
    if latent_sampling is not None:
        z = latent_sampling.predict([z_mean, z_log_var], verbose=verbose, batch_size=batch_size)
        if z.ndim == 3:
            z = z[:, 0, :]

    # z_mean has shape (samples, latent_dim)
    latent_dim = z_mean.shape[-1]

    if order is None:
        order = np.arange(latent_dim)

    limit = np.max(np.abs(z_mean[:, order]))
    bins = np.array(bins)
    if bins.size == 1:
        bins = np.linspace(-limit, limit, bins)

    fig, axs = _create_figure(fig=name, rows=len(order), columns=len(order), sharex=False, sharey=False)

    for row, row_k in enumerate(order):
        for column, column_k in enumerate(order):
            ax = axs[row, column]

            if row == column:
                if latent_sampling is None:
                    ax.hist(z_mean[:, column_k],
                            bins=bins,
                            density=True,
                            color='red',
                            histtype='stepfilled',
                            edgecolor='k')
                else:
                    ax.hist(z[:, column_k],
                            bins=bins,
                            density=True,
                            color='tab:cyan',
                            histtype='stepfilled',
                            edgecolor='k')

            else:
                if latent_sampling is None:
                    ax.hist2d(z_mean[:, column_k], z_mean[:, row_k], bins=bins, density=True, norm=norm, cmap=cmap)
                else:
                    ax.hist2d(z[:, column_k], z[:, row_k], bins=bins, density=True, norm=norm, cmap=cmap)

                ax.axhline(0, color='k', zorder=0)
                ax.axvline(0, color='k', zorder=0)
                ax.set_xlim([-limit, limit])
                ax.set_ylim([-limit, limit])

            if row == 0:
                ax.set_xlabel('$z_{{{}}}$'.format(column_k))

            if column == 0:
                ax.set_ylabel('$z_{{{}}}$'.format(row_k), rotation=0)

            if row == len(order):
                z_var = np.median(np.exp(z_log_var[:, column_k]))
                title = r'$\sigma_{{{}}}^2={:1.2f}$'.format(column_k, z_var)
                ax.set_title(title)

    return fig, axs


def encoder_scatter(encoder,
                    inputs,
                    latent_sampling=None,
                    bins=30,
                    order=None,
                    batch_size=None,
                    name='Encoder scatter',
                    verbose=1):
    """Scatter plot of latent variables from encoder output.

    The function creates pairwise scatter plots of the latent variable `z_mean` that correspond to ``inputs`` given to
    the encoder. If `latent_sampling` is specified, then pairwise scatter plots of the random samples `z` from
    `latent_sampling` are likewwise shown.

    Parameters:
        encoder : class:`ks.Model`
            Instance of the encoder model.
        inputs : 4D Numpy array
            Input to the encoder. The first dimension is the batch dimension.
        latent_sampling : class:`keras.Model`
            Instance of the latent sampling model. Defaults to `None`.
         bins: Integer or sequence
            Number or edges of bins of histograms along main diagonal.
        order: Sequence
            Plotting order of the latent dimensions.
        name: String
            Name of the figure.

     Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    # get latent samples
    z_mean, z_log_var, *_ = encoder.predict(inputs, verbose=verbose, batch_size=batch_size)

    # draw random z
    if latent_sampling is not None:
        z = latent_sampling.predict([z_mean, z_log_var], verbose=verbose, batch_size=batch_size)
        if z.ndim == 3:
            z = z[:, 0, :]

    # z has shape (samples, latent_dim)
    latent_dim = z_mean.shape[1]

    limit = np.max(np.abs(z_mean))
    bins = np.array(bins)
    if bins.size == 1:
        bins = np.linspace(-limit, limit, bins)

    if order is None:
        order = np.arange(latent_dim)

    fig, axs = _create_figure(fig=name, rows=len(order), columns=len(order), sharex=False, sharey=False)

    for row, row_k in enumerate(order):
        for column, column_k in enumerate(order):
            ax = axs[row, column]

            if row == column:
                if latent_sampling is None:
                    ax.hist(z_mean[:, column_k],
                            bins=bins,
                            density=True,
                            color='tab:red',
                            histtype='stepfilled',
                            edgecolor='k')
                else:
                    ax.hist(z[:, column_k],
                            bins=bins,
                            density=True,
                            color='tab:cyan',
                            histtype='stepfilled',
                            edgecolor='k')

            else:
                if latent_sampling is not None:
                    ax.plot(z[:, column_k], z[:, row_k], '.', color='tab:cyan', markersize=3, zorder=1)

                ax.plot(z_mean[:, column_k], z_mean[:, row_k], '.', color='tab:red', markersize=3, zorder=1.2)
                ax.axhline(0, color='k', zorder=1.1)
                ax.set_ylim([-limit, limit])

            ax.set_xlim([-limit, limit])
            ax.axvline(0, color='k', zorder=0)

            if row == 0:
                ax.set_xlabel('$z_{{{}}}$'.format(column_k))

            if column == 0:
                ax.set_ylabel('$z_{{{}}}$'.format(row_k), rotation=0)

            if row == len(order):
                z_var = np.median(np.exp(z_log_var[:, column_k]))
                title = r'$\sigma_{{{}}}^2={:1.2f}$'.format(column_k, z_var)
                ax.set_title(title)

    return fig, axs


def _create_figure(rows=1,
                   columns=1,
                   fig=None,
                   grid=None,
                   sharex=True,
                   sharey=True,
                   subplots_adjust=True,
                   flip_rows=True):
    """Prepare figure and axis."""
    if fig is None:
        fig = plt.figure()
    elif isinstance(fig, (str, int)):
        fig = plt.figure(fig)

    fig.clf()

    # adjust spacing between subplots
    if subplots_adjust:
        fig.subplots_adjust(left=0.075, right=0.95, bottom=0.1, top=0.9, wspace=0.0, hspace=0.0)

    subplot_kw = {'xticks': [], 'yticks': []}

    fig, axs = plt.subplots(rows,
                            columns,
                            sharex=sharex,
                            sharey=sharey,
                            subplot_kw=subplot_kw,
                            num=fig.number,
                            squeeze=False)

    # flip row order
    if flip_rows:
        axs = axs[::-1, :]

    # add xticks to lower axes
    for ax in axs[0, :]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator('auto', prune='both'))

    # add yticks to left axes
    for ax in axs[:, 0]:
        ax.yaxis.set_major_locator(ticker.MaxNLocator('auto', prune='both'))

    for ax in axs.flatten():
        ax.tick_params(axis='y', direction='inout', length=6)
        if grid is not None:
            ax.grid(axis=grid)

    return fig, axs


def _set_figure_size(fig, axs, figwidth):
    """"Adjust figure size of GeoAxesSubplots."""
    fig.canvas.draw()
    try:
        width, height = figwidth
    except TypeError:
        nrows, ncols = axs.shape
        extents = axs.flat[0].spines['geo'].get_tightbbox(fig.canvas.get_renderer())
        nrows = max(1.25, nrows)
        factor = extents.height / extents.width / ncols * nrows
        width, height = figwidth, factor * figwidth

    fig.set_size_inches(width, height)


def _map_to_dict(map_data, labels):
    """Convert map_data to dict. Adjust labels to match length of map_data."""
    if labels is not None and len(labels) != len(map_data):
        raise ValueError("Length of labels must match length of map_data")

    if isinstance(map_data, dict):
        map_data_dict = map_data
    else:
        # wrap 2d array into a list
        if isinstance(map_data, np.ndarray) and map_data.ndim == 2:
            map_data = [map_data]

        # convert sequence to dict
        keys = range(len(map_data))
        map_data_dict = dict()
        for key, value in zip(keys, map_data):
            map_data_dict[key] = value
            if value.ndim != 2:
                raise ValueError('Unsupported map_data.')

    if labels is None:
        labels = map_data_dict.keys()

    return map_data_dict, labels


def map_plot(latitude,
             longitude,
             map_data,
             labels=None,
             ncols=1,
             vmin=None,
             vmax=None,
             projection=ccrs.PlateCarree(),
             type='pcolormesh',
             coastlinespec_kw=None,
             colorbar_kw=None,
             gridlinespec_kw=None,
             gridspec_kw=None,
             landspec_kw=None,
             figwidth=None,
             **kwargs):
    """Plot a sequence of maps.

    The function creates a map for each entry in `map_data`.

    Parameters:
        latitude : Numpy array
            Latitude coordinate corresponding to the leading dimension of the maps.
        longitude : Numpy array
            Longitude coordinate corresponding to the second dimension of the maps.
        map_data : dict or sequence of ndarray
            Dictionary or sequence of 2D map data. Can also be a 3D array.
        labels : List of str
            List of labels for the sequence of the map data. Length must match the length of map_data.
        ncols : int
            Number of columns in which the subplots will be arranged.
        **kwargs :
            Additional keyword arguments passed to the plotting function.

    Returns:
        tuple:
            fig : Figure object.
            axs : Array of axes objects.
            cb :  Colorbar object.

    """
    map_data, labels = _map_to_dict(map_data, labels)
    nrows = -(-len(map_data) // ncols)  # ceil

    coastlinespec_kw = coastlinespec_kw or {'color': 'k', 'alpha': 1, 'linewidth': 1}
    colorbar_kw = colorbar_kw or {'shrink': 0.7 / nrows, 'pad': 0.01, 'aspect': 40 / nrows, 'extend': 'both'}
    gridspec_kw = gridspec_kw or {'wspace': 0.05, 'hspace': 0.25}
    gridlinespec_kw = gridlinespec_kw or {'draw_labels': True, 'linestyle': ':'}

    subplot_kw = dict(projection=projection)
    fig, axs = plt.subplots(nrows, ncols, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, squeeze=False)

    if vmin is None:
        vmins = [np.nanmin(value) for value in map_data.values()]
        vmin = np.nanmin(vmins)

    if vmax is None:
        vmaxs = [np.nanmax(value) for value in map_data.values()]
        vmax = np.nanmax(vmaxs)

    for ax, label, value in zip(axs.flat, labels, map_data.values()):
        value_cyclic, longitude_cyclic = add_cyclic_point(value, coord=longitude)
        plot_fcn = getattr(ax, type)
        im = plot_fcn(longitude_cyclic,
                      latitude,
                      value_cyclic,
                      vmin=vmin,
                      vmax=vmax,
                      shading='nearest',
                      transform=ccrs.PlateCarree(),
                      **kwargs)
        gl = ax.gridlines(**gridlinespec_kw)
        gl.top_labels = False
        gl.right_labels = False
        if ax not in axs[:, 0]:
            gl.left_labels = False

        if ax not in axs[-1, :]:
            gl.bottom_labels = False

        ax.coastlines(**coastlinespec_kw)
        if landspec_kw is not None:
            ax.add_feature(cfeature.LAND, **landspec_kw)
        ax.set_title(label)

    if figwidth:
        _set_figure_size(fig, axs, figwidth)

    for ax in axs.flat[len(map_data):]:
        fig.delaxes(ax)

    if len(map_data) == axs.size:
        cb = fig.colorbar(im, ax=axs, **colorbar_kw)
    else:
        pad = colorbar_kw.get('pad', 0.05)
        if colorbar_kw.get('orientation', 'vertical').lower() == 'vertical':
            cb = fig.colorbar(im, ax=axs[-1, :], **{**colorbar_kw, 'pad': pad - 1 / ncols})
        else:
            cb = fig.colorbar(im, ax=axs[:, -1], **{**colorbar_kw, 'pad': pad - 1 / nrows})

    return fig, axs, cb


def map_zonal(datetime,
              latitude,
              longitude,
              map_data,
              lon_lim=(-180, 180),
              labels=None,
              cmap='seismic',
              norm=None,
              vmin=None,
              vmax=None,
              figsize=None):
    """Plot a sequencde of zonal averages.

    The function plots the zonal average for each entry in `map_data`.

    Parameters:
        datetime: datetime
            Datetime corresponding to the leading dimension in the entries of `map_data`.
        latitude : Numpy array
            Latitude coordinate corresponding to the second dimension in the entries of `map_data`.
        longitude : Numpy array
            Longitude coordinate corresponding to the third dimension in the entries of `map_data`.
        map_data : dict or sequence of ndarray
            Dictionary or sequence of 2D map data. Can also be a 3D array.
        lon_lim : tuple of two float
            Longitude limits in which the zonal average is obtained.
        labels : List of str
            List of labels . Must match the length of map_data.

    Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    map_data, labels = _map_to_dict(map_data, labels)

    coord = SimpleNamespace(lon=longitude)
    lon_idx = DataHandler.get_lon_index(coord, *lon_lim)

    nrows = len(map_data)
    fig, axs = plt.subplots(nrows, 1, squeeze=True, sharex=True, sharey=True, figsize=figsize)
    lon_fmt = cticker.LongitudeFormatter()

    for ax, label, value in zip(axs.flat, labels, map_data.values()):
        zonal_mean = np.nansum(value[:, :, lon_idx], axis=-1) / len(lon_idx)
        if vmax is None and vmin is None:
            vmax = np.nanpercentile(np.abs(zonal_mean), 99)
            vmin = -vmax

        any_isfinite = np.any(np.isfinite(value), axis=(1, 2))
        any_isfinite = np.flatnonzero(any_isfinite)
        sl = slice(any_isfinite[0], any_isfinite[-1])

        im = ax.pcolormesh(datetime[sl],
                           latitude,
                           zonal_mean[sl, :].T,
                           shading='nearest',
                           cmap=cmap,
                           norm=norm,
                           vmin=vmin,
                           vmax=vmax)

        ax.set_title('{} ({}\u2013{})'.format(label, *[lon_fmt._format_value(lon, None) for lon in lon_lim]))
        ax.grid(which='major', axis='both')
        ax.grid(which='minor', axis='x', linestyle=':')
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=0.9 / min(nrows, 3))

    ax.xaxis.set_major_locator(dates.YearLocator(5))
    ax.xaxis.set_minor_locator(dates.YearLocator(1))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(cticker.LatitudeLocator())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    return fig, axs


def map_meridional(datetime,
                   latitude,
                   longitude,
                   map_data,
                   lat_lim=(-90, 90),
                   labels=None,
                   cmap='seismic',
                   norm=None,
                   vmin=None,
                   vmax=None,
                   figsize=None):
    """Plot a sequence of meridional averages.

    The function plots the meridional average for each entry in `map_data`.

    Parameters:
        datetime: datetime
            Datetime corresponding to the leading dimension in the entries of `map_data`.
        latitude : Numpy array
            Latitude coordinate corresponding to the second dimension in the entries of `map_data`.
        longitude : Numpy array
            Longitude coordinate corresponding to the third dimension in the entries of `map_data`.
        map_data : dict or sequence of ndarray
            Dictionary or sequence of 2D map data. Can also be a 3D array.
        lat_lim : tuple of two float
            Latitude limits in which the meridional average is obtained.
        labels : List of str
            List of labels . Must match the length of map_data.

    Returns:
        tuple:
            fig: :class:`matplotlib.figure.Figure`
                The figure instance.
            axs: Array of :class:`matplotlib.axes.Axes`
                Array of axes objects.

    """
    map_data, labels = _map_to_dict(map_data, labels)

    coord = SimpleNamespace(lat=latitude)
    lat_idx = DataHandler.get_lat_index(coord, *lat_lim)

    longitude_wrap = longitude.copy()
    longitude_wrap[longitude_wrap > 180] -= 360
    lon_idx = np.argsort(longitude_wrap)

    nrows = len(map_data)
    fig, axs = plt.subplots(nrows, 1, squeeze=True, sharex=True, sharey=True, figsize=figsize)
    lat_fmt = cticker.LatitudeFormatter()

    for ax, label, value in zip(axs.flat, labels, map_data.values()):
        meridional_mean = np.nansum(value[:, lat_idx, :], axis=1) / len(lat_idx)
        if vmax is None and vmin is None:
            vmax = np.nanpercentile(np.abs(meridional_mean), 99)
            vmin = -vmax

        any_isfinite = np.any(np.isfinite(value), axis=(1, 2))
        any_isfinite = np.flatnonzero(any_isfinite)
        sl = slice(any_isfinite[0], any_isfinite[-1])

        im = ax.pcolormesh(datetime[sl],
                           longitude_wrap[lon_idx],
                           meridional_mean[sl, lon_idx].T,
                           shading='nearest',
                           cmap=cmap,
                           norm=norm,
                           vmin=vmin,
                           vmax=vmax)

        ax.set_title('{} ({}\u2013{})'.format(label, *[lat_fmt._format_value(lat, None) for lat in lat_lim]))
        ax.grid(which='major', axis='both')
        ax.grid(which='minor', axis='x', linestyle=':')
        fig.colorbar(im, ax=ax, pad=0.01, fraction=0.05, shrink=0.9 / min(nrows, 3))

    ax.xaxis.set_major_locator(dates.YearLocator(5))
    ax.xaxis.set_minor_locator(dates.YearLocator(1))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(cticker.LongitudeLocator())
    ax.yaxis.set_major_formatter(cticker.LongitudeFormatter())

    return fig, axs
