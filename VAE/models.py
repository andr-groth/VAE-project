# -*- coding: utf-8 -*-
"""Many-to-many version of variational auto-encoder.

Model specifications:
    - Multiple set members (`set_size`) as time distributed layers, the leading dimension after the batch dimension.
    - Input shape to encoder `(set_size, input_length, channels)`.
    - Output shape of encoder `(latent_dim,)`: mean and log-variance of latent space, common to all members in set.
    - Output shape of latent_sampling `(set_size, latent_dim)`: independent random samples for each member in set.
    - Input shape to decoder `(set_size, latent_dim)`.
    - Output shape of decoder `(set_size, input_length, channels)`.
    - Masked multihead attention over filters x set_size. Temporal mask along dimension of size input_length.
    - Residual blocks with causal convolution with pre-activation and pixel-shuffling for downsampling and upsampling
    - Decoder with internal padding blocks
    - 3 FiLM blocks:
        encoder: after input expansion, after encoder blocks, and after attention,
        decoder: after input expansion, after sttention blocks, and decoder blocks

"""
__version__ = '2022-09-08'

# Changelog
# ---------
# 2022-09-08 : added Similarity and SimilarityBetween to VAELoss
# 2022-09-07 : added BatchNorm to encoder input and decoder output
# 2022-09-01 : added Similarity and SimilarityBetween to PredictionLoss
# 2022-08-15 : added option `ratio` to ResidualUnit
# 2022-07-05 : replaced permute with transposed in MultiHeadAttentionBlock
# 2022-07-04 : varying kernel size for convolutional layers
# 2022-07-04 : internal padding is more flexible
# 2022-06-30 : added optional free_bits argument to VAE and VAEp
# 2022-06-28 : added GumbelSoftmax layer to condition
# 2022-06-09 : introduced TotalCorrelationWithin loss with parameter `gamma_within`
# 2022-06-09 : introduced TotalCorrelationBetween loss with parameter `gamma_between`
# 2022-05-28 : added support for cond_size being a list of integers
# 2022-05-28 : add film_temporal option to EnsembleFilm
# 2022-05-28 : revised params for FiLM layers
# 2020-05-28 : replace FilmBlocks with Film layers
# 2022-05-18 : added support for TotalCorrelationBlkDiag loss
# 2022-05-18 : set BatchNorm momemtum to default value of 0.99
# 2022-05-16 : added BatchNormalization to MAB
# 2022-05-11 : added option for time-dependence to FiLM block
# 2022-05-15 : added third FiLM to MAB output
# 2022-05-11 : added second FiLM to input/output
# 2022-05-08 : changed method names to CamelCase
# 2022-02-16 : set kernel_size=1 in MAB residual units
# 2021-12-15 : implemented option for MC Dropout
# 2021-10-10 : replaced Attention with AttentionMasked in MAB

from typing import Union

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

from VAE import layers as vaelayers
from VAE import logs as vaelogs
from VAE import losses as vaelosses
from VAE.utils import collection


def Encoder(input_shape: tuple[int, int],
            latent_dim: int,
            set_size: int = 1,
            set_noise: bool = True,
            encoder_blocks: int = 0,
            residual_units: int = 1,
            activation: str = None,
            depth_wise: bool = False,
            filters: int = 1,
            kernel_size: Union[int, list[int]] = 2,
            ratio: int = 1,
            attn_activation: str = 'softmax',
            attn_blocks: int = 0,
            attn_filters: Union[int, str] = 'full',
            attn_heads: int = 1,
            attn_masked: bool = True,
            attn_transposed: bool = False,
            cond_activation: Union[str, list[str]] = None,
            cond_size: Union[int, tuple[int, int]] = None,
            cond_units: Union[int, list[int]] = 16,
            cond_use_bias: bool = True,
            cond_use_scale: bool = True,
            cond_use_offset: bool = True,
            cond_ens_size: int = None,
            cond_temperature: float = 1.,
            fc_activation: str = None,
            fc_units: int = None,
            film_activation: str = None,
            film_temporal: bool = False,
            film_use_bias: bool = True,
            film_use_scale: bool = True,
            film_use_offset: bool = True,
            pooling: str = 'reduce_sum',
            name: str = 'encoder',
            **kwargs) -> ks.Model:
    """Encoder model.

    This function creates an encoder model, for use in :func:`VAEp`, for example. This model takes multiple set members
    as input and returns a single `z_mean` and `z_log_var` via pooling.

    Parameters:
        input_shape:
            Input shape of the encoder. A tuple of two integers `(input_length, channels)`, with `input_length` being
            the length of the input sequence and `channels` being the number of input channels. If the input length is
            not a power of 2, the input will be zero-padded to the next power of 2.
        latent_dim:
            Size of the latent space.
        set_size:
            Number of samples in a set. This is the leading dimension, after the batch dimension.
        encoder_blocks:
            Number of encoder blocks. Defaults to 0.
        residual_units:
            Number of residual units in each of the encoder blocks.
        filters:
            Number of filters that the input is expanded to.
        activation:
            Activation function for convolutions in residual units.
        kernel_size:
            Kernel size of convolutions in residual units. If a tuple is given, the kernel size can be different for the
            different encoder blocks. In this case, the length of the tuple must be equal to `encoder_blocks`.
        ratio:
            Compression ratio of filters in the residual units. Defaults to 1.
        fc_units:
            Number of units in last dense layer, i.e. before sampling of `z_mean` and `z_log_var`. Set to `None` to
            disable the dense layer.
        fc_activation:
            Activation function of last dense layer.
        pooling:
            Pooling function. Set to `None` to disable pooling.
        name:
            Name of the encoder model. Must be one of the following: `encoder`.

    Parameters: Attention parameters:
        attn_activation:
            Name of activation function for the scores in the attention.
        attn_blocks:
            Number of attention blocks. Defaults to 0.
        attn_filters:
            Number of attention filters.
        attn_heads:
            Number of attention heads.
        attn_masked:
            Whether apply a causal mask to the scores in the attention layer. Defaults to `True`.
        attn_transposed:
            Whether the attention is applied to the spatial dimension (`False`) or the channel dimension (`True`).

    Parameters: Conditional parameters:
        cond_size: int or tuple of two ints
            Size of conditional input. If `cond_size` is an integer, the conditional input is passed to a regular dense
            layer. If `cond_size` is a tuple of two integers, the conditional input of size `sum(cond_size)` is split
            along the channel axis into two tensors of size `cond_size[0]` and `cond_size[1]`, providing the condition
            and the ensemble id. The condition is passed to a (set) of dense layer(s), while the ensemble id is passed
            to a GumbelSoftmax layer, which is then used to FiLM the output of the dense layer(s) along the channel
            axis.
        cond_activation:
            Name of the activation function for dense layer(s).
        cond_units:
            Number of units for dense layer(s) applied to the input condition.  Set to `None` to disable dense layer.
        cond_use_bias:
            Whether to use bias in the input condition. Defaults to `True`.
        cond_use_scale:
            Whether to use scale in FiLM. Defaults to `True`.
        cond_use_offset:
            Whether to use offset in FiLM. Defaults to `True`.
        cond_ens_size:
            Size of the ensemble of GumbelSoftmax. Defaults to `None`.
        cond_temperature:
            Temperature of the GumbelSoftmax. Defaults to 1.

    Parameters: FiLM parameters:
        film_activation:
            Name of the activation function for the scale in the Film layers. Defaults to `None`.
        film_temporal:
            Whether the Film layers are time-dependent (apply to the second last dimension). Defaults to `False`.
        film_use_offset:
            Whether to use offset in the Film layers. Defaults to `True`.
        film_use_scale:
            Whether to use scale in the Film layers. Defaults to `True`.
        film_use_bias:
            Whether to use bias in the Film layers. Defaults to `True`.

    Returns:
        Instance of :class:`ks.Model`.

    Model inputs:
        If `cond_size` is None:
            Tensor of shape `(set_size, input_length, channels)`.
        If `cond_size` is not None:
            Tuple of two tensors of shape `(set_size, input_length, channels)` and `(set_size, cond_size)`.

    Model outputs:
        Tensor of shape `(latent_dim, )`

    """
    valid_names = {'encoder'}
    if name not in valid_names:
        raise ValueError(f"Name must be one of {valid_names}.")

    kernel_size = (kernel_size, ) * encoder_blocks if isinstance(kernel_size, int) else kernel_size
    if len(kernel_size) != encoder_blocks:
        raise ValueError(f"{len(kernel_size)=} must be equal to {encoder_blocks=}.")

    # Input of x
    x_in = ks.layers.Input(shape=(set_size, ) + tuple(input_shape), name=name + '_input')

    # Left zero-padding
    scale = 2**encoder_blocks
    input_length, _ = input_shape
    padded_length = int(np.ceil(input_length / scale)) * scale
    if padded_length > input_length:
        y = ks.layers.ZeroPadding2D(padding=((0, 0), (padded_length - input_length, 0)), name=name + '_input_pad')(x_in)
    else:
        y = x_in

    # Input BatchNormalization
    y = ks.layers.BatchNormalization(name=name + '_input_bn')(y)

    # Input expansion
    y = ks.layers.Dense(units=filters, name=name + '_input_expand')(y)

    shape = [set_size, padded_length, filters]

    # Input of condition
    if cond_size is not None:
        if isinstance(cond_size, int):
            cond_size = [cond_size]
        cond_in = ks.layers.Input(shape=(set_size, sum(cond_size)), name=name + '_cond')

        # optionally split the condition into two tensors
        if len(cond_size) == 1:
            y_cond, y_id = cond_in, None
        elif len(cond_size) == 2:
            y_cond, y_id = vaelayers.Split(size_splits=cond_size, axis=-1, name=name + '_cond_split')(cond_in)
        else:
            raise ValueError('cond_size must be either an integer or a list of two integers.')

        # apply dense layer(s) to the condition
        y_cond = MLP(units=cond_units, activation=cond_activation, use_bias=cond_use_bias,
                     name=name + '_cond_dense')(y_cond)

        # film condition
        if y_id is not None:
            if cond_ens_size is not None:
                y_id = ks.layers.Dense(units=cond_ens_size, name=name + '_cond_id')(y_id)
                noise_shape = None if set_noise else (None, 1, None)
                y_id = vaelayers.GumbelSoftmax(noise_shape=noise_shape,
                                               temperature=cond_temperature,
                                               name=name + '_cond_id_gumbel')(y_id)

            y_cond = vaelayers.Film(use_scale=cond_use_scale,
                                    use_offset=cond_use_offset,
                                    use_bias=False,
                                    name=name + '_cond_film')([y_cond, y_id])

    else:
        cond_in = None
        y_cond = None

    if film_temporal:
        # apply Film to last and second last dimension
        film_shape = (None, None)
    else:
        # apply Film to last dimension only
        film_shape = (1, None)

    film_kw = {
        'activation': film_activation,
        'use_bias': film_use_bias,
        'use_offset': film_use_offset,
        'use_scale': film_use_scale,
    }

    # Conditional input FiLM, always applied to the last dimension
    if y_cond is not None:
        y = vaelayers.Film(shape=(1, None), name=name + '_input_film', **film_kw)([y, y_cond])

    # Encoder blocks
    for n, k_size in enumerate(kernel_size):
        y = EncoderBlock(shape,
                         residual_units=residual_units,
                         activation=activation,
                         depth_wise=depth_wise,
                         kernel_size=k_size,
                         ratio=ratio,
                         name=name + '_block_{}'.format(n + 1))(y)
        shape[1] //= 2
        shape[2] *= 2

    # Conditional FiLM of conv output
    if (y_cond is not None) and (encoder_blocks > 0):
        y = vaelayers.Film(shape=film_shape, name=name + '_conv_film', **film_kw)([y, y_cond])

    # Attention blocks
    for n in range(attn_blocks):
        y = MultiheadAttentionBlock(shape,
                                    filters=attn_filters,
                                    heads=attn_heads,
                                    masked=attn_masked,
                                    transposed=attn_transposed,
                                    attn_activation=attn_activation,
                                    res_activation=activation,
                                    name=name + '_MAB_{}'.format(n + 1))(y)

    # Conditional FiLM of attn output
    if (y_cond is not None) and (attn_blocks > 0):
        y = vaelayers.Film(shape=film_shape, name=name + '_attn_film', **film_kw)([y, y_cond])

    # Pooling over set_size
    if set_size > 1 and pooling is not None:
        pooling_fcn = getattr(tf, pooling)
        y = ks.layers.Lambda(lambda x: pooling_fcn(x, axis=1, keepdims=True), name=name + '_pooling_set')(y)

    # Flatten
    y = ks.layers.Flatten(name=name + '_flatten')(y)

    # Final nonlinearity
    if fc_units:
        y = ks.layers.Dense(fc_units, activation=fc_activation, name=name + '_z_dense')(y)

    # Compress to latent dimensions
    z_mean = ks.layers.Dense(latent_dim, name=name + '_z_mean')(y)
    z_log_var = ks.layers.Dense(latent_dim, name=name + '_z_log_var')(y)

    # Build encoder model
    if cond_in is not None:
        inputs = [x_in, cond_in]
    else:
        inputs = [x_in]

    model = ks.Model(inputs=inputs, outputs=[z_mean, z_log_var], name=name)

    return model


def Decoder(output_shape: tuple[int, int],
            latent_dim: int,
            set_size: int = 1,
            set_noise: bool = True,
            decoder_blocks: int = 0,
            residual_units: int = 1,
            activation: str = None,
            depth_wise: bool = False,
            filters: int = 1,
            kernel_size: Union[int, list[int]] = 2,
            ratio: int = 1,
            attn_activation: str = 'softmax',
            attn_blocks: int = 0,
            attn_filters: Union[int, str] = 'full',
            attn_heads: int = 1,
            attn_masked: bool = True,
            attn_transposed: bool = False,
            cond_activation: Union[str, list[str]] = None,
            cond_size: Union[int, tuple[int, int]] = None,
            cond_units: Union[int, list[int]] = 16,
            cond_use_bias: bool = True,
            cond_use_scale: bool = True,
            cond_use_offset: bool = True,
            cond_ens_size: int = None,
            cond_temperature: float = 1.,
            film_activation: str = None,
            film_temporal: bool = False,
            film_use_bias: bool = True,
            film_use_scale: bool = True,
            film_use_offset: bool = True,
            fc_units: int = None,
            fc_activation: str = None,
            padding_blocks: int = 0,
            output_reverse: bool = True,
            name: str = 'decoder',
            **kwargs) -> ks.Model:
    """Decoder model.

    This function creates a decoder model, for use in :func:`VAE`, for example. This model
    takes multiple realizations `z` from the latent space and returns multiple samples.

    Parameters:
        output_shape:
            Output shape of the decoder. A tuple of two integers `(output_length, channels)`, with `output_length` being
            the length of the output sequence and `channels` being the number of output channels. If `output_length` is
            not a power of 2, the decoder will internally work with a sequence length to the next power of 2 and
            crop the sequence to the desired length.
        latent_dim:
            Size of the latent space.
        set_size:
            Number of output samples in a set. This is the leading dimension, after the batch dimension.
        decoder_blocks:
            Number of decoder blocks. Defaults to 0.
        output_reverse:
            Reverse temporal order of output.
        padding_blocks:
            Number of internal padding blocks. Instead of zero-padding, further padding blocks are drawn from `z`.
            Defaults to 0.
        kernel_size:
            Size of the convolutional kernel. If a list, the kernel size can be different for each decoder block. In
            this case, the length of the list must be equal to `decoder_blocks` and the order is reversed.
        name:
            Name of the decoder model. Must be one of the following: `decoder`, `prediction`.

    For the rest of the parameters, see :func:`Encoder`.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        If `cond_size` is None:
            Tensor of shape `(set_size, latent_dim)`.
        If `cond_size` is not None:
            Tuple of two tensors of shape `(set_size, latent_dim)` and `(set_size, cond_size)`.

    Model outputs:
        Tensor of shape `(set_size, output_length, channels)`.

    """
    valid_names = {'decoder', 'prediction'}
    if name not in valid_names:
        raise ValueError(f"Name must be one of {valid_names}.")

    kernel_size = (kernel_size, ) * decoder_blocks if isinstance(kernel_size, int) else kernel_size
    if len(kernel_size) != decoder_blocks:
        raise ValueError(f"{len(kernel_size)=} must be equal to {decoder_blocks=}.")

    # Input of z
    z_in = ks.layers.Input(shape=(set_size, latent_dim), name=name + '_input')

    # optional first layer
    if fc_units:
        y = ks.layers.Dense(fc_units, activation=fc_activation, name=name + '_input_dense')(z_in)
    else:
        y = z_in

    # Right pad
    # note: if the decoder output is reversed, the padding is on the left such as in the encoder
    scale = 2**decoder_blocks
    output_length, _ = output_shape
    padded_length = int(np.ceil(output_length / scale)) * scale
    right_pad = padded_length - output_length

    # Left pad with internal padding blocks
    left_pad = padding_blocks * scale
    full_length = left_pad + padded_length

    # Expand
    y = ks.layers.Dense(full_length * filters, name=name + '_input_expand')(y)

    # Reshape
    shape = [set_size, full_length // scale, filters * scale]
    y = ks.layers.Reshape(shape, name=name + '_input_reshape')(y)

    # Input of condition
    if cond_size is not None:
        if isinstance(cond_size, int):
            cond_size = [cond_size]
        cond_in = ks.layers.Input(shape=(set_size, sum(cond_size)), name=name + '_cond')

        # optionally split the condition into two tensors
        if len(cond_size) == 1:
            y_cond, y_id = cond_in, None
        elif len(cond_size) == 2:
            y_cond, y_id = vaelayers.Split(size_splits=cond_size, axis=-1, name=name + '_cond_split')(cond_in)
        else:
            raise ValueError('cond_size must be either an integer or a list of two integers.')

        # apply dense layer(s) to the condition
        y_cond = MLP(units=cond_units, activation=cond_activation, use_bias=cond_use_bias,
                     name=name + '_cond_dense')(y_cond)

        # film condition
        if y_id is not None:
            if cond_ens_size is not None:
                y_id = ks.layers.Dense(units=cond_ens_size, name=name + '_cond_id')(y_id)
                noise_shape = None if set_noise else (None, 1, None)
                y_id = vaelayers.GumbelSoftmax(noise_shape=noise_shape,
                                               temperature=cond_temperature,
                                               name=name + '_cond_id_gumbel')(y_id)

            y_cond = vaelayers.Film(use_scale=cond_use_scale,
                                    use_offset=cond_use_offset,
                                    use_bias=False,
                                    name=name + '_cond_film')([y_cond, y_id])

    else:
        cond_in = None
        y_cond = None

    if film_temporal:
        # apply Film to last and second last dimension
        film_shape = (None, None)
    else:
        # apply Film to last dimension only
        film_shape = (1, None)

    film_kw = {
        'activation': film_activation,
        'use_bias': film_use_bias,
        'use_offset': film_use_offset,
        'use_scale': film_use_scale,
    }

    # Conditional input FiLM
    if y_cond is not None:
        y = vaelayers.Film(shape=film_shape, name=name + '_input_film', **film_kw)([y, y_cond])

    # Attention blocks
    for n in range(attn_blocks):
        y = MultiheadAttentionBlock(shape,
                                    filters=attn_filters,
                                    heads=attn_heads,
                                    masked=attn_masked,
                                    transposed=attn_transposed,
                                    attn_activation=attn_activation,
                                    res_activation=activation,
                                    name=name + '_MAB_{}'.format(n + 1))(y)

    # Conditional FiLM of attention output
    if (y_cond is not None) and (attn_blocks > 0):
        y = vaelayers.Film(shape=film_shape, name=name + '_attn_film', **film_kw)([y, y_cond])

    # Decoder blocks
    for n, k_size in enumerate(reversed(kernel_size)):
        y = DecoderBlock(shape,
                         residual_units=residual_units,
                         activation=activation,
                         depth_wise=depth_wise,
                         kernel_size=k_size,
                         ratio=ratio,
                         name=name + '_block_{}'.format(n + 1))(y)
        shape[1] *= 2
        shape[2] //= 2

    # Conditional FiLM of conv output, always applied to the last dimension
    if (y_cond is not None) and (decoder_blocks > 0):
        y = vaelayers.Film(shape=(1, None), name=name + '_conv_film', **film_kw)([y, y_cond])

    # Output aggregation
    y = ks.layers.Dense(units=output_shape[-1], name=name + '_output_aggregation')(y)

    # Output BatchNormalization
    y = ks.layers.BatchNormalization(name=name + '_output_bn')(y)

    # Output crop. Remove internal padding blocks
    y = ks.layers.Cropping2D(cropping=((0, 0), (left_pad, right_pad)), name=name + '_output_crop')(y)

    # Time reverse output
    if output_reverse:
        y = ks.layers.Lambda(lambda x: tf.reverse(x, axis=[2]), name=name + '_output_reverse')(y)

    outputs = ks.layers.Lambda(lambda x: x, name=name)(y)

    # Build decoder model
    if cond_in is not None:
        inputs = [z_in, cond_in]
    else:
        inputs = [z_in]

    model = ks.Model(inputs=inputs, outputs=outputs, name=name)

    return model


def LatentSampling(latent_dim: int,
                   set_size: int = 1,
                   set_noise: bool = True,
                   input_sigma: bool = False,
                   name: str = 'latent',
                   **kwargs) -> ks.Model:
    """Latent-sampling model.

    This function creates a model for latent sampling, for use in :func:`VAE`. The  model has input `z_mean`
    and `z_log_var` from the encoder and returns `set_size` random samples `z`.

    Parameters:
        latent_dim:
            Size of the latent space.
        set_size:
            Number of output samples in a set. This is the leading dimension, after the batch dimension.
        set_noise:
            Whether the noise varies between members in a set.
        input_sigma:
            Specify, whether random samples are provided as input to the model.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(latent_dim,)`.

    Model outputs:
        Tensor of shape `(set_size, latent_dim)`.

    """
    # Input layer
    input_shape = (latent_dim, )
    z_mean_in = ks.layers.Input(shape=input_shape, name=name + '_input_z_mean')
    z_log_var_in = ks.layers.Input(shape=input_shape, name=name + '_input_z_log_var')

    # broadcast the input to set_size
    z_mean = ks.layers.RepeatVector(set_size, name=name + '_z_mean_repeat')(z_mean_in)
    z_log_var = ks.layers.RepeatVector(set_size, name=name + '_z_log_var_repeat')(z_log_var_in)

    # Optional input layer
    if input_sigma:
        sigma = ks.layers.Input(shape=(set_size, ) + input_shape, name=name + '_sigma')
        inputs = [z_mean_in, z_log_var_in, sigma]
        sample_args = [z_mean, z_log_var, sigma]
    else:
        inputs = [z_mean_in, z_log_var_in]
        sample_args = [z_mean, z_log_var]

    # Random sampling
    noise_shape = None if set_noise else (None, 1, None)
    z = vaelayers.RandomSampling(noise_shape=noise_shape, name=name + '_z_sampling')(sample_args)

    # Build model
    model = ks.Model(inputs=inputs, outputs=z, name=name)

    return model


def EncoderBlock(shape: tuple[int, int, int],
                 residual_units: int = 1,
                 activation: str = None,
                 depth_wise: bool = False,
                 kernel_size: int = 1,
                 ratio: int = 1,
                 name: str = 'encoder_block') -> ks.Model:
    """Encoder-block model.

    This function creates an encoder-block model for the use in  :func:`Encoder`.

    Parameters:
        shape:
            Input shape of the block `(set_size, input_length, filters)`.
        residuals_units:
            Number of residual units.
        activation:
            Activation function.
        kernel_size:
            Size of the convolutional kernel.
        name:
            Name of encoder-block model.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(set_size, input_length, filters)`.

    Model outputs:
        Tensor of shape `(set_size, input_length // 2, 2 * filters)`.

    """
    y_in = ks.layers.Input(shape, name=name + '_in')
    y = y_in

    # residual unit(s)
    for n in range(residual_units):
        y = ResidualUnit(filters=y.shape.as_list()[-1],
                         activation=activation,
                         depth_wise=depth_wise,
                         kernel_size=kernel_size,
                         ratio=ratio,
                         name=name + '_R{}'.format(n + 1))(y)

    # downsampling by pixel shuffling
    set_size, input_length, filters = shape
    input_length //= 2
    filters *= 2
    y = ks.layers.Reshape((set_size, input_length, filters), name=name + '_down_shuffle')(y)

    model = ks.Model(inputs=y_in, outputs=y, name=name)

    return model


def DecoderBlock(shape: tuple[int, int, int],
                 residual_units: int = 1,
                 activation: str = None,
                 depth_wise: bool = False,
                 kernel_size: int = 1,
                 ratio: int = 1,
                 name: int = 'decoder_block') -> ks.Model:
    """Decoder-block model.

    This function creates a decoder-block model for the use in  :func:`Decoder`.

    Parameters:
        shape:
            Input shape of the block `(set_size, input_length, filters)`.
        residuals_units:
            Number of residual units.
        activation:
            Activation function.
        kernel_size:
            Size of the convolutional kernel.
        name:
            Name of decoder-block model.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(set_size, input_length, filters)`.

    Model outputs:
        Tensor of shape `(set_size, 2 * input_length, filters // 2)`.

    """
    y_in = ks.layers.Input(shape, name=name + '_in')

    # upsampling by pixel shuffling
    set_size, input_length, filters = shape
    input_length *= 2
    filters //= 2
    y = ks.layers.Reshape((set_size, input_length, filters), name=name + '_up_shuffle')(y_in)

    # residual units
    for n in range(residual_units):
        y = ResidualUnit(filters=y.shape.as_list()[-1],
                         activation=activation,
                         depth_wise=depth_wise,
                         kernel_size=kernel_size,
                         ratio=ratio,
                         name=name + '_R{}'.format(n + 1))(y)

    model = ks.Model(inputs=y_in, outputs=y, name=name)

    return model


def MLP(units: Union[int, list[int]],
        activation: Union[str, list[str]] = None,
        use_bias: Union[bool, list[bool]] = None,
        **kwargs):
    """Multi-layer perceptron.

    This is a convenience function to create a multi-layer perceptron. The parameters can be either single values or
    lists of values. If a list is given, the length of the list determines the number of layers. Otherwise, a single
    dense layer is created.

    Parameters:
        units:
            Number of units in each layer.
        activation:
            Activation function for each layer.
        use_bias:
            Whether to use bias in each layer.
        kwargs:
            Additional keyword arguments are passed to the dense layers.

    Model inputs:
        Tensor of arbitrary shape.

    Model outputs:
        Tensor of same shape as input, except for the last dimension which is specified by the last value in `units`.
    """
    if not isinstance(units, list):
        units = [units]
    if not isinstance(activation, list):
        activation = [activation] * len(units)
    if not isinstance(use_bias, list):
        use_bias = [use_bias] * len(units)

    name = kwargs.pop('name', '')

    def mlp(inputs):
        x = inputs
        for n, (_units, _activation, _use_bias) in enumerate(zip(units, activation, use_bias), start=1):
            x = ks.layers.Dense(units=_units, activation=_activation, use_bias=_use_bias, name=name + f'_{n}',
                                **kwargs)(x)
        return x

    return mlp


def MultiheadAttentionBlock(shape: tuple[int, int, int],
                            filters: Union[int, str] = 'full',
                            heads: int = 4,
                            masked: bool = True,
                            transposed: bool = False,
                            attn_activation: str = 'softmax',
                            res_activation: str = None,
                            name: str = 'mab') -> ks.Model:
    """Multihead-attention model.

    This function creates a model for multihead attention for use in :func:`Decoder` and
    :func:`Encoder`. The input shape of the model is `(set_size, input_length, filters)`. The dot-product
    attention is applied over reshaped matrices of shape `(set_size * input_length, filters)` . The score has a shape of
    `(set_size * input_length, set_size * input_length)`. Optional causal masking is applied to the scores and no
    temporal information is merged.

    Parameters:
        shape:
            Input shape of the model.
        filters:
            Number of filters in each of the heads. If `full`, the number of filters will be the same as in the input.
            If `reduced`, the number of filters will be divided by the number of heads. Default is `full`.
        heads:
            Number of heads. Default is 4.
        masked:
            Whether apply a causal mask to the score. Default is `True`.
        transposed:
            Whether the attention is applied to the `set_size` dimension (`False`) or the `filters` dimension (`True`).
            Default is `False`.
        attn_activation:
            Name of activation function for the scores. Default is `softmax`.
        res_activation:
            Name of activation function for the residual unit.  Default is `None`.
        name:
            Name of the model.  Default is `mab`.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(set_size, input_length, filters)`.

    Model outputs:
        Tensor of shape `(set_size, input_length, filters)`.

    References:
        Vaswani et al. (2017): https://arxiv.org/abs/1706.03762v5.

    """
    y_in = ks.layers.Input(shape=shape, name=name + '_y_in')

    # batch normalization
    y = ks.layers.BatchNormalization(name=name + '_bn')(y_in)

    if isinstance(filters, str):
        if filters not in ['full', 'reduced']:
            raise ValueError("filters must be either 'full' or 'reduced'.")
        if filters == 'full':
            filters = shape[-1]
        elif filters == 'reduced':
            filters = shape[-1] // heads

    # attention heads
    permute = (0, 3, 2, 1) if transposed else None
    y_list = []
    for n in range(heads):
        q = ks.layers.Dense(units=filters, name=name + '_query_{}'.format(n + 1))(y)
        k = ks.layers.Dense(units=filters, name=name + '_key_{}'.format(n + 1))(y)
        v = ks.layers.Dense(units=filters, name=name + '_value_{}'.format(n + 1))(y)

        y_list.append(
            vaelayers.AttentionMasked(activation=attn_activation,
                                      masked=masked,
                                      permute=permute,
                                      name=name + '_attn_{}'.format(n + 1))([q, k, v]))

    # concatenate and merge outputs of different heads
    if heads > 1:
        y = ks.layers.Concatenate(name=name + '_concat')(y_list)

        # optionally revert to number of input channels
        if filters * heads != shape[-1]:
            y = ks.layers.Dense(units=shape[-1], name=name + '_dense')(y)
    else:
        y = y_list[-1]

    # add residual connection
    y = ks.layers.Add(name=name + '_add')([y_in, y])

    # Final residual unit
    out = ResidualUnit(filters=shape[-1], activation=res_activation, kernel_size=1, name=name + '_R')(y)

    model = ks.Model(inputs=y_in, outputs=out, name=name)

    return model


def ResidualUnit(filters: int,
                 activation: str = None,
                 depth_wise: bool = False,
                 kernel_size: int = 1,
                 ratio: int = 1,
                 name: str = 'residual_unit'):
    """Residual unit.

    Residual unit for use in :func:`Decoder_block` and :func:`Encoder_block`.

    Parameters:
        filters:
            Number of filters in the convolutional layers.
        activation:
            Name of activation function. Default is `None`.
        depth_wise:
            Whether to use depth-wise separable convolutions. Default is `False`.
        kernel_size:
            Kernel size of the convolutional layers. Default is 1.
        ratio:
            Ratio of filters in the first convolutional layer, which will have `filters // ratio` filters.
            The second convolutional layer will have `filters` filters. Default is 1.
        name:
            Name of the model.  Default is `residual_unit`.

    Model inputs:
        Tensor of shape `(set_size, input_length, filters)`.

    Model outputs:
        Tensor of shape `(set_size, input_length, filters)`.

    """

    # filters = y.shape.as_list()[-1]

    def residal_unit(x: tf.Tensor) -> tf.Tensor:
        shortcut = x
        Conv1D = ks.layers.SeparableConv1D if depth_wise else ks.layers.Conv1D

        # 1. convolution
        x = ks.layers.BatchNormalization(name=name + '_bn1')(x)
        x = ks.layers.Activation(activation, name=name + '_act1')(x)
        x = ks.layers.TimeDistributed(
            Conv1D(filters=filters // ratio, kernel_size=kernel_size, padding='causal', name=name + '_conv1'))(x)

        # 2. convolution
        x = ks.layers.BatchNormalization(name=name + '_bn2')(x)
        x = ks.layers.Activation(activation, name=name + '_act2')(x)
        x = ks.layers.TimeDistributed(
            Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', name=name + '_conv2'))(x)

        # residual connection
        x = ks.layers.Add(name=name + '_add')([shortcut, x])

        return x

    return residal_unit


def VAE(encoder: ks.Model,
        decoder: ks.Model,
        latent_sampling: ks.Model,
        beta: Union[float, str] = 1.,
        delta: float = 0.,
        delta_between: float = 0.,
        gamma: float = 0.,
        gamma_between: float = 0.,
        gamma_within: float = 0.,
        repeat_samples: int = 1,
        learning_rate: float = 0.001,
        clipnorm: float = 1.,
        kl_threshold: float = None,
        free_bits: float = None,
        taper_range: tuple[float, float] = None,
        trainable: list[str] = None,
        name='mVAE',
        **kwargs) -> ks.Model:
    """Variational auto-encoder model.

    This function takes a `decoder`, `encoder`, and `latent sampling` model and creates and compiles a variational
    auto-encoder (VAE).

    Parameters:
        encoder:
            Encoder model.
        decoder:
            Decoder model.
        latent_sampling:
            LatentSampling model.
        beta:
            Loss weight of the KL divergence. If `str`, loss weight will be input to the model, which can be used to
            to anneal the KL divergence loss during training. The input name is determined by the `beta` argument.
        clipnorm:
            Gradient clipping norm.
        delta:
            Scale of similarity loss.
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        gamma:
            Scale of total correlation loss.
        gamma_between:
            Scale of total correlation loss between repeated samples.
        gamma_within:
            Scale of total correlation loss within repeated samples.
        learning_rate:
            Learning rate.
        kl_threshold:
            Lower bound for the KL divergence.
        free_bits:
            Number of bits to keep free for the KL divergence per latent dimension.
        repeat_samples:
            Number of repetitions of input samples in a batch ensemble.
        taper_range:
            Start and stop of the linear taper used to scale the squared error loss of the decoder. The taper is
            normalized to have mean 1.
        trainable:
            Names of the model's trainable layers. Defaults is None, meaning all layers are trainable.
        name:
            Name of the model.

    Returns:
        Instance of :class:`keras.Model`.

    """
    # model inputs
    if len(encoder.inputs) == 1:
        encoder_input = [ks.layers.Input(encoder.input_shape[1:], name=encoder.input_names[0])]
        encoder_cond = []
    else:
        encoder_input = [ks.layers.Input(encoder.inputs[0].shape.as_list()[1:], name=encoder.input_names[0])]
        encoder_cond = [ks.layers.Input(encoder.inputs[1].shape.as_list()[1:], name=encoder.input_names[1])]

    if len(decoder.inputs) == 1:
        decoder_cond = []
    else:
        decoder_cond = [ks.layers.Input(decoder.inputs[1].shape.as_list()[1:], name=decoder.input_names[1])]

    if len(latent_sampling.inputs) == 2:
        latent_sigma = []
    else:
        latent_sigma = [
            ks.layers.Input(latent_sampling.inputs[2].shape.as_list()[1:], name=latent_sampling.input_names[2])
        ]

    inputs = encoder_input + encoder_cond + decoder_cond + latent_sigma

    # add optional input for beta parameter
    if isinstance(beta, str):
        beta = ks.layers.Input((1, ), name=beta)
        inputs += [beta]
    else:
        beta = tf.constant(beta, shape=(1, ), name='beta')

    # encoding
    z_mean, z_log_var = encoder(encoder_input + encoder_cond)

    # latent sampling
    z = latent_sampling([z_mean, z_log_var] + latent_sigma)

    # decoding
    outputs = decoder([z] + decoder_cond)

    # create model
    model = ks.Model(inputs=inputs, outputs=outputs, name=name)

    # set trainable layers
    if trainable is not None:
        collection.set_trainable(model, trainable)

    _, _, decoder_length, decoder_channels = decoder.outputs[0].shape.as_list()
    decoder_size = decoder_length * decoder_channels

    # optional taper
    if taper_range is not None:
        start, stop = taper_range
        decoder_taper = np.linspace(start, stop, num=decoder_length)
        decoder_taper /= np.mean(decoder_taper)
    else:
        decoder_taper = None

    # training loss
    loss = vaelosses.VAEloss(z,
                             z_mean,
                             z_log_var,
                             size=decoder_size,
                             beta=beta,
                             delta=delta,
                             delta_between=delta_between,
                             gamma=gamma,
                             gamma_between=gamma_between,
                             gamma_within=gamma_within,
                             kl_threshold=kl_threshold,
                             free_bits=free_bits,
                             repeat_samples=repeat_samples,
                             taper=decoder_taper)

    # metrics
    weighted_metrics = [vaelosses.SquaredError(size=decoder_size), vaelosses.KLDivergence(z_mean, z_log_var), 'mse']

    if gamma:
        weighted_metrics.append(vaelosses.TotalCorrelation(z, z_mean, z_log_var))

    if gamma_between:
        weighted_metrics.append(vaelosses.TotalCorrelationBetween(z, z_mean, z_log_var, repeat_samples=repeat_samples))

    if gamma_within:
        weighted_metrics.append(vaelosses.TotalCorrelationWithin(z, z_mean, z_log_var, repeat_samples=repeat_samples))

    if delta:
        weighted_metrics.append(vaelosses.Similarity())

    if delta_between:
        weighted_metrics.append(vaelosses.SimilarityBetween(repeat_samples=repeat_samples))

    metrics = [vaelogs.Beta(beta), vaelogs.ActiveUnits(z_mean, z_log_var)]

    # compile model
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  weighted_metrics=weighted_metrics,
                  sample_weight_mode='temporal')

    return model


def VAEp(encoder: ks.Model,
         decoder: ks.Model,
         latent_sampling: ks.Model,
         prediction: ks.Model,
         beta: Union[float, str] = 1.,
         gamma: float = 0.,
         gamma_between: float = 0.,
         gamma_within: float = 0.,
         delta: float = 0.,
         delta_between: float = 0.,
         repeat_samples: int = 1,
         learning_rate: float = 0.001,
         clipnorm: float = 1.,
         kl_threshold: float = None,
         free_bits: float = None,
         loss_weights: dict = None,
         taper_ranges: tuple[float, float] = None,
         trainable: list[str] = None,
         name: str = 'mVAEp',
         **kwargs) -> ks.Model:
    """Variational auto-encoder model with prediction.

    This function takes `decoder`, `encoder`, `latent sampling`, and `prediction` models and creates and compiles a
    variational auto-encoder, combined with a decoder for prediction. The `prediction` model is second a decoder, also
    linked to the latent space, but that predicts samples forward in time.

    Parameters:
        encoder:
            Encoder model.
        decoder:
            Decoder model.
        latent_sampling:
            LatentSampling model.
        prediction:
            Second decoder model for prediction.
        beta:
            Loss weight of the KL divergence. If `str`, loss weight will be input to the model, which can be used to
            to anneal the KL divergence loss during training. The input name is determined by the `beta` argument.
        clipnorm:
            Gradient clipping norm.
        delta:
            Scale of similarity loss.
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        free_bits:
            Number of bits to keep free for the KL divergence per latent dimension.
        gamma:
            Scale of total correlation loss.
        gamma_between:
            Scale of total correlation loss between repeated samples.
        gamma_within:
            Scale of total correlation loss within repeated samples.
        kl_threshold:
            Lower bound for the KL divergence.
        learning_rate:
            Learning rate.
        loss_weights : dict
            Loss weights for the output of the decoder and prediction. The dict maps outputs names to scalar
            coefficients; e.g. `{'decoder': 1., 'prediction': 0.5}` .
        repeat_samples:
            Number of repetitions of input samples.
        taper_range:
            Start and stop of the linear taper used to scale the squared error loss of the decoder. The taper is
            normalized to have mean 1.
        trainable:
            Names of the model's trainable layers. Defaults is None, meaning all layers are trainable.
        name:
            Name of the model.

    Returns:
        Instance of :class:`keras.Model`.

    """
    # model inputs
    if len(encoder.inputs) == 1:
        encoder_input = [ks.layers.Input(encoder.input_shape[1:], name=encoder.input_names[0])]
        encoder_cond = []
    else:
        encoder_input = [ks.layers.Input(encoder.inputs[0].shape.as_list()[1:], name=encoder.input_names[0])]
        encoder_cond = [ks.layers.Input(encoder.inputs[1].shape.as_list()[1:], name=encoder.input_names[1])]

    if len(decoder.inputs) == 1:
        decoder_cond = []
    else:
        decoder_cond = [ks.layers.Input(decoder.inputs[1].shape.as_list()[1:], name=decoder.input_names[1])]

    if len(prediction.inputs) == 1:
        prediction_cond = []
    else:
        prediction_cond = [ks.layers.Input(prediction.inputs[1].shape.as_list()[1:], name=prediction.input_names[1])]

    if len(latent_sampling.inputs) == 2:
        latent_sigma = []
    else:
        latent_sigma = [
            ks.layers.Input(latent_sampling.inputs[2].shape.as_list()[1:], name=latent_sampling.input_names[2])
        ]

    inputs = encoder_input + encoder_cond + decoder_cond + prediction_cond + latent_sigma

    # add optional input for beta parameter
    if isinstance(beta, str):
        beta = ks.layers.Input((1, ), name=beta)
        inputs += [beta]
    else:
        beta = tf.constant(beta, shape=(1, ), name='beta')

    # encoding
    z_mean, z_log_var = encoder(encoder_input + encoder_cond)

    # latent sampling
    z = latent_sampling([z_mean, z_log_var] + latent_sigma)

    # decoding
    outputs = [
        decoder([z] + decoder_cond),
        prediction([z] + prediction_cond),
    ]

    # create model
    model = ks.Model(inputs=inputs, outputs=outputs, name=name)

    # set trainable layers
    if trainable is not None:
        collection.set_trainable(model, trainable)

    _, _, decoder_length, decoder_channels = decoder.outputs[0].shape.as_list()
    decoder_size = decoder_length * decoder_channels

    _, _, prediction_length, prediction_channels = prediction.outputs[0].shape.as_list()
    prediction_size = prediction_length * prediction_channels

    # optional taper
    if taper_ranges is not None:
        start, stop = taper_ranges.get(decoder.output_names[0], (1, 1))
        decoder_taper = np.linspace(start, stop, num=decoder_length)
        decoder_taper /= np.mean(decoder_taper)

        start, stop = taper_ranges.get(prediction.output_names[0], (1, 1))
        prediction_taper = np.linspace(start, stop, num=prediction_length)
        prediction_taper /= np.mean(prediction_taper)
    else:
        decoder_taper = None
        prediction_taper = None

    # training loss
    loss = {
        decoder.name:
            vaelosses.VAEloss(z,
                              z_mean,
                              z_log_var,
                              beta=beta,
                              delta=delta,
                              delta_between=delta_between,
                              gamma=gamma,
                              gamma_between=gamma_between,
                              gamma_within=gamma_within,
                              kl_threshold=kl_threshold,
                              free_bits=free_bits,
                              repeat_samples=repeat_samples,
                              size=decoder_size,
                              taper=decoder_taper),
        prediction.name:
            vaelosses.VAEploss(beta=beta,
                               delta=delta,
                               delta_between=delta_between,
                               repeat_samples=repeat_samples,
                               size=prediction_size,
                               taper=prediction_taper),
    }

    # metrics
    weighted_metrics = {
        decoder.name: [
            vaelosses.SquaredError(size=decoder_size),
            vaelosses.KLDivergence(z_mean, z_log_var),
            'mse',
        ],
        prediction.name: [
            vaelosses.SquaredError(size=prediction_size),
            'mse',
        ]
    }

    if gamma:
        weighted_metrics[decoder.name].append(vaelosses.TotalCorrelation(z, z_mean, z_log_var))

    if gamma_between:
        weighted_metrics[decoder.name].append(
            vaelosses.TotalCorrelationBetween(z, z_mean, z_log_var, repeat_samples=repeat_samples))

    if gamma_within:
        weighted_metrics[decoder.name].append(
            vaelosses.TotalCorrelationWithin(z, z_mean, z_log_var, repeat_samples=repeat_samples))

    if delta:
        weighted_metrics[decoder.name].append(vaelosses.Similarity())
        weighted_metrics[prediction.name].append(vaelosses.Similarity())

    if delta_between:
        weighted_metrics[decoder.name].append(vaelosses.SimilarityBetween(repeat_samples=repeat_samples))
        weighted_metrics[prediction.name].append(vaelosses.SimilarityBetween(repeat_samples=repeat_samples))

    metrics = {
        decoder.name: [vaelogs.Beta(beta), vaelogs.ActiveUnits(z_mean, z_log_var)],
    }

    # compile model
    optimizer = ks.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  loss_weights=loss_weights,
                  metrics=metrics,
                  weighted_metrics=weighted_metrics,
                  sample_weight_mode='temporal')

    return model


def IdentityModel(input_shape: tuple[int, int], set_size: int = 1) -> ks.Model:
    """Identity model.

    This function creates an identity model, which returns the input as output.

    Parameters:
        input_shape:
            Input shape of the model, excluding the `set_size` dimension.
        set_size:
            Number of input samples in a set.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(set_size, ) + input_shape`.

    Model outputs:
        Tensor of shape `(set_size, ) + input_shape`.

    """
    xin = ks.layers.Input((set_size, ) + tuple(input_shape), name='encoder_input')
    y = ks.layers.Lambda(lambda x: x, name='decoder')(xin)
    model = ks.Model(xin, y, name='Identity')
    return model


def example_VAE():
    """Example of a VAE model.

    Example:
        This function demonstrates how to build a VAE model using the method :func:`VAE.models.VAE`.
    """

    # We first define the parameters of the model:
    params = {
        'encoder_blocks': 1,
        'cond_size': 12,
        'fc_units': 48,
        'filters': 16,
        'input_shape': [16, 7],
        'latent_dim': 10,
        'trainable': ['*bn*'],
    }

    # Then we build the different parts of the model. We start with the encoder:
    encoder = Encoder(**params, name='encoder')

    # and the latent sampling layer:
    latent_sampling = LatentSampling(**params, name='latent')

    # and finally the decoder:
    decoder = Decoder(output_shape=params['input_shape'],
                      decoder_blocks=params['encoder_blocks'],
                      output_reverse=True,
                      **params,
                      name='decoder')

    # Once we have the different parts of the model, we can build the full model:
    model = VAE(encoder, decoder, latent_sampling, **params, name='VAE')

    # Let's have a look at the model:
    model.summary()

    # We can also have a look at the trainable parameters:
    collection.summary_trainable(model)

    # and plot the model:
    ks.utils.plot_model(model, show_shapes=True, dpi=75, rankdir='LR', to_file='example_VAE.png')

    return model


def example_VAEp():
    """Example of a VAEp model.

    Example:
        This function demonstrates how to build a VAEp model using the method :func:`VAE.models.VAEp`.
    """

    # We first define the parameters of the model:
    params = {
        'encoder_blocks': 1,
        'cond_size': 12,
        'fc_units': 48,
        'filters': 16,
        'input_shape': [16, 7],
        'latent_dim': 10,
        'trainable': ['*bn*'],
        'prediction_shape': [16, 1],
    }

    # Then we build the different parts of the model. We start with the encoder:
    encoder = Encoder(**params, name='encoder')

    # and the latent sampling layer:
    latent_sampling = LatentSampling(**params, name='latent')

    # Then we build the decoder:
    decoder = Decoder(output_shape=params['input_shape'],
                      decoder_blocks=params['encoder_blocks'],
                      output_reverse=True,
                      **params,
                      name='decoder')

    # and a second decoder for the prediction:
    prediction = Decoder(output_shape=params['prediction_shape'],
                         decoder_blocks=params['encoder_blocks'],
                         output_reverse=False,
                         **params,
                         name='prediction')

    # Once we have the different parts of the model, we can build the full model:
    model = VAEp(encoder, decoder, latent_sampling, prediction, **params, name='VAEp')

    # Let's have a look at the model:
    model.summary()

    # We can also have a look at the trainable parameters:
    collection.summary_trainable(model)

    # and plot the model:
    ks.utils.plot_model(model, show_shapes=True, dpi=75, rankdir='LR', to_file='example_VAEp.png')

    return model


if __name__ == '__main__':
    # model = example_VAE()
    model = example_VAEp()
