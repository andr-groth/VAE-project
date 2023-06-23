# -*- coding: utf-8 -*-
"""Many-to-many version of variational auto-encoder.

Model specifications:
    - Multiple samples (set_size) as time distributed layers.
    - Input shape to encoder (set_size, input_length, channels).
    - Output shape of latent_sampling (set_size, latent_dim,): independent random samples for each member in set
    - Input shape to decoder (set_size, latent_dim,)
    - Masked multihead attention over filters x set_size. Temporal mask along dimension of size input_length.
    - Residual blocks with causal convolution with pre-activation and pixel-shuffling for downsampling and upsampling
    - Decoder with internal padding blocks
    - 3 FiLM blocks:
        encoder: after input expansion, after encoder blocks, and after attention,
        decoder: after input expansion, after sttention blocks, and decoder blocks

@author: Andreas Groth, Imperial College London

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

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from VAE import layers as vaelayers
from VAE import logs as vaelogs
from VAE import losses as vaelosses
from VAE.utils import collection


def Encoder(input_shape,
            latent_dim,
            set_size=1,
            set_noise=True,
            encoder_blocks=0,
            residual_units=1,
            activation=None,
            depth_wise=False,
            filters=1,
            kernel_size=2,
            ratio=1,
            attn_activation='softmax',
            attn_blocks=0,
            attn_filters='full',
            attn_heads=1,
            attn_masked=True,
            attn_transposed=False,
            cond_activation=None,
            cond_size=None,
            cond_units=None,
            cond_use_bias=True,
            cond_use_scale=True,
            cond_use_offset=True,
            cond_ens_size=None,
            cond_temperature=1.,
            ens_film_size=None,
            fc_activation=None,
            fc_units=None,
            film_activation=None,
            film_temporal=False,
            film_use_bias=True,
            film_use_scale=True,
            film_use_offset=True,
            pooling='reduce_sum',
            name='encoder',
            **kwargs) -> ks.Model:
    """Encoder model.

    This function creates an encoder model, for use in :func:`build_vae`, for example. This model
    takes multiple samples as input and returns a single `z_mean` and `z_log_var`. The input to the model has shape
    `(batch_size, set_size, input_length, channels)` and the output has shape `(batch_size, latent_dim)`.

    Parameters:
        input_shape : tuple of two integers
            Input shape of the encoder. A tuple of two integers `(input_length, channels)`, with `input_length` being
            the length of the input sequence and `channels` being the number of input channels. If the input length is
            not a power of 2, the input will be zero-padded to the next power of 2.
        latent_dim : int
            Size of the latent space.
        set_size : int
            Number of samples in a set. This is the leading dimension, after the batch dimension.
        encoder_blocks : int
            Number of encoder blocks. Defaults to 0.
        residual_units : int
            Number of residual units in each of the encoder blocks.
        filters : int
            Number of filters that the input is expanded to.
        activation: str
            Activation function for convolutions in residual units.
        kernel_size : int or tuple of ints
            Kernel size of convolutions in residual units. If a tuple is given, the kernel size can be different for the
            different encoder blocks. In this case, the length of the tuple must be equal to `encoder_blocks`.
        ratio : int
            Compression ratio of filters in the residual units. Defaults to 1.

        attn_activation : str
            Name of activation function for the scores in the attention.
        attn_blocks : int
            Number of attention blocks. Defaults to 0.
        attn_filters : int or str
            Number of attention filters.
        attn_heads : int
            Number of attention heads.
        attn_masked : bool
            Whether apply a causal mask to the scores in the attention layer. Defaults to `True`.
        attn_transposed : bool
            Whether the attention is applied to the spatial dimension (`False`) or the channel dimension (`True`).

        cond_size : int or list of two ints
            Size of conditional input. If `cond_size` is an integer, the conditional input is passed to a regular dense
            layer. If `cond_size` is a list of two integers, the conditional input of size `sum(cond_size)` is split
            along the channel axis into two tensors of size `cond_size[0]` and `cond_size[1]`, providing the condition
            and the ensemble id. The condition is passed to (set) of dense layer(s), while the ensemble id is passed to
            a GumbelSoftmax layer.
        cond_activation : str or list of str
            Name of the activation function for dense layer(s).
        cond_units : int or list of int
            Number of units for dense layer(s) applied to the input condition.  Set to `None` to disable dense layer.
        cond_use_bias : bool or list of bool
            Whether to use bias in the input condition. Defaults to `True`.
        cond_use_scale : bool
            Whether to use scale in the DenseFilmed unit. Defaults to `True`.
        cond_use_offset : bool
            Whether to use offset in the DenseFilmed unit. Defaults to `True`.
        cond_ens_size : int
            Size of the ensemble of GumbelSoftmax. Defaults to `None`.
        cond_temperature : float
            Temperature of the GumbelSoftmax. Defaults to 1.

        ens_film_size : int
            Size of random ensemble in EnsembleFilm layer. Set to `None` to disable EnsembleFilm layer.

        fc_units : int
            Number of units in last dense layer, i.e. before sampling of `z_mean` and `z_log_var`. Set to `None` to
            disable the dense layer.
        fc_activation : str
            Activation function of last dense layer.

        film_activation : str
            Name of the activation function for the scale in the Film layers. Defaults to `None`.
        film_temporal : bool
            Whether the Film layers are time-dependent (apply to the second last dimension). Defaults to `False`.
        film_use_offset : bool
            Whether to use offset in the Film layers. Defaults to `True`.
        film_use_scale : bool
            Whether to use scale in the Film layers. Defaults to `True`.
        film_use_bias : bool
            Whether to use bias in the Film layers. Defaults to `True`.

        pooling : str
            Pooling function. Set to `None` to disable pooling.
        name : str
            Name of the encoder model. Must be one of the following: `encoder`.

    Returns:
        Instance of :class:`ks.Model`.

    Model inputs:
        If cond_size is None:
            Tensor of shape `(set_size, input_length, channels)`.
        If cond_size is not None:
            Tuple of two tensors of shape `(set_size, input_length, channels)` and `(set_size, cond_size)`.

    Model outputs:
        Tensor of shape `(latent_dim, )`

    See also:
        :func:`build_vae`:
            Build and compile variational auto-encoder model.
        :func:`build_encoder_block`:
            Build encoder-block model.
        :func:`residual_unit`:
            Build residual unit.
        :func:`build_multihead_attention_block`:
            Build multihead-attention block model.

    """
    if name not in {'encoder'}:
        raise ValueError("Name must be one of the following: 'encoder'.")

    kernel_size = (kernel_size, ) * encoder_blocks if isinstance(kernel_size, int) else kernel_size
    if len(kernel_size) != encoder_blocks:
        raise ValueError("Length of kernel_size must be equal to encoder_blocks.")

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

    # Random ensemble FiLM
    if ens_film_size:
        # add singleton dimension to the shape for set_size
        y = vaelayers._EnsembleFilm(ensemble_size=ens_film_size, shape=(1, ) + film_shape, name=name + '_ens_film')(y)

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


def Decoder(output_shape,
            latent_dim,
            set_size=1,
            set_noise=True,
            decoder_blocks=0,
            residual_units=1,
            activation=None,
            depth_wise=False,
            filters=1,
            kernel_size=2,
            ratio=1,
            attn_activation='softmax',
            attn_blocks=0,
            attn_filters='full',
            attn_heads=1,
            attn_masked=True,
            attn_transposed=False,
            cond_activation=None,
            cond_size=None,
            cond_units=None,
            cond_use_bias=True,
            cond_use_scale=True,
            cond_use_offset=True,
            cond_ens_size=None,
            cond_temperature=1.,
            ens_film_size=None,
            film_activation=None,
            film_temporal=False,
            film_use_bias=True,
            film_use_scale=True,
            film_use_offset=True,
            fc_units=None,
            fc_activation=None,
            padding_blocks=0,
            output_reverse=True,
            name='decoder',
            **kwargs) -> ks.Model:
    """Decoder model.

    This function creates a decoder model, for use in :func:`build_vae`, for example. This model
    takes multiple realizations `z` from the latent space and returns multiple samples. The input shape of the model is
    `(batch_size, set_size, latent_dim)`. The output shape is `(batch_size, set_size, output_length, channels)`.

    Parameters:
        output_shape : tuple of two integers
            Output shape of the decoder. A tuple of two integers `(output_length, channels)`, with `output_length` being
            the length of the output sequence and `channels` being the number of output channels. If `output_length` is
            not a power of 2, the decoder will internally work with a sequence length to the next power of 2 and
            crop the sequence to the desired length.
        latent_dim : int
            Size of the latent space.
        set_size : int
            Number of output samples in a set. This is the leading dimension, after the batch dimension.
        decoder_blocks : int
            Number of decoder blocks. Defaults to 0.
        output_reverse : bool
            Reverse temporal order of output.
        padding_blocks : int
            Number of internal padding blocks. Instead of zero-padding, further padding blocks are drawn from `z`.
            Defaults to 0.
        kernel_size : int or tuple of ints
            Size of the convolutional kernel. If a tuple, the kernel size can be different for each decoder block. In
            this case, the length of the tuple must be equal to `decoder_blocks` and the order is reversed.
        name: str
            Name of the decoder model. Must be one of the following: `decoder`, `prediction`.

        For the rest of the parameters, see :func:`Encoder`.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        If cond_size is None:
            Tensor of shape `(set_size, latent_dim)`.
        If cond_size is not None:
            Tuple of two tensors of shape `(set_size, latent_dim)` and `(set_size, cond_size)`.

    Model outputs:
        Tensor of shape `(set_size, output_length, channels)`.

    See also:
        :func:`build_vae`:
            Build and compile variational auto-encoder model.
        :func:`build_decoder_block`:
            Build decoder-block model.
        :func:`residual_unit`:
            Build residual unit.
        :func:`build_multihead_attention_block`:
            Build multihead-attention block model.

    """
    if name not in {'decoder', 'prediction'}:
        raise ValueError("Name must be one of 'decoder' or 'prediction'.")

    kernel_size = (kernel_size, ) * decoder_blocks if isinstance(kernel_size, int) else kernel_size
    if len(kernel_size) != decoder_blocks:
        raise ValueError("Length of kernel_size must be equal to decoder_blocks.")

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

    # Random ensemble FiLM
    if ens_film_size:
        # add singleton dimension to the shape for set_size
        y = vaelayers._EnsembleFilm(ensemble_size=ens_film_size, shape=(1, ) + film_shape, name=name + '_ens_film')(y)

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


def LatentSampling(latent_dim, set_size=1, set_noise=True, input_sigma=False, name='latent', **kwargs) -> ks.Model:
    """Latent-sampling model.

    This function creates a model for latent sampling, for use in :func:`build_vae`. The  model has input `z_mean`
    and `z_log_var` from the encoder and returns `set_size` random samples `z`. The input shape is
    `(batch_size, latent_dim,)` and the output shape is `(batch_size, set_size, latent_dim)`.

    Parameters:
        latent_dim : int
            Size of the latent space.
        set_size : int
            Number of output samples in a set. This is the leading dimension, after the batch dimension.
        set_noise : bool
            Whether the noise varies between samples in a set.
        input_sigma : bool
            Specify, whether random samples are provided as input to the model.

    Returns:
        Instance of :class:`keras.Model`.

    Model inputs:
        Tensor of shape `(latent_dim,)`.

    Model outputs:
        Tensor of shape `(set_size, latent_dim)`.

    References:
        https://keras.io/examples/variational_autoencoder

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


def EncoderBlock(shape,
                 residual_units,
                 activation=None,
                 depth_wise=False,
                 kernel_size=1,
                 ratio=1,
                 name='encoder_block') -> ks.Model:
    """Encoder-block model.

    This function crates an encoder-block model for the use in  :func:`build_encoder`. The input to
    the block has shape `(set_size, input_length, filters)`, excluding the batch dimension, and the
    output has shape `(set_size, input_length // 2, 2 * filters)`.

    Parameters:
        shape : tuple of int
            Input shape of the block; i.e. `shape = (set_size, input_length, filters)`.
        residuals_units : int
            Number of residual units.
        activation : str
            Activation function.
        kernel_size : int
            Size of the convolutional kernel.
        name : str
            Name of encoder-block model.

    Returns:
        Instance of :class:`keras.Model`.

    See also:
        :func:`build_encoder`:
            Build encoder model.
        :func:`residual_unit`:
            Build residual unit.

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


def DecoderBlock(shape,
                 residual_units,
                 activation=None,
                 depth_wise=False,
                 kernel_size=1,
                 ratio=1,
                 name='decoder_block') -> ks.Model:
    """Decoder-block model.

    This function creates a decoder-block model for the use in  :func:`build_decoder`. The input to
    the block has shape `(set_size, input_length, filters)`, excluding the batch dimension, and the
    output has shape `(set_size, 2 * input_length, filters // 2)`.

    Parameters:
        shape : Tuple
            Input shape of the block; i.e. `shape = (set_size, input_length, filters)`.
        residuals_units : int
            Number of residual units.
        activation : str
            Activation function.
        kernel_size : int
            Size of the convolutional kernel.
        name : str
            Name of decoder-block model.

    Returns:
        Instance of :class:`keras.Model`.

    See also:
        :func:`build_decoder`:
            Build decoder model.
        :func:`residual_unit`:
            Build residual unit.

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


def MLP(units, activation=None, use_bias=None, **kwargs):
    """Multi-layer perceptron.

    This is a convenience function to create a multi-layer perceptron. The parameters can be either single values or
    lists of values. If a list is given, the length of the list determines the number of layers. Otherwise, a single
    dense layer is created.

    Parameters:
        units: int or list of int
            Number of units in each layer.
        activation: str or list of str
            Activation function for each layer.
        use_bias: bool or list of bool
            Whether to use bias in each layer.
        kwargs:
            Additional keyword arguments are passed to the dense layers.

    Inputs:
        Tensor of arbitrary shape.
    Outputs:
        Tensor of same shape as input, except for the last dimension which is specified by the last units parameter.
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


def MultiheadAttentionBlock(shape,
                            filters,
                            heads,
                            masked=True,
                            transposed=False,
                            attn_activation='softmax',
                            res_activation=None,
                            name='mab') -> ks.Model:
    """Multihead-attention model.

    This function creates a model for multihead attention for use in :func:`build_decoder` and
    :func:`build_encoder`. The input shape of the model is `(set_size, input_length, filters)`. The dot-product
    attention is applied over reshaped matrices of shape `(set_size * input_length, filters)` . The score has a shape
    of `(set_size * input_length, set_size * input_length)`. Causal masking is applied to the scores and no temporal
    information merged.

    Parameters:
        shape : tuple
            Input shape of the model.
        filters : int or str
            Number of filters in each of the heads. If 'full', the number of filters will be the same as in the input.
            If 'reduced', the number of filters will be divided by the number of heads.
        heads : int
            Number of heads.
        masked : bool
            Whether apply a causal mask to the score.
        transposed : bool
            Whether the attention is applied to the `set_size` dimension (`False`) or the `filters` dimension (`True`).
        attn_activation : str
            Name of activation function for the scores.
        res_activation : str
            Name of activation function for the residual unit.
        name : str
            Name of the model.

    Returns:
        Instance of :class:`keras.Model`.

    See also:
        :func:`build_decoder`:
            Build decoder model.
        :func:`build_encoder`:
            Build encoder model.
        :class:`VAE.layers.Attention`:
            Dot-product attention layer.

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


def ResidualUnit(filters, activation=None, depth_wise=False, kernel_size=1, ratio=1, name='residual_unit'):
    """Residual unit.

    Residual unit for use in :func:`build_decoder_block` and :func:`build_encoder_block`.

    See also:
        :func:`build_decoder_block`:
            Build decoder block.
        :func:`build_encoder_block`:
            Build encoder block.

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
        beta=1.,
        delta=0.,
        delta_between=0.,
        gamma=0.,
        gamma_between=0.,
        gamma_within=0.,
        repeat_samples=1,
        learning_rate=0.001,
        clipnorm=1.,
        kl_threshold=None,
        free_bits=None,
        taper_range=None,
        trainable=None,
        name='mVAE',
        **kwargs) -> ks.Model:
    """Variational auto-encoder model.

    This function takes a `decoder`, `encoder`, and `latent sampling` model and creates and compiles a variational
    auto-encoder (VAE).

    Parameters:
        encoder : :class:`keras.Model`
            Instance of :class:`keras.Model`, specifying the encoder.
        decoder : :class:`keras.Model`
            Instance of :class:`keras.Model`, specifying the decoder.
        latent_sampling : :class:`keras.Model`
            Instance of :class:`keras.Model`, specifying the latent sampling.
        beta : float/str
            Loss weight of the KL divergence. If `str`, loss weight will be input to the model.
        delta : float (optional)
            Scale of similarity loss.
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        gamma : float
            Scale of total correlation loss. See :func:`VAE.losses.TotalCorrelation` for details.
        gamma_between : float
            Scale of total correlation loss between repeated samples. See :func:`VAE.losses.TotalCorrelationBetween`
            for details.
        gamma_within : float
            Scale of total correlation loss within repeated samples. See :func:`VAE.losses.TotalCorrelationWithin`
            for details.
        repeat_samples : int
            Number of repetitions of input samples. This will be used to compute the total correlation loss.
        learning_rate : float
            Learning rate.
        clipnorm : float
            Gradient clipping norm.
        kl_threshold : float
            Lower bound for the KL divergence.
        free_bits : float
            Number of bits to keep free for the KL divergence per latent dimension.

        taper_range : tuple of two floats
            Start and stop of the linear taper used to scale the squared error loss of the decoder. The taper is
            normalized to have mean 1.
        trainable : iterable of str
            Names of the model's trainable layers. Defaults is None, meaning all layers are trainable.
        name : str
            Name of the model.

    Returns:
        Instance of :class:`keras.Model`.

    See also:
        :func:`build_decoder`:
            Build decoder model.
        :func:`build_encoder`:
            Build encoder model.
        :func:`build_latent_sampling`:
            Build model for latent sampling.
        :func:`build_conditionin`:
            Build model for conditioning.

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
         beta=1.,
         gamma=0.,
         gamma_between=0.,
         gamma_within=0.,
         delta=0.,
         delta_between=0.,
         repeat_samples=1,
         learning_rate=0.001,
         clipnorm=1.,
         kl_threshold=None,
         free_bits=None,
         loss_weights=None,
         taper_ranges=None,
         trainable=None,
         name='mVAEp',
         **kwargs) -> ks.Model:
    """Variational auto-encoder model with prediction.

    This function takes `decoder`, `encoder`, `latent sampling`, and `prediction` models and creates and compiles a
    variational auto-encoder, combined with a decoder for prediction. The `prediction` model is second a decoder, also
    linked to the latent space, but that predicts samples forward in time.

    Parameters:
        encoder : :class:`ks.Model`
            Instance of :class:`keras.Model`, specifying the encoder.
        decoder : :class:`ks.Model`
            Instance of :class:`ks.Model`, specifying the decoder.
        latent_sampling : :class:`ks.Model`
            Instance of :class:`keras.Model`, specifying the latent sampling.
        prediction : :class:`ks.Model`
            Instance of :class:`ks.Model`, specifying the decoder for prediction.
        beta : float or str
            Loss weight of the KL divergence. If `str`, `beta` will be an input to the model.
        gamma : float
            Scale of total correlation loss. See :func:`VAE.losses.TotalCorrelation` for details.
        gamma_between : float
            Scale of total correlation loss between repeated samples. See :func:`VAE.losses.TotalCorrelationBetween`
            for details.
        gamma_within : float
            Scale of total correlation loss within repeated samples. See :func:`VAE.losses.TotalCorrelationWithin`
            for details.
        delta : float (optional)
            Scale of similarity loss.
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        repeat_samples : int
            Number of repetitions of input samples. This will be used to compute the total correlation loss on block
            diagonal elements.
        learning_rate : float
            Learning rate.
        clipnorm : float
            Gradient clipping norm.
        kl_threshold : float
            Lower bound for the KL divergence.
        free_bits : float
            Number of bits to keep free for the KL divergence per latent dimension.
        loss_weights : dict
            Loss weights of the different model outputs or the decoder and  prediction. The dict maps outputs names
            to scalar coefficients.
        taper_ranges : dict
            Start and stop values for taper of the decoder and prediction. The dict maps output names to tuples of two
            floats specifying the start and stop values of the linear taper. The tapers are normalized to have mean 1.
            To change the weights between decoder and prediction, use the `loss_weights` argument.
        trainable : iterable of str
            Names of the model's trainable layers. Defaults is None, meaning all layers are trainable.
        name: str
            Name of the model.

    Returns:
        Instance of :class:`keras.Model`.

    See also:
        :func:`build_decoder`:
            Build decoder model.
        :func:`build_encoder`:
            Build encoder model.
        :func:`build_latent_sampling`:
            Build model for latent sampling.
        :func:`build_vae`:
            Build and compile variational auto-encoder model.

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


def IdentityModel(input_shape, set_size=1) -> ks.Model:
    """Identity model.

    This function creates an identity model, which returns the input as output.

    Parameters:
        input_shape : tuple
            Tuple of two integers, specifying the input shape of the model.
        set_size : int
            Number of input samples in a set.

    Returns:
        Instance of :class:`keras.Model`.

    """
    xin = ks.layers.Input((set_size, ) + tuple(input_shape), name='encoder_input')
    y = ks.layers.Lambda(lambda x: x, name='decoder')(xin)
    model = ks.Model(xin, y, name='Identity')
    return model


# Example model
if __name__ == '__main__':
    K.clear_session()

    params = {
        'cond_ens_size': 6,
        'cond_size': [12, 38],
        'cond_units': 12,
        'cond_use_scale': False,
        'delta': 1,
        'delta_between': 1,
        'encoder_blocks': 3,
        'fc_units': 48,
        'film_temporal': True,
        'filters': 14,
        'input_shape': [16, 7],
        'kernel_size': 2,
        'latent_dim': 10,
        'padding_blocks': 1,
        'prediction_shape': [16, 1],
        'residual_units': 1,
        'set_size': 1,
        'trainable': ['*cond*', '*bn*'],
    }

    # Build encoder model
    encoder = Encoder(**params, name='encoder')

    # Build model for latent sampling
    latent_sampling = LatentSampling(**params, name='latent')

    # Build decoder model
    decoder = Decoder(output_shape=params['input_shape'],
                      decoder_blocks=params['encoder_blocks'],
                      output_reverse=True,
                      **params,
                      name='decoder')

    # Build decoder model for prediction
    prediction = Decoder(output_shape=params['prediction_shape'],
                         decoder_blocks=params['encoder_blocks'],
                         output_reverse=False,
                         **params,
                         name='prediction')

    # build variational Auto-Encoder
    # model = build_vae(encoder, decoder, latent_sampling)

    # build VAE + prediction
    model = VAEp(encoder, decoder, latent_sampling, prediction, **params)
    # model.summary()
    encoder.summary()
    # decoder.summary()

    # collection.summary_trainable(model)
    # ks.utils.plot_model(encoder, show_shapes=True)

    # dict of shapes of model weights
    # w_shapes = {w.name: w.shape.as_list() for w in sorted(encoder.weights, key=lambda w: w.name)}
