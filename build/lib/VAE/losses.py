# -*- coding: utf-8 -*-
"""
Collection of loss functions.

@author: Andreas Groth, Imperial College London
"""

from typing import Union

import tensorflow as tf
import tensorflow.keras as ks

from VAE.utils import math as vaemath

# Changelog
# ---------
# 2022-09-01 : added PredictionLoss()
# 2022-09-01 : added Similarity() and SimilarityBetween() losses to VAEloss()
# 2022-06-30 : added free_bits parameter to KLDivergence()
# 2022-06-23 : added z to TotalCorrelation(), TotalCorrelationBetween(), TotalCorrelationWithin()


def KLDivergence(z_mean: tf.Tensor, z_log_var: tf.Tensor, kl_threshold=None, free_bits=None):
    """Kullback-Leibler divergence.

    This is the KL divergence between N(`z_mean`, `z_log_var`) and the prior N(0, 1).

    Parameters:
        z_mean:
            Tensor of shape (batch_size, latent_dim) specifying mean.
        z_log_var:
            Tensor of shape (batch_size, latent_dim) specifying log of variance.
        kl_threshold : float
            Lower bound for the KL divergence. Default is None.
        free_bits : float
            Number of bits to keep free for the KL divergence per latent dimension; cf. Appendix C8 in [1]. Default is
            None.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, 1, 1)`.

    References:
        [1] Kingma et al. (2016): Improved Variational Inference with Inverse Autoregressive Flow. NIPS 2016.

    See also:
        :func:`models.VAE`
            Build and compile variational auto-encoder model.
        :func:`models.Encoder`
            Build encoder model.

    """
    def kl_divergence(y_true, y_pred):
        # KL divergence to N(0, 1) of shape (batch_size, latent_dim)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss *= -0.5

        # apply threshold per latent dimension
        if free_bits is not None:
            kl_loss = tf.maximum(kl_loss, free_bits)

        # reduce to shape (batch_size, 1)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1, keepdims=True)

        # apply global threshold
        if kl_threshold is not None:
            kl_loss = tf.maximum(kl_loss, kl_threshold)

        # expand to shape (batch_size, 1, 1)
        kl_loss = tf.expand_dims(kl_loss, axis=-1)

        return kl_loss

    return kl_divergence


def SquaredError(size=1, taper=None):
    """Squared error loss.

    This is the reconstruction loss of the model, without the KL divergence.

    Parameters:
        size : int
            Size of the model output.
        taper : Numpy array
            Numpy array of length `output_length` to taper the squared error.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    """
    def squared_error(y_true, y_pred):
        # losses reduce last channel dimension to shape (batch_size, set_size, output_length)
        reconstruction_loss = ks.losses.mse(y_true, y_pred)
        if taper is not None:
            reconstruction_loss *= taper
        # scale back to sum of squared errors
        reconstruction_loss *= size
        # further reduce to shape (batch_size, set_size, 1)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=-1, keepdims=True)

        return reconstruction_loss

    return squared_error


def Similarity(temperature=1.):
    """Similarity loss.

    This loss flattens all but the leading batch dimension of `y_pred` to the shape `(batch_size, -1)`. The similarity
    is then calculated of the reshaped input.

    Parameters:
        temperature : float
            Temperature for the softmax.
    Returns:
        Loss function.

    """
    def sim(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        # flatten input to shape (batch_size, -1)
        x = tf.reshape(y_pred, (batch_size, -1))
        # compute similarity loss
        loss = vaemath.similarity(x, temperature=temperature)
        # broadcast loss to shape (batch_size, 1, 1)
        loss = tf.reshape(loss, (-1, 1, 1))

        return loss

    return sim


def _SimilarityBetween(repeat_samples=1, temperature=1.):
    """Similarity between repeated samples.

    This function is the same as `SimilarityBetween` except that it uses :func:`math.similarity` with :func:`tf.map_fn`
    The latter function is very slow and is only left here for illustrative purposes of the einsum implementation in
    `SimilarityBetween`.

    Parameters:
        repeat_samples : int
            Number of repeated samples.
        temperature : float
            Temperature for the softmax.
    Returns:
        Loss function.

    """
    def fn(x):
        return vaemath.similarity(x, temperature=temperature)

    def sim_between(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        # reshape to (batch_size // repreat_samples, repeat_samples, -1)
        inputs = tf.reshape(y_pred, (batch_size // repeat_samples, repeat_samples, -1))
        # similarities over slices of shape (repeat_samples, -1)
        # stacked into tensor of shape (batch_size // repeat_samples, repeat_samples)
        loss = tf.map_fn(fn, inputs)
        # reshape loss to shape (batch_size, 1, 1)
        loss = tf.reshape(loss, (-1, 1, 1))

        return loss

    return sim_between


def SimilarityBetween(repeat_samples=1, temperature=1.):
    """Similarity between repeated samples (fast implementation).

    This function returns the similarity between repeated samples. The input `y_pred` is first reshaped to the shape
    `(batch_size // repeat_samples, repeat_samples, ...)` and the similarity is calculated for each slice along the
    first dimension.

    Note: This is an implementation with einsum that avoids calling :func:`math.similarity` with :func:`tf.map_fn`.

    Parameters:
        repeat_samples : int
            Number of repeated samples.
        temperature : float
            Temperature for the softmax.
    Returns:
        Loss function.

    """
    def sim_between(y_true, y_pred):
        batch_size = tf.shape(y_pred)[0]
        # reshape to (batch_size // repreat_samples, repeat_samples, -1)
        inputs = tf.reshape(y_pred, (batch_size // repeat_samples, repeat_samples, -1))
        # normalize input
        l2 = tf.math.l2_normalize(inputs, axis=-1)
        # correlation matrices of slices along first axis, each of shape (repeat_samples, -1)
        # shape (batch_size // repreat_samples, repeat_samples, repeat_samples)
        similarity = tf.einsum('ijk, ilk -> ijl', l2, l2)
        # reshape to (batch_size, repeat_samples)
        similarity = tf.reshape(similarity, (-1, repeat_samples))
        # apply temperature
        similarity /= temperature
        # target labels = diagonal elements
        labels = tf.tile(tf.range(repeat_samples), (batch_size // repeat_samples, ))
        # cross entropy loss between target labels and similarity matrices
        loss = ks.losses.sparse_categorical_crossentropy(labels, similarity, from_logits=True, axis=-1)

        # broadcast to shape (batch_size, 1, 1)
        loss = tf.reshape(loss, (-1, 1, 1))

        return loss

    return sim_between


def TotalCorrelation(z: tf.Tensor, z_mean: tf.Tensor, z_log_var: tf.Tensor):
    """Total correlation.

    The total correlation is already part of the KL divergence, see KL decomposition in [1]. This function only returns
    the batch-wise sampled total correlation loss that will be added on top of the KL divergence. The total correlation
    is computed separately for each step along the axis of length `set_size` of the input tensor `z`.

     Parameters:
        z : Tensor
            Sample from latent space of shape `(batch_size, set_size, latent_dim)`.
        z_mean : Tensor
            Mean of latent space of shape `(batch_size, latent_dim)`.
        z_log_var : Tensor
            Log of variance of latent space of shape `(batch_size, latent_dim)`.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    References:
        [1] Chen et al. (2018): Isolating sources of disentanglement in Variational Autoencoders.


     See also:
        :func:`models.VAE`
            Build and compile variational auto-encoder model.
        :func:`models.Encoder`
            Build encoder model.

    """
    # expand to shape (batch_size, 1, latent_dim) for broadcasting with z of shape (batch_size, set_size, latent_dim)
    z_mean = tf.expand_dims(z_mean, axis=1)
    z_log_var = tf.expand_dims(z_log_var, axis=1)

    def tc(y_true, y_pred) -> tf.Tensor:
        # log prob of all combinations along first axis of length batch_size
        # shape (batch_size, batch_size, set_size, latent_dim)
        mat_log_qz = vaemath.log_density_gaussian(z, z_mean, z_log_var, all_combinations=True, axis=0)

        # log prob of joint distribution of shape (batch_size, set_size, 1)
        log_qz = vaemath.reduce_logmeanexp(tf.reduce_sum(mat_log_qz, axis=-1, keepdims=True), axis=1)

        # log prob of product of marginal distributions of shape (batch_size, set_size, 1)
        log_prod_qz = tf.reduce_sum(vaemath.reduce_logmeanexp(mat_log_qz, axis=1), axis=-1, keepdims=True)

        # total correlation loss of shape (batch_size, set_size, 1)
        tc_loss = log_qz - log_prod_qz

        return tc_loss

    return tc


def TotalCorrelationBetween(z: tf.Tensor, z_mean: tf.Tensor, z_log_var: tf.Tensor, repeat_samples=1):
    """Total correlation between repeated samples.

    Returns the total correlation loss between repeated samples. This is the same as the total correlation but
    restricted to the same repeated samples. This means that `z_mean` and `z_log_var` are split into segments of length
    `repeat_samples` along the first axis and the total correlation is computed within each segment.

    This version of the total correlation is useful for the case where the model is trained with repeated input samples.
    It helps increase the diversity of the latent distribution between repeated samples.

    Parameters:
        z : tf.Tensor
            Sample from latent space of shape `(batch_size, set_size, latent_dim)`.
        z_mean : Tensor
            Mean of latent space of shape `(batch_size, latent_dim)`.
        z_log_var : Tensor
            Log of variance of latent space of shape `(batch_size, latent_dim)`.
        repeat_samples : int
            Number of repeated samples.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    """
    # reshape z_mean and z_log_var to shape (batch_size // repeat_samples, repeat_samples, 1, latent_dim)
    z_mean = tf.reshape(z_mean, (-1, repeat_samples, 1, z_mean.shape[-1]))
    z_log_var = tf.reshape(z_log_var, (-1, repeat_samples, 1, z_log_var.shape[-1]))

    # reshape z to shape (batch_size // repeat_samples, repeat_samples, set_size, latent_dim)
    z = tf.reshape(z, (-1, repeat_samples, z.shape[-2], z.shape[-1]))

    def tc_between(y_true, y_pred):
        # log prob of all combinations along second axis of size repeat_samples
        # shape (batch_size // repeat_samples, repeat_samples, repeat_samples, set_size, latent_dim)
        mat_log_qz = vaemath.log_density_gaussian(z, z_mean, z_log_var, all_combinations=True, axis=1)

        # log prob of joint distribution, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        log_qz = vaemath.reduce_logmeanexp(tf.reduce_sum(mat_log_qz, axis=-1, keepdims=True), axis=2)

        # log prob of product of marg. distribution, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        log_prod_qz = tf.reduce_sum(vaemath.reduce_logmeanexp(mat_log_qz, axis=2), axis=-1, keepdims=True)

        # total correlation loss, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        tc_loss = log_qz - log_prod_qz

        # reshape to shape (batch_size, set_size, 1)
        tc_loss = tf.reshape(tc_loss, (-1, tc_loss.shape[-2], 1))

        return tc_loss

    return tc_between


def TotalCorrelationWithin(z: tf.Tensor, z_mean: tf.Tensor, z_log_var: tf.Tensor, repeat_samples=1):
    """Total correlation within repeated samples.

    Returns the total correlation loss within all samples of same repetition. This is the same as the total correlation
    but restricted to samples of the same repetition. This means that `z_mean` and `z_log_var` are split into strided
    views with stride `repeat_samples` along the first axis and the total correlation is computed within each view.

    Parameters:
        z : tf.Tensor
            Sample from latent space of shape `(batch_size, set_size, latent_dim)`.
        z_mean : Tensor
            Mean of latent space of shape `(batch_size, latent_dim)`.
        z_log_var : Tensor
            Log of variance of latent space of shape `(batch_size, latent_dim)`.
        repeat_samples : int
            Number of repeated samples.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    """
    # reshape z_mean and z_log_var to shape (batch_size // repeat_samples, repeat_samples, 1, latent_dim)
    z_mean = tf.reshape(z_mean, (-1, repeat_samples, 1, z_mean.shape[-1]))
    z_log_var = tf.reshape(z_log_var, (-1, repeat_samples, 1, z_log_var.shape[-1]))

    # reshape z to shape (batch_size // repeat_samples, repeat_samples, set_size, latent_dim)
    z = tf.reshape(z, (-1, repeat_samples, z.shape[-2], z.shape[-1]))

    def tc_within(y_true, y_pred):
        # log prob of all combinations along first axis of size batch_size // repeat_samples
        # shape (batch_size // repeat_samples, batch_size // repeat_samples, repeat_samples, set_size, latent_dim)
        mat_log_qz = vaemath.log_density_gaussian(z, z_mean, z_log_var, all_combinations=True, axis=0)

        # log prob of joint distribution, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        log_qz = vaemath.reduce_logmeanexp(tf.reduce_sum(mat_log_qz, axis=-1, keepdims=True), axis=1)

        # log prob of product of marg. distribution, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        log_prod_qz = tf.reduce_sum(vaemath.reduce_logmeanexp(mat_log_qz, axis=1), axis=-1, keepdims=True)

        # total correlation loss, shape (batch_size // repeat_samples, repeat_samples, set_size, 1)
        tc_loss = log_qz - log_prod_qz

        # revert shape to (batch_size, set_size, 1)
        tc_loss = tf.reshape(tc_loss, (-1, tc_loss.shape[-2], 1))

        return tc_loss

    return tc_within


def VAEloss(z: tf.Tensor,
            z_mean: tf.Tensor,
            z_log_var: tf.Tensor,
            beta: Union[float, tf.Tensor] = 1.,
            size=1,
            gamma=0.,
            gamma_between=0.,
            gamma_within=0.,
            delta=0.,
            delta_between=0.,
            kl_threshold: float = None,
            free_bits: float = None,
            repeat_samples=1,
            taper=None):
    """Variational auto-encoder loss function.

    Loss function of variational auto-encoder, for use in :func:`models.VAE`. The input to the loss function has shape
    `(batch_size, set_size, output_length, channels)`. The output of the loss function has shape `(batch_size, set_size,
    1)`. The sample weights from the generator must have shape `(batch_size, set_size)`. This will make the sample
    weights sample-dependent; see also `sample_weight_mode='temporal'` in model compile.

    Parameters:
        z : tf.Tensor
            Sample from latent space of shape `(batch_size, set_size, latent_dim)`.
        z_mean : Tensor
            Mean of latent space of shape `(batch_size, latent_dim)`.
        z_log_var : Tensor
            Log of variance of latent space of shape `(batch_size, latent_dim)`.
        size : int
            Size of decoder output, i.e. total number of elements.
        beta : float or Tensor
            Loss weight of the KL divergence. If `beta` is a float, the loss weight is constant. If `beta` is a
            tensor, it should have shape `(batch_size, 1)`.
        gamma : float (optional)
            Scale of total correlation loss.
        gamma_between : float (optional)
            Scale of total correlation loss between repeated samples.
        gamma_within : float (optional)
            Scale of total correlation loss within repeated samples.
        delta : float (optional)
            Scale of similarity loss
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        kl_threshold : float (optional)
            Lower bound for the KL divergence; cf. Appendix C8 in [1].
        free_bits : float (optional)
            Number of bits to keep free for the KL divergence per latent dimension.
        repeat_samples : int (optional)
            Number of repetitions of input samples present in the batch.
        taper : Numpy array (optional)
            Numpy array of length `output_length` to taper the squared error.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    See also:
        :func:`models.VAE`
            Build and compile variational auto-encoder model.
        :func:`models.Encoder`
            Build encoder model.

    """
    if isinstance(beta, tf.Tensor):
        # add singleton dimension to beta to shape (batch_size, 1, 1)
        beta = tf.expand_dims(beta, axis=-1)

    def vae_loss(y_true, y_pred):
        squared_error_fcn = SquaredError(size=size, taper=taper)
        squared_error = squared_error_fcn(y_true, y_pred)

        kl_loss_fcn = KLDivergence(z_mean, z_log_var, kl_threshold=kl_threshold, free_bits=free_bits)
        entropy = kl_loss_fcn(y_true, y_pred)

        if gamma:
            tc_loss_fcn = TotalCorrelation(z, z_mean, z_log_var)
            entropy += gamma * tc_loss_fcn(y_true, y_pred)

        if gamma_between:
            tc_between_loss_fcn = TotalCorrelationBetween(z, z_mean, z_log_var, repeat_samples=repeat_samples)
            entropy += gamma_between * tc_between_loss_fcn(y_true, y_pred)

        if gamma_within:
            tc_within_loss_fcn = TotalCorrelationWithin(z, z_mean, z_log_var, repeat_samples=repeat_samples)
            entropy += gamma_within * tc_within_loss_fcn(y_true, y_pred)

        if delta:
            sim_loss_fcn = Similarity()
            entropy += delta * sim_loss_fcn(y_true, y_pred)

        if delta_between:
            sim_between_loss_fcn = SimilarityBetween(repeat_samples=repeat_samples)
            entropy += delta * sim_between_loss_fcn(y_true, y_pred)

        return squared_error + beta * entropy

    return vae_loss


def VAEploss(beta: Union[float, tf.Tensor] = 1., delta=0., delta_between=0., repeat_samples=1, size=1, taper=None):
    """VAE prediction loss function.

    Parameters:
        beta : float or Tensor
            Loss weight of the KL divergence. If `beta` is a float, the loss weight is constant. If `beta` is a
            tensor, it should have shape `(batch_size, 1)`.
        delta : float (optional)
            Scale of similarity loss
        delta_between : float (optional)
            Scale of similarity loss between repeated samples.
        size : int
            Size of prediction output, i.e. total number of elements.
        repeat_samples : int (optional)
            Number of repetitions of input samples present in the batch.
        taper : Numpy array (optional)
            Numpy array of length `output_length` to taper the squared error.

    Returns:
        Loss function that returns a tensor of shape `(batch_size, set_size, 1)`.

    See also:
        :func:`models.VAEp`
            Build and compile variational auto-encoder model.

    """
    if isinstance(beta, tf.Tensor):
        # add singleton dimension to beta to shape (batch_size, 1, 1)
        beta = tf.expand_dims(beta, axis=-1)

    def vaep_loss(y_true, y_pred):
        squared_error_fcn = SquaredError(size=size, taper=taper)
        squared_error = squared_error_fcn(y_true, y_pred)

        entropy = tf.zeros_like(squared_error)

        if delta:
            sim_loss_fcn = Similarity()
            entropy += delta * sim_loss_fcn(y_true, y_pred)

        if delta_between:
            sim_between_loss_fcn = SimilarityBetween(repeat_samples=repeat_samples)
            entropy += delta_between * sim_between_loss_fcn(y_true, y_pred)

        return squared_error + beta * entropy

    return vaep_loss


def example_total_correlation_losses():
    """Example of total correlation loss functions."""
    batch_size = 32
    repeat_samples = 20
    shape = (batch_size * repeat_samples, 8)
    set_size = 7

    z_mean = tf.random.normal(shape)
    z_log_var = tf.random.normal(shape) * 0.1 - 1.
    z = z_mean + tf.exp(z_log_var * 0.5) * tf.random.normal(shape)
    z = tf.expand_dims(z, axis=1)
    z = tf.repeat(z, repeats=set_size, axis=1)

    fcns = {
        'TC loss': TotalCorrelation(z, z_mean, z_log_var),
        'TC loss between': TotalCorrelationBetween(z, z_mean, z_log_var, repeat_samples=repeat_samples),
        'TC loss within': TotalCorrelationWithin(z, z_mean, z_log_var, repeat_samples=repeat_samples),
    }

    print(f'{"Batch size":<20} {batch_size} * {repeat_samples} = {batch_size * repeat_samples}')

    for name, fcn in fcns.items():
        tc_loss = fcn(None, None)
        tc_mean = tf.reduce_mean(tc_loss)
        tc_std = tf.math.reduce_std(tc_loss)
        print(f'{name:<20} mean={tc_mean:.2f}  std={tc_std:.2f}  shape={tc_loss.shape}')


def example_similarity_losses():
    """Example of similarity loss functions."""
    batch_size = 32
    repeat_samples = 5
    shape = (batch_size * repeat_samples, 1, 160, 3)

    inputs = tf.random.normal(shape)

    fcns = {
        'Sim loss': Similarity(),
        'Sim loss between': _SimilarityBetween(repeat_samples=repeat_samples),
        'Sim loss between (fast)': SimilarityBetween(repeat_samples=repeat_samples),
    }

    print(f'{"Batch size":<25} {batch_size} * {repeat_samples} = {batch_size * repeat_samples}')

    losses = []
    for name, fcn in fcns.items():
        loss = fcn(None, inputs)
        losses.append(loss)
        mean_loss = tf.reduce_mean(loss)
        std_loss = tf.math.reduce_std(loss)
        print(f'{name:<25} mean={mean_loss:.2f}  std={std_loss:.2f}  shape={loss.shape}')

    return losses


if __name__ == '__main__':
    # example_total_correlation_losses()
    losses = example_similarity_losses()
