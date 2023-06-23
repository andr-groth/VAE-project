# -*- coding: utf-8 -*-
"""Collection of Tensor mathematical functions.

@author: Andreas Groth
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
from scipy.special import logsumexp

LOG2PI = tf.math.log(2 * np.pi)


def log_density_gaussian(sample: tf.Tensor,
                         mean: tf.Tensor,
                         logvar: tf.Tensor,
                         all_combinations=False,
                         axis=0) -> tf.Tensor:
    """Calculates the sampled log density of a Gaussian.

    Version for Tensor arrays as inputs.

    Parameters:
        sample : Tensor
            Sample at which to compute the density.
        mean : Tensor
            Mean of Gaussian distribution.
        logvar : Tensor
            Log variance of Gaussian distribution.
        all_combinations : bool
            Whether to return values for all combinations of batch pairs of `sample` and `mean`.
        axis : int
            The dimension over which all combinations are computed. Default is 0 meaning the batch dimension.

    Returns : Tensor

    """
    if all_combinations:
        sample = tf.expand_dims(sample, axis=axis + 1)
        mean = tf.expand_dims(mean, axis=axis)
        logvar = tf.expand_dims(logvar, axis=axis)

    log_density = LOG2PI + logvar + tf.exp(-logvar) * (sample - mean)**2
    log_density *= -0.5
    return log_density


def reduce_logmeanexp(input_tensor: tf.Tensor, axis=None, keepdims=False) -> tf.Tensor:
    """Calculates Log(Mean(Exp(x))).

    Version for Tensor arrays as inputs.

    Parameters:
        input_tensor : Tensor
            Input tensor.
        axis : int
            The dimensions to reduce.
        keepdims : bool
            Whether to keep the axis as singleton dimensions.

    Returns:
        Tensor

    """
    lse = tf.reduce_logsumexp(input_tensor, axis=axis, keepdims=keepdims)
    n = tf.size(input_tensor) // tf.size(lse)
    log_n = tf.math.log(tf.cast(n, lse.dtype))
    return lse - log_n


def similarity(input_tensor: tf.Tensor, temperature=1.):
    """Returns the cosine similarity.

    The function returns a similarity measure between all samples in a batch. First, a correlation matrix of the
    normalized input is obtained (the cosine similarity matrix). Then, the similarity measure is defined as the
    categorical cross entropy of the cosine similarity matrix of shape `(batch_size, batch_size)` with the identity
    matrix as target. The correlation can be scaled by `temperature` prior to the calculation of the cross entropy.

    Parameters:
        input_tensor : Tensor
            Input tensor of shape `(batch_size, channels)`.
        temperature: float
            Temperature for the softmax.

    Returns:
        Tensor of shape `(batch_size,)`.

    """
    # normalize input
    l2 = tf.math.l2_normalize(input_tensor, axis=-1)
    # correlation matrix
    similarity = tf.matmul(l2, l2, transpose_b=True)
    # apply temperature
    similarity /= temperature
    # target labels = diagonal elements
    labels = tf.range(tf.shape(similarity)[0])
    # cross entropy loss
    loss = ks.losses.sparse_categorical_crossentropy(labels, similarity, from_logits=True, axis=-1)
    return loss


def log_density_gaussian_numpy(sample: np.ndarray,
                               mean: np.ndarray,
                               logvar: np.ndarray,
                               all_combinations: bool = False) -> np.ndarray:
    """Calculates the sampled log density of a Gaussian.

    Numpy version of :func:`log_density_gaussian`.

    Parameters:
        sample : ndarray
            Array of shape `(batch_size, laten_dim)` at which to compute the density.
        mean : ndarray
            Array of shape `(batch_size, laten_dim)` represnting the mean of Gaussian distribution.
        logvar : ndarray
            Array of shape `(batch_size, laten_dim)` represnting the log variance of Gaussian distribution.
        all_combinations : bool
            Whether to return values for all combinations of batch pairs of `sample` and `mean`.

    Returns : Tensor
        If `all_combinations=True` returns an array of shape `(batch_size, batch_size, latent_dim)`.
        If `all_combinations=False` returns an array of shape `(batch_size, latent_dim)`.

    """
    if all_combinations:
        # reshape to (batch_size, 1, latent_dim)
        sample = np.expand_dims(sample, axis=1)

        # reshape to (1, batch_size, latent_dim)
        mean = np.expand_dims(mean, axis=0)
        logvar = np.expand_dims(logvar, axis=0)

    log_density = np.log(2 * np.pi) + logvar + np.exp(-logvar) * (sample - mean)**2
    log_density *= -0.5
    return log_density


def logmeanexp_numpy(x: np.ndarray, axis: int = None, keepdims: bool = False) -> np.ndarray:
    """Calculates Log(Mean(Exp(x))).

    Numpy version of :func:`logmeanexp`.

    Parameters:
        input_tensor : np.ndarray
            Input array.
        axis : int
            The dimensions to reduce.
        keepdims : bool
            Whether to keep the axis as singleton dimensions.

    Returns:
        np.ndarray

    """
    lse = logsumexp(x, axis=axis, keepdims=keepdims)
    n = np.size(x) // np.size(lse)
    log_n = np.log(n)
    return lse - log_n


def kl_divergence_numpy(mean1: float,
                        logvar1: float,
                        mean2: float = 0,
                        logvar2: float = 0,
                        all_combinations: bool = False) -> np.ndarray:
    """Calculates the KL divergence between two multivariate Gaussian distributions with diagonal covariance matrices.

    The KL divergence is returned for each latent dimensions separately. Since the KL divergence is additive in the case
    of diagonal covariance matrices, the KL divergence can be obtained from the sum over the last dimensions of the
    output.

    Parameters:
        mean1 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the mean of the first Gaussian distribution.
        logvar1 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the log variance of the second Gaussian distribution.
        mean2 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the mean of the second Gaussian distribution. Defaults
            to 0.
        logvar2 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the log variance of the second Gaussian distribution.
            Defaults to 0.
        all_combinations : bool
            Whether to return values for all combinations of batch pairs of `sample` and `mean`.

    Returns : np.ndarray
        If `all_combinations=True` returns an array of shape `(batch_size, batch_size, latent_dim)`.
        If `all_combinations=False` returns an array of shape `(batch_size, latent_dim)`.

    """
    if all_combinations:
        # reshape to (batch_size, 1, latent_dim)
        mean1 = np.expand_dims(mean1, axis=1)
        logvar1 = np.expand_dims(logvar1, axis=1)

        # reshape to (1, batch_size, latent_dim)
        mean2 = np.expand_dims(mean2, axis=0)
        logvar2 = np.expand_dims(logvar2, axis=0)

    kl_div = logvar2 - logvar1 + (np.exp(logvar1) + (mean1 - mean2)**2) * np.exp(-logvar2) - 1
    kl_div *= 0.5
    return kl_div


def wasserstein_distance_numpy(mean1: float,
                               logvar1: float,
                               mean2: float = 0,
                               logvar2: float = 0,
                               all_combinations: bool = False) -> np.ndarray:
    """Calculates the Wasserstein distance between two multivariate Gaussian distributions with diagonal covariance
    matrices.

    The Wasserstein distance is returned for each latent dimensions separately. Since the Wasserstein distance is
    additive in the case of diagonal covariance matrices, the Wasserstein distance can be obtained from the sum over the
    last dimensions of the output.

    Parameters:
        mean1 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the mean of the first Gaussian distribution.
        logvar1 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the log variance of the second Gaussian distribution.
        mean2 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the mean of the second Gaussian distribution. Defaults
            to 0.
        logvar2 : np.ndarray
            Array of shape `(batch_size, laten_dim)` representing the log variance of the second Gaussian distribution.
            Defaults to 0.
        all_combinations : bool
            Whether to return values for all combinations of batch pairs of `sample` and `mean`.

    Returns : np.ndarray
        If `all_combinations=True` returns an array of shape `(batch_size, batch_size, latent_dim)`.
        If `all_combinations=False` returns an array of shape `(batch_size, latent_dim)`.

    """
    if all_combinations:
        # reshape to (batch_size, 1, latent_dim)
        mean1 = np.expand_dims(mean1, axis=1)
        logvar1 = np.expand_dims(logvar1, axis=1)

        # reshape to (1, batch_size, latent_dim)
        mean2 = np.expand_dims(mean2, axis=0)
        logvar2 = np.expand_dims(logvar2, axis=0)

    wd = (mean1 - mean2)**2 + np.exp(logvar1) + np.exp(logvar2) - 2 * np.exp(0.5 * (logvar1 + logvar2))

    return wd
