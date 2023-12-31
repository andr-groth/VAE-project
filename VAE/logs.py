# -*- coding: utf-8 -*-
"""
Collection of log functions for model training.
"""

import tensorflow as tf


def ActiveUnits(z_mean: tf.Tensor, z_log_var: tf.Tensor, kl_threshold: float = 0.1) -> callable:
    """Number of active units.

    Return metric for number of active units that are above the threshold specified in `kl_threshold`.

    Parameters:
        z_mean:
            Output node of encoder, specifying `z_mean`.
        z_log_var:
            Output node of encoder, specifying `z_log_var`.
        kl_threshold:
            Lower bound for the KL divergence above which a latent dimension is assumed to be active.

    Returns:
        Metrics that returns the number of active units.

    """
    def active_units(y_true, y_pred):
        """Prepare function that takes y_true and y_pred as input"""
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss *= -0.5
        kl_loss = tf.reduce_mean(kl_loss, axis=0)
        is_active = tf.greater(kl_loss, kl_threshold)
        is_active = tf.cast(is_active, kl_loss.dtype)

        return tf.reduce_sum(is_active)

    return active_units


def Beta(beta: tf.Tensor) -> callable:
    """Beta value.

    Returns metrics for beta value, which can be added to the model metrics for logging purposes.

    Parameters:
        beta:
            Beta multiplier of the KL divergence.

    Returns:
        Metric function that reduces the beta tensor to a scalar.
    """
    def beta_value(y_true, y_pred):
        return tf.reduce_mean(beta)

    return beta_value
