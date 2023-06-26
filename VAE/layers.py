# -*- coding: utf-8 -*-
"""
Collection of Keras layers.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

from VAE.utils.collection import complete_shape


class Attention(ks.layers.Layer):
    """Dot-product attention layer.

    This layers builds on the attention layer :class:`keras.layers.Attention`. The matrix multiplication in the
    attention is applied to the two inner dimensions and broadcasted otherwise. Use `permute` to reorder the input
    before attention. The output has again the same order of dimenions as the input.

    Parameters:
        activation:
            Name of activation function applied to the score. Defaults to 'softmax'.
        permute:
            Permutation applied to the dimensions of the input tensors before attention. The output has the same order
            of dimenions as the input. Defaults to `None`, meaning that no permutation is applied.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    """
    def __init__(self, activation: str = 'softmax', permute: list[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.activation = ks.activations.get(activation)
        self.permute = permute
        self.ipermute = None

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Call the layer.

        Parameters:
            inputs:
                Tuple of three tensors representing the query, key, and value, respectively. The shapes of the three
                tensors is `(batch_size, set_size, time_length, channels)`.

        Returns:
            Tensor of shape `(batch_size, set_size, time_length, channels)`.

        """
        q, k, v = inputs

        if self.permute is not None:
            q = tf.transpose(q, perm=self.permute)
            k = tf.transpose(k, perm=self.permute)
            v = tf.transpose(v, perm=self.permute)

        qk = tf.matmul(q, k, transpose_b=True)
        qk = self.activation(qk)
        qkv = tf.matmul(qk, v)

        if self.ipermute is not None:
            qkv = tf.transpose(qkv, perm=self.ipermute)

        return qkv

    def build(self, input_shape):
        assert len(input_shape) == 3, "List of three input elements expected."

        # get inverse permutation of `permute`
        if self.permute is not None:
            assert len(self.permute) == len(input_shape[0]), "Length of `permute` must match number of input dimensions"
            ipermute = list(self.permute)
            for i in range(len(self.permute)):
                ipermute[self.permute[i]] = i

            self.permute = tuple(self.permute)
            self.ipermute = tuple(ipermute)

        self.built = True

    def compute_output_shape(self, input_shape):
        return tf.stack(input_shape[0][:-1], input_shape[-1][-1])

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': ks.activations.serialize(self.activation),
            'permute': self.permute,
            'ipermute': self.ipermute,
        })
        return config


class AttentionMasked(ks.layers.Layer):
    """Dot-product attention layer.

    This layers builds on the attention layer :class:`keras.layers.Attention`. Each input tensor of shape `(batch_size,
    set_size, time_length, channels)` is first reshaped into a tensor of shape `(batch_size, set_size * time_length,
    channels)`. The dot-product attention is applied on the last dimension. If `masked=True`, score values with a query
    time index larger than the key time index are masked out.

    Parameters:
        activation:
            Activation function applied to the scores. Defaults to 'softmax'.
        masked:
            Whether a causal masked is applied to the scores. Defaults to `True`.
        permute:
            More general permutation of the dimensions of the input tensors before attention.  Note that the time index
            of the causal mask always refers to the second dimension of the permuted dimensions. Defaults to `None`,
            meaning that no permutation is applied.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    """
    def __init__(self,
                 activation: str = 'softmax',
                 masked: bool = True,
                 permute: tuple[int, int, int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.activation = ks.activations.get(activation)
        self.masked = masked
        self.permute = permute

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Apply the the dot-product attention to the inputs.

        Parameters:
            inputs:
                Tuple of three tensors representing the query, key, and value, respectively. The shapes of the three
                tensors is `(batch_size, set_size, time_length, channels)`.

        Returns:
            Tensor of shape `(batch_size, set_size, time_length, channels)`.

        """
        q, k, v = inputs

        # permute dimensions
        if self.permute is not None:
            q = tf.transpose(q, perm=self.permute)
            k = tf.transpose(k, perm=self.permute)
            v = tf.transpose(v, perm=self.permute)

        q_shape = tf.shape(q)
        k_shape = tf.shape(k)
        v_shape = tf.shape(v)

        q = tf.reshape(q, shape=(-1, q_shape[1] * q_shape[2], q_shape[3]))
        k = tf.reshape(k, shape=(-1, k_shape[1] * k_shape[2], k_shape[3]))
        v = tf.reshape(v, shape=(-1, v_shape[1] * v_shape[2], v_shape[3]))

        # get dot product of queries and keys
        qk = tf.matmul(q, k, transpose_b=True)
        # scale
        qk /= tf.sqrt(tf.cast(q_shape[-1], dtype=qk.dtype))
        # apply mask to dot product: set masked positions to large negative values results in near zero activation
        if self.mask is not None:
            qk -= 1e9 * self.mask
        # activation
        qk = self.activation(qk)

        # get output of attention
        qkv = tf.matmul(qk, v)

        # reshape output
        qkv = tf.reshape(qkv, shape=v_shape)

        # restore order of dimensions in output
        if self.ipermute is not None:
            qkv = tf.transpose(qkv, perm=self.ipermute)

        return qkv

    def build(self, input_shape):
        assert len(input_shape) == 3, "List of three input elements expected."

        # get inverse permutation of `permute`
        if self.permute is not None:
            assert len(self.permute) == len(input_shape[0]), "Length of `permute` must match number of input dimensions"
            ipermute = list(self.permute)
            for i in range(len(self.permute)):
                ipermute[self.permute[i]] = i

            self.permute = tuple(self.permute)
            self.ipermute = tuple(ipermute)
        else:
            self.ipermute = None

        # get mask
        if self.masked:
            input_shape_as_list = [shape.as_list() for shape in input_shape]

            if self.permute is None:
                self.mask = self._get_causal_mask(input_shape_as_list)
            else:
                input_shape_permute = [[shape[i] for i in self.permute] for shape in input_shape_as_list]
                self.mask = self._get_causal_mask(input_shape_permute)
        else:
            self.mask = None

        self.built = True

    def compute_output_shape(self, input_shape):
        q_shape, _, v_shape = input_shape
        return tf.stack(q_shape[:-1], v_shape[-1])

    def _get_causal_mask(self, input_shape):
        """Get causal mask for score."""
        q_shape, k_shape, _ = input_shape
        _, q_set_size, q_input_length, _ = q_shape
        _, k_set_size, k_input_length, _ = k_shape

        # time index of query tensor (second axis)
        q_time_index = np.arange(q_input_length)
        q_time_index = np.repeat(q_time_index[None, :], q_set_size, axis=0)
        # flatten set_size and input_length dimensions as in attention
        q_time_index = q_time_index.flatten()

        # time index of key tensor (second axis)
        k_time_index = np.arange(k_input_length)
        k_time_index = np.repeat(k_time_index[None, :], k_set_size, axis=0)
        # flatten set_size and input_length dimensions as in attention
        k_time_index = k_time_index.flatten()

        # causal mask for score. set mask=1 where query time index is before key time index
        mask = q_time_index[:, None] < k_time_index[None, :]

        return tf.constant(mask, dtype=K.floatx())

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': ks.activations.serialize(self.activation),
            'masked': self.masked,
            'permute': self.permute,
            'ipermute': self.ipermute,
        })
        return config


class _Film_v1(ks.layers.Layer):
    """Feature-wise linear modulation (version 1).

    Feature-wise linear modulation of the last dimension of the input tensor.

    A linear layer is applied to the last dimension of the condition tensor and the output gives the scale that is
    multiplied with the input tensor. The scale can be activated by a non-linearity prior to the multiplication.

    Another linear layer is applied to the last dimension of the condition tensor and the output gives the offset that
    is added to the input tensor.

    The rank of the condition tensor can be smaller than the rank of the input tensor. In this case, the scale is
    applied according to shape parameter.

    Note: In this version of the FiLM layer, the shape parameter affects only the scale.

    Parameters:
        activation :
            Activation function in the linear layer of the scale.
        use_scale : bool
            Whether to use the scale.
        use_offset : bool
            Whether to use the offset.
        use_bias : bool
            Whether to use a bias in the linear layers of the scale and offset.
        shape : tuple
            Shape of the scale. If None, the scale is applied to the last dimension of the input. This is the default
            case from [1]. If shape is a tuple of length `n` (where `n = rank(input) - rank(condition) + 1`), the scale
            is applied to the last `n` dimensions of the input according to the specified shape. Setting values to
            `None` in the shape tuple results in the scale being applied to this dimension. Setting values to 1 in the
            shape results in the scale being broadcasted to this dimension. Note that the bias is always applied to the
            last dimension of the input. Default is None.
        kernel_initializer :
            Initializer for the kernel of the linear layers.
        bias_initializer :
            Initializer for the bias of the linear layers.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    Layer inputs:
        Tuple of two tensors of arbitrary shape. The first tensor is the input. The second tensor is the condition. The
        `k - 1` leading dimensions of the condition tensor must be broadcastable to the first `k - 1` leading dimensions
        of the input tensor, where `k` is the rank of the condition tensor.

    Layer outputs:
        Tensor of the same shape as input tensor.

     References:
        [1] Perez et al. (2017): https://arxiv.org/abs/1709.07871.
    """
    def __init__(self,
                 activation=None,
                 use_scale=True,
                 use_offset=True,
                 use_bias=True,
                 shape=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        self.activation = ks.activations.get(activation)
        self.use_scale = use_scale
        self.use_offset = use_offset
        self.use_bias = use_bias
        self.shape = shape
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.bias_initializer = ks.initializers.get(bias_initializer)
        super().__init__(**kwargs)

    def call(self, inputs):
        x, cond = inputs

        # scale input
        if self.use_scale:
            gamma = tf.tensordot(cond, self.gamma_kernel, axes=[[-1], [0]])
            if self.use_bias:
                gamma += self.gamma_bias

            if self.activation:
                gamma = self.activation(gamma)

            x *= gamma

        # add offset
        if self.use_offset:
            beta = tf.tensordot(cond, self.beta_kernel, axes=[[-1], [0]])
            if self.use_bias:
                beta += self.beta_bias

            x += beta

        return x

    def build(self, input_shape):
        data_shape, cond_shape = input_shape
        if len(cond_shape) > len(data_shape):
            raise ValueError("Condition tensor must be of rank smaller than input tensor.")

        # length of shape
        len_shape = len(data_shape) - len(cond_shape) + 1

        if self.shape is None:
            # default: apply scale to last dimension of input
            gamma_shape = [1] * (len_shape - 1) + [data_shape[-1]]
        else:
            assert len(self.shape) == len_shape, "Shape of length {} expected.".format(len_shape)
            # user specified shape, replace None with corresponding values of data shape
            gamma_shape = []
            for d, s in zip(data_shape[-len_shape:], self.shape):
                if s is None:
                    gamma_shape.append(d)
                else:
                    gamma_shape.append(s)

        # preprend with channel dimension of condition tensor
        gamma_shape = [cond_shape[-1]] + gamma_shape

        # offset always applies to the last dimension of the input
        beta_shape = [cond_shape[-1]] + [1] * (len_shape - 1) + [data_shape[-1]]

        if self.use_scale:
            self.gamma_kernel = self.add_weight('gamma_kernel',
                                                shape=gamma_shape,
                                                initializer=self.kernel_initializer,
                                                trainable=True)
            if self.use_bias:
                self.gamma_bias = self.add_weight('gamma_bias',
                                                  shape=gamma_shape[1:],
                                                  initializer=self.bias_initializer,
                                                  trainable=True)

        if self.use_offset:
            self.beta_kernel = self.add_weight('beta_kernel',
                                               shape=beta_shape,
                                               initializer=self.kernel_initializer,
                                               trainable=True)

            if self.use_bias:
                self.beta_bias = self.add_weight('beta_bias',
                                                 shape=beta_shape[1:],
                                                 initializer=self.bias_initializer,
                                                 trainable=True)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': ks.activations.serialize(self.activation),
            'shape': self.shape,
            'use_bias': self.use_bias,
            'use_scale': self.use_scale,
            'use_offset': self.use_offset,
        })
        return config


class Film(ks.layers.Layer):
    """Feature-wise linear modulation.

    Feature-wise linear modulation of the last dimension of the input tensor.

    A linear layer is applied to the last dimension of the condition tensor and the output gives the scale that is
    multiplied with the input tensor. The scale can be activated by a non-linearity prior to the multiplication.

    Another linear layer is applied to the last dimension of the condition tensor and the output gives the offset that
    is added to the input tensor.

    The rank of the condition tensor can be smaller than the rank of the input tensor. In this case, the scale is
    applied according to shape parameter.

    Note: In this version of the FiLM layer, the shape parameter affects both, the scale and the offset.

    Parameters:
        activation:
            Activation function in the linear layer of the scale.
        use_scale:
            Whether to use the scale.
        use_offset:
            Whether to use the offset.
        use_bias:
            Whether to use a bias in the linear layers of the scale and offset.
        shape:
            Shape of the modulation. If None, the modulation is applied to the last dimension of the input. This is the
            default case from [1]. If shape is a tuple of length `n` (where `n = rank(input) - rank(condition) + 1`),
            the modulation is applied to the last `n` dimensions of the input according to the specified shape. Setting
            values to `None` in the shape tuple results in the modulation being applied to this dimension. Setting
            values to 1 in the shape results in the modulation being broadcasted to this dimension.Default is None.
        kernel_initializer:
            Initializer for the kernel of the linear layers.
        bias_initializer:
            Initializer for the bias of the linear layers.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

     References:
        [1] Perez et al. (2017): https://arxiv.org/abs/1709.07871.
    """
    def __init__(self,
                 activation: str = None,
                 use_scale: bool = True,
                 use_offset: bool = True,
                 use_bias: bool = True,
                 shape: list[int] = None,
                 kernel_initializer: str = 'glorot_uniform',
                 bias_initializer: str = 'zeros',
                 **kwargs):
        self.activation = ks.activations.get(activation)
        self.use_scale = use_scale
        self.use_offset = use_offset
        self.use_bias = use_bias
        self.shape = shape
        self.kernel_initializer = ks.initializers.get(kernel_initializer)
        self.bias_initializer = ks.initializers.get(bias_initializer)
        super().__init__(**kwargs)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Apply FiLM to input tensor.

        Parameters:
            inputs:
                Tuple of two tensors of arbitrary shape. The first tensor is the input. The second tensor is the
                condition. The `k - 1` leading dimensions of the condition tensor must be broadcastable to the first
                `k- 1` leading dimensions of the input tensor, where `k` is the rank of the condition tensor.

        Returns:
            Tensor of the same shape as input tensor.

        """
        x, cond = inputs

        # scale input
        if self.use_scale:
            gamma = tf.tensordot(cond, self.gamma_kernel, axes=[[-1], [0]])
            if self.use_bias:
                gamma += self.gamma_bias

            if self.activation:
                gamma = self.activation(gamma)

            x *= gamma

        # add offset
        if self.use_offset:
            beta = tf.tensordot(cond, self.beta_kernel, axes=[[-1], [0]])
            if self.use_bias:
                beta += self.beta_bias

            x += beta

        return x

    def build(self, input_shape):
        data_shape, cond_shape = input_shape
        if len(cond_shape) > len(data_shape):
            raise ValueError("Condition tensor must be of rank smaller than input tensor.")

        # length of shape
        len_shape = len(data_shape) - len(cond_shape) + 1

        if self.shape is None:
            # default: apply scale to last dimension of input
            shape = [1] * (len_shape - 1) + [data_shape[-1]]
        else:
            assert len(self.shape) == len_shape, "Shape of length {} expected.".format(len_shape)
            # user specified shape, replace None with corresponding values of data shape
            shape = []
            for d, s in zip(data_shape[-len_shape:], self.shape):
                if s is None:
                    shape.append(d)
                else:
                    shape.append(s)

        # preprend with channel dimension of condition tensor
        shape = [cond_shape[-1]] + shape

        if self.use_scale:
            self.gamma_kernel = self.add_weight('gamma_kernel',
                                                shape=shape,
                                                initializer=self.kernel_initializer,
                                                trainable=True)
            if self.use_bias:
                self.gamma_bias = self.add_weight('gamma_bias',
                                                  shape=shape[1:],
                                                  initializer=self.bias_initializer,
                                                  trainable=True)

        if self.use_offset:
            self.beta_kernel = self.add_weight('beta_kernel',
                                               shape=shape,
                                               initializer=self.kernel_initializer,
                                               trainable=True)

            if self.use_bias:
                self.beta_bias = self.add_weight('beta_bias',
                                                 shape=shape[1:],
                                                 initializer=self.bias_initializer,
                                                 trainable=True)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'activation': ks.activations.serialize(self.activation),
            'shape': self.shape,
            'use_bias': self.use_bias,
            'use_scale': self.use_scale,
            'use_offset': self.use_offset,
        })
        return config


class GumbelSoftmax(ks.layers.Layer):
    """Random sampling from Gumbel softmax distribution.

    This layer is used to sample from a Gumbel softmax distribution.

    Parameters:
        axis:
            The axis along which to apply the Gumbel softmax. Default is last axis.
        temperature:
            The temperature of the Gumbel softmax. Default is 1.
        hard:
            Whether to sample from the hard or soft Gumbel distribution.
        noise_shape:
            The shape of the random noise. Must be of the same length as the number of dimensions in the input. `None`
            values in the tuple can be used to infer the shape from the input shape. If `None`, the noise shape will be
            equal to the shape of the input. Default is `None`.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    """
    def __init__(self,
                 axis: int = -1,
                 temperature: float = 1.,
                 hard: bool = False,
                 noise_shape: list[int] = None,
                 **kwargs):
        self.axis = axis
        self.temperature = temperature
        self.hard = hard
        self.noise_shape = noise_shape
        super().__init__(**kwargs)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply Gumbel softmax to input tensor.

        Parameters:
            inputs:
                Tensor of arbitrary shape. The input is expected to be logits.

        Returns:
            Tensor of the same shape as input tensor.

        """
        x = inputs
        shape = complete_shape(x, self.noise_shape)
        u = tf.random.uniform(shape, minval=0, maxval=1)
        eps = K.epsilon()
        g = -tf.math.log(-tf.math.log(u + eps) + eps)
        y = x + g
        y = tf.nn.softmax(y / self.temperature, axis=self.axis)
        if self.hard:
            y_hard = tf.one_hot(tf.argmax(y, axis=self.axis), depth=tf.shape(y)[self.axis], axis=self.axis)
            y_hard = tf.cast(y_hard, x.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def build(self, input_shape):
        if self.noise_shape is not None:
            assert len(self.noise_shape) == len(input_shape), "Noise shape must have same length as input shape."

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'axis': self.axis,
            'temperature': self.temperature,
            'hard': self.hard,
            'noise_shape': self.noise_shape,
        })
        return config


class _MCDropout(ks.layers.Dropout):
    """Monte Carlo dropout.

    This layer is identical to the default dropout layer during training. However, it behaves differently during
    inference, in which case dropout is also enabled [1].

    References:
        [1] Gal and Ghahramani, 2016: "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
        Learning," Proceedings of the 33rd International Conference on Machine Learning.
    """
    def call(self, inputs):
        return super().call(inputs, training=True)


class RandomSampling(ks.layers.Layer):
    """Random sampling from normal distribution.

    This layer samples from an isotropic Gaussian with mean `z_mean` and log variance `z_log_var`.

    Parameters:
        noise_shape:
            The shape of the random noise. Must be of the same length as the number of dimensions in the input and be
            broadcastable to the shape of the input. If `None`, the noise shape will be equal to the shape of the input.
            Default is `None`.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    """
    def __init__(self, noise_shape: list[int] = None, **kwargs):
        self.noise_shape = noise_shape
        super().__init__(**kwargs)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Sample from normal distribution.

        Parameters:
            inputs:
                Tuple of two or three tensors of the same shape. The first tensor is the mean of the normal
                distribution, the second tensor is the logarithm of the variance of the normal distribution. If a third
                tensor is given, it is used as the random sample. Otherwise, a random sample is generated.

        Returns:
            Tensor of the same shape as input tensors.

        """
        if len(inputs) == 2:
            z_mean, z_log_var = inputs
            shape = complete_shape(z_mean, self.noise_shape)
            sigma = tf.random.normal(shape=shape)
        else:
            z_mean, z_log_var, sigma = inputs

        return z_mean + tf.exp(0.5 * z_log_var) * sigma

    def build(self, input_shape):
        if self.noise_shape is not None:
            assert len(self.noise_shape) == len(input_shape[0]), "Noise shape must have same length as input shape."

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_shape': self.noise_shape,
        })
        return config


class Split(ks.layers.Layer):
    """Split input tensor into smaller chunks.

    Parameters:
        size_splits:
            Containing the sizes of each output tensor along `axis`.
        axis:
            The dimension along which to split. Defaults to 0.
        **kwargs:
            Additional keyword arguments passed to the `Layer` superclass.

    """
    def __init__(self, size_splits: list[int], axis: int = 0, **kwargs):
        self.size_splits = size_splits
        self.axis = axis
        super().__init__(**kwargs)

    def build(self, input_shape):
        if input_shape[self.axis] != sum(self.size_splits):
            raise ValueError('Cannot split dimension of size {} into chunks of size {}'.format(
                input_shape[self.axis], self.size_splits))

        self.built = True

    def call(self, inputs: tf.Tensor) -> list[tf.Tensor]:
        """Split input tensor into smaller chunks.

        Parameters:
            inputs:
                A tensor of arbitrary shape.

        Returns:
            A list of tensors.

        """
        return tf.split(inputs, self.size_splits, self.axis)

    def compute_output_shape(self, input_shape):
        sample_input = tf.constant(0, shape=input_shape)
        outputs = self.call(sample_input)
        return [output.shape for output in outputs]

    def get_config(self):
        config = super().get_config()
        config.update({
            'size_splits': self.size_splits,
            'axis': self.axis,
        })
        return config


def example_Film():
    """Example of Film layer."""
    input_shape = (1, 16, 12)
    cond_shape = (10, )
    x = tf.constant(1., shape=(32, ) + input_shape)
    c = tf.constant(1., shape=(32, ) + cond_shape)
    x_in = ks.layers.Input(shape=input_shape)
    cond_in = ks.layers.Input(shape=cond_shape)
    out = Film(
        use_scale=True,
        use_offset=True,
        use_bias=True,
        shape=(1, None, None),
    )([x_in, cond_in])
    model = ks.Model(inputs=[x_in, cond_in], outputs=out)
    model.summary()
    _ = model.predict([x, c])
    for w in model.weights:
        print(w.name, ':', w.shape)


def example_GumbelSoftmax():
    """Example of GumbelSoftmax layer."""
    input_shape = (5, 4)
    x = tf.zeros((2, ) + input_shape)
    x_in = ks.layers.Input(shape=input_shape)
    out = GumbelSoftmax(axis=-1, hard=True, noise_shape=(None, 1, None))(x_in)
    model = ks.Model(inputs=x_in, outputs=out)
    y = model.predict(x)
    print(y)


def example_RandomSampling():
    """Example of RandomSampling layer."""
    input_shape = (5, 4)
    z_mean = tf.zeros((2, ) + input_shape)
    z_log_var = tf.zeros((2, ) + input_shape)
    z_mean_in = ks.layers.Input(shape=input_shape)
    z_log_var_in = ks.layers.Input(shape=input_shape)
    out = RandomSampling(noise_shape=(None, 1, None))([z_mean_in, z_log_var_in])
    model = ks.Model(inputs=[z_mean_in, z_log_var_in], outputs=out)
    z = model.predict([z_mean, z_log_var])
    print(z)
