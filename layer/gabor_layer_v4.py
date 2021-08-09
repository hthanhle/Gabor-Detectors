"""
Reference:
H. T. Le, S. L. Phung, P. B. Chapple, A. Bouzerdoum, C. H. Ritz, and L. C. Tran, “Deep Gabor neural network for
automatic detection of mine-like objects in sonar imagery,” IEEE Access, vol. 8, pp. 94 126–94 139, 2020.
@author: Thanh Le
"""
from keras import backend as K
from keras import activations, regularizers, initializers, constraints
from keras.engine.topology import Layer
from keras.utils import conv_utils
import tensorflow as tf
import numpy as np


class Gabor2D(Layer):
    def __init__(self, num_kernels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_regularizer=None,
                 **kwargs):

        super(Gabor2D, self).__init__(**kwargs)

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.strides = conv_utils.normalize_tuple(strides, 2,
                                                  'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        self.num_channels = input_shape[channel_axis]

        self.sigma = self.add_weight(name='sigma',
                                     shape=(self.num_channels, self.num_kernels),
                                     initializer=initializers.TruncatedNormal(mean=5.0, stddev=1.5),
                                     trainable=True)
        self.theta = self.add_weight(name='theta',
                                     shape=(self.num_channels, self.num_kernels),
                                     initializer=initializers.RandomUniform(minval=0.0, maxval=1.0),
                                     trainable=True)
        self.lamda = self.add_weight(name='lambda',
                                     shape=(self.num_channels, self.num_kernels),
                                     initializer=initializers.TruncatedNormal(mean=5.0, stddev=1.5),
                                     trainable=True)
        self.gamma = self.add_weight(name='gamma',
                                     shape=(self.num_channels, self.num_kernels),
                                     initializer=initializers.TruncatedNormal(mean=1.5, stddev=0.4),
                                     trainable=True)
        self.psi = self.add_weight(name='psi',
                                   shape=(self.num_channels, self.num_kernels),
                                   initializer=initializers.RandomUniform(minval=0.0, maxval=1.0),
                                   trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_kernels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True)
        else:
            self.bias = None

        super(Gabor2D, self).build(input_shape)

    def call(self, inputs):
        sigma_x = self.sigma
        sigma_y = self.sigma / self.gamma

        half_size = np.floor(self.kernel_size / 2)
        y, x = np.mgrid[-half_size: half_size + 1, -half_size: half_size + 1]
        x_kernels = np.tile(x, (self.num_channels, self.num_kernels, 1, 1))  # 4D Numpy Array
        x_kernels = np.transpose(x_kernels, (2, 3, 0, 1))
        y_kernels = np.tile(y, (self.num_channels, self.num_kernels, 1, 1))
        y_kernels = np.transpose(y_kernels, (2, 3, 0, 1))

        rot_x_kernels = x_kernels * tf.cos(self.theta - np.pi) + y_kernels * tf.sin(self.theta - np.pi)
        rot_y_kernels = -x_kernels * tf.sin(self.theta - np.pi) + y_kernels * tf.cos(self.theta - np.pi)

        self.gabor_kernels = tf.exp(-0.5 * (rot_x_kernels ** 2 / sigma_x ** 2 + rot_y_kernels ** 2 / sigma_y ** 2))
        self.gabor_kernels /= 2 * np.pi * sigma_x * sigma_y
        self.gabor_kernels *= tf.cos(2 * np.pi / self.lamda * rot_x_kernels + self.psi)

        outputs = K.conv2d(inputs, self.gabor_kernels,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias,
                                 data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        kernel_size_tuple = conv_utils.normalize_tuple(self.kernel_size, 2, 'kernel_size_tuple')

        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []

            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size_tuple[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0],) + tuple(new_space) + (self.num_kernels,)

        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    kernel_size_tuple[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            return (input_shape[0], self.num_kernels) + tuple(new_space)

    def get_config(self):
        config = {'num_kernels': self.num_kernels,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': self.activation,
                  'use_bias': self.use_bias,
                  'bias_initializer': self.bias_initializer,
                  'bias_regularizer': self.bias_regularizer,
                  'bias_constraint': self.bias_constraint,
                  'kernel_regularizer': self.kernel_regularizer}
        base_config = super(Gabor2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
