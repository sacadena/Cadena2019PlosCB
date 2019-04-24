'''
Data driven CNN model for neural system identification

Author: Santiago Cadena
Last update: April 2019
'''

import numpy as np
import os
from scipy import stats
import tensorflow as tf
import hashlib
import inspect
import random
from tensorflow.contrib import layers
from tensorflow import losses
from cnn_sys_ident.utils import *
from cnn_sys_ident.base import Model
            
class ConvNet(Model):

    def __init__(self, *args, **kwargs):
        super(ConvNet, self).__init__(*args, **kwargs)
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0


    def build(self,
              filter_sizes,
              out_channels,
              strides,
              paddings,
              smooth_weights,
              sparse_weights,
              readout_sparse_weight,
              output_nonlin_smooth_weight):

        with self.graph.as_default():

            self.temp = tf.reduce_mean(self.images,name='temp')
            # convolutional layers
            for i, (filter_size,
                    out_chans,
                    stride,
                    padding,
                    smooth_weight,
                    sparse_weight) in enumerate(zip(filter_sizes,
                                                    out_channels,
                                                    strides,
                                                    paddings,
                                                    smooth_weights,
                                                    sparse_weights)):
                x = self.images if not i else self.conv[i-1]
                bn_params = {'decay': 0.9, 'is_training': self.is_training}
                scope = 'conv{}'.format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + \
                                group_sparsity_regularizer_2d(w, sparse_weight)
                c = layers.convolution2d(inputs=x,
                                         num_outputs=out_chans,
                                         kernel_size=int(filter_size),
                                         stride=int(stride),
                                         padding=padding,
                                         activation_fn=tf.nn.elu,
                                         normalizer_fn=layers.batch_norm,
                                         normalizer_params=bn_params,
                                         weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                         weights_regularizer=reg,
                                         scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable('weights')
                self.W.append(W)
                self.conv.append(c)

            # readout layer
            sz = c.get_shape()
            px_x_conv = int(sz[1])
            px_y_conv = int(sz[2])
            px_conv = px_x_conv * px_y_conv
            conv_flat = tf.reshape(c, [-1, px_conv, out_channels[-1], 1])
            self.W_spatial = tf.get_variable('W_spatial',
                                             shape=[px_conv, self.data.num_neurons],
                                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
            h_spatial = tf.nn.conv2d(conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding='VALID')
            self.W_features = tf.get_variable('W_features',
                                              shape=[out_channels[-1], self.data.num_neurons],
                                              initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.h_out = tf.reduce_sum(tf.multiply(h_spatial, self.W_features), [1, 2])

            # L1 regularization for readout layer
            self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                tf.reduce_sum(tf.abs(self.W_spatial), 0) * \
                tf.reduce_sum(tf.abs(self.W_features), 0)
            )
            losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

            # output nonlinearity
            _, responses, realresp = self.data.train()
            b = inv_elu(responses.mean(axis=0))
            self.b_out = tf.get_variable('b_out',
                                         shape=[self.data.num_neurons],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(b))
            self.prediction = tf.identity(output_nonlinearity(self.h_out + self.b_out, self.data.num_neurons,
                                                  vmin=-3, vmax=6,
                                                  num_bins=50,
                                                  alpha=output_nonlin_smooth_weight), name = 'predictions')

            # loss
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            losses.add_loss(self.get_log_likelihood())
            self.total_loss = losses.get_total_loss()

            # regularizers
            if output_nonlin_smooth_weight > -1:
                self.output_regularizer = tf.add_n(tf.get_collection('smoothness_regularizer_1d'))
            self.smoothness_regularizer = tf.add_n(tf.get_collection('smoothness_regularizer_2d'))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection('group_sparsity_regularizer_2d'))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # summaries
            mse_summ = tf.summary.scalar('mean-squared_error', self.mse)
            smooth_summ = tf.summary.scalar('smoothness', self.smoothness_regularizer)
            conv_sparse_summ = tf.summary.scalar('conv_sparseness', self.group_sparsity_regularizer)
            readout_sparse_summ = tf.summary.scalar('readout_sparseness', self.readout_sparseness_regularizer)
            filter_summ = tf.summary.image('conv1_filters', tf.transpose(self.W[0], perm=[3, 0, 1, 2]), max_outputs=out_channels[0])
            rfs = tf.reshape(self.W_spatial, [px_x_conv, px_y_conv, self.data.num_neurons])
            rfs = tf.transpose(rfs, perm=[2, 0, 1])
            rf_summ = tf.summary.image('receptive_fields', tf.expand_dims(rfs, 3), max_outputs=20)
            lr_summ = tf.summary.scalar('learning_rate', self.learning_rate)

            # initialize TF session
            self.initialize()


    def get_test_ops(self):
        return [self.get_log_likelihood(),
                self.readout_sparseness_regularizer,
                self.group_sparsity_regularizer,
                self.smoothness_regularizer,
                self.total_loss,
                self.prediction]


