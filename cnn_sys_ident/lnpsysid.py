'''
Regularized LNP model for neural system identification

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


class LNP(Model):

    def build(self, smooth_reg_weight, sparse_reg_weight):
        self.smooth_reg_weight = smooth_reg_weight
        self.sparse_reg_weight = sparse_reg_weight
        with self.graph.as_default():
            tmp = tf.contrib.layers.convolution2d(self.images, self.data.num_neurons, self.data.px_x, 1, 'VALID',
                                                  activation_fn=tf.exp,
                                                  normalizer_fn=None,
                                                  #weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                  weights_regularizer=lambda w: smoothness_regularizer_2d_lnp(w, smooth_reg_weight)+\
                                                  l1_regularizer(w, sparse_reg_weight),
                                                  biases_initializer=tf.constant_initializer(value=0),
                                                  scope='lnp')
            with tf.variable_scope('lnp', reuse=True):
                self.weights = tf.get_variable('weights')
                self.biases = tf.get_variable('biases')
            self.prediction = tf.squeeze(tmp, squeeze_dims=[1, 2])
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            self.total_loss = self.get_log_likelihood() + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.initialize()

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.mse, self.prediction]