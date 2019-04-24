'''
Util functions and classes for neural system identification

Author: Santiago Cadena
Last update: April 2019
'''


import numpy as np
import os
from scipy import stats
import tensorflow as tf
import tensorflow.contrib.slim as slim
import hashlib
import inspect
import random
from tensorflow.contrib import layers
from tensorflow import losses
from numpy import pi
from collections import OrderedDict

###### Regularization Functions and Static Nonlinearities ###########

def smoothness_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
        #lap = np.array([[0,-1.0,0],[-1.0, 4.0,-1],[0,-1.0,0]]).astype(np.float32)
        lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
        out_channels = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(tf.transpose(W, perm=[3, 0, 1, 2]),
                                       tf.tile(lap, [1, 1, out_channels, 1]),
                                       strides=[1, 1, 1, 1], padding='SAME')
        penalty = tf.reduce_sum(tf.reduce_sum(tf.square(W_lap), [1, 2]) / tf.transpose(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty

def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_2d', penalty)
        return penalty
    
# Same two regularization functions as above with slight changes for the vgg model:

def smoothness_regularizer_2d_vgg(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = np.array([[0,-1.0,0],[-1.0, 4.0,-1],[0,-1.0,0]]).astype(np.float32)
        num_filters = W.get_shape().as_list()[2]
        W_ = tf.transpose(W, perm=[3, 0, 1, 2])
        lap_ = tf.expand_dims(tf.expand_dims(lap, 2), 3) # shape= [3,3,1,1]
        W_lap = tf.nn.depthwise_conv2d(W_, tf.tile(lap_, [1, 1, num_filters, 1]),\
                                   strides=[1, 1, 1, 1], padding='SAME')     
        penalty = tf.reduce_mean(tf.reduce_sum(tf.square(W_lap), [1, 2, 3]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty

def group_sparsity_regularizer_2d_vgg(W, weight=1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W),[0,1])),[1]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('group_sparsity_regularizer_2d', penalty)
        return penalty

# Same smoothness regularizer 2d for the LNP model with slight change:

def smoothness_regularizer_2d_lnp(W, weight=1.0):
    with tf.variable_scope('smoothness'):
        lap = np.array([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]]).astype(np.float32)
        num_filters = W.get_shape().as_list()[2]
        W_ = tf.transpose(W, perm=[3, 0, 1, 2])
        lap_ = tf.expand_dims(tf.expand_dims(lap, 2), 3) # shape= [3,3,1,1]
        W_lap = tf.nn.depthwise_conv2d(W_, tf.tile(lap_, [1, 1, num_filters, 1]),\
                                   strides=[1, 1, 1, 1], padding='SAME')    
        penalty = tf.reduce_mean(tf.reduce_sum(tf.square(W_lap), [1, 2, 3]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('smoothness_regularizer_2d', penalty)
        return penalty
    
def l1_regularizer(W, weight = 1.0):
    with tf.variable_scope('group_sparsity'):
        penalty = tf.reduce_mean(tf.reduce_sum(tf.abs(W), [0,1,2]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('l1_regularizer_', penalty)
        return penalty


def negbino(x, mu, r):
    return tf.lgamma(r) - tf.lgamma(x + r) - x * tf.log(mu + 1e-5) + (x + r) * tf.log(mu + r) - r * tf.log(r)


def lin_step(x, a, b):
    return tf.minimum(tf.constant(b - a, dtype=tf.float32), tf.nn.relu(x - tf.constant(a, dtype=tf.float32))) / (b - a)


def tent(x, a, b):
    z = tf.constant(0, dtype=tf.float32)
    d = tf.constant(2 * (b - a), dtype=tf.float32)
    a = tf.constant(a, dtype=tf.float32)
    return tf.minimum(tf.maximum(x - a, z), tf.maximum(a + d - x, z)) / (b - a)


def smoothness_regularizer_1d(w, weight=1.0, order=2):
    penalty = 0
    kernel = tf.constant([-1.0, 1.0], shape=[2, 1, 1], dtype=tf.float32)
    for k in range(order):
        w = tf.nn.conv1d(w, kernel, 1, 'VALID')
        penalty += tf.reduce_sum(tf.reduce_mean(tf.square(w), 1))
    penalty = tf.identity(weight * penalty, name='penalty')
    tf.add_to_collection('smoothness_regularizer_1d', penalty)
    return penalty


def output_nonlinearity(x, num_neurons, vmin=-3.0, vmax=6.0, num_bins=10, alpha=0, scope='output_nonlinearity'):
    with tf.variable_scope(scope):
        elu = tf.nn.elu(x - 1.0) + 1.0
        if alpha == -1:
            tf.add_to_collection('output_nonlinearity', 0)
            return elu
        _, neurons = x.get_shape().as_list()
        #neurons = 166
        k = int(num_bins / 2)
        num_bins = 2 * k
        bins = np.linspace(vmin, vmax, num_bins+1, endpoint=True)
        bin_size = (vmax - vmin) / num_bins
        segments = [tent(x, a, b) for a, b in zip(bins[:-2], bins[1:-1])] + \
                   [lin_step(x, bins[-2], bins[-1])]
        #v = tf.transpose(tf.concat(2, [tf.reshape(s, [-1, neurons, 1]) for s in segments]), [1, 0, 2])
        v = tf.transpose(tf.concat([tf.reshape(s, [-1, neurons, 1]) for s in segments],axis=2), [1, 0, 2])
        reg = lambda w: smoothness_regularizer_1d(w, weight=alpha, order=2)
        a = tf.get_variable('weights',
                            shape=[neurons, num_bins, 1],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0),
                            regularizer=reg)
        a = tf.exp(a)
        tf.add_to_collection('output_nonlinearity', a)
        multiplier = tf.transpose(tf.reshape(tf.matmul(v, a), [neurons, -1]))
        return multiplier * elu


def inv_elu(x):
    y = x.copy()
    idx = y < 1.0
    y[idx] = np.log(y[idx]) + 1.0
    return y

# Functions for the GFB:
# Regularizer:
def l1_regularizer_flatten(W, weight = 1.0):
    with tf.variable_scope('sparsity'):
        penalty = tf.reduce_mean(tf.reduce_sum(tf.abs(W), [0]))
        penalty = tf.identity(weight * penalty, name='penalty')
        tf.add_to_collection('l1_regularizer_', penalty)
        return penalty

def elu(x, *args, **kwargs):
    return tf.identity(tf.nn.elu(x - 1.0) + 1.0, *args, **kwargs)

def relu(x, *args, **kwargs):
    return tf.nn.relu(x, *args, **kwargs)
    

############### VGG 19 Definition ####################

def vgg19(images, reuse=False, pooling='max', subtract_mean=True, final_endpoint='pool5', padding='VALID'):

    filter_size = [3, 3]
    conv1 = lambda net, name: slim.conv2d(net, 64, filter_size, padding=padding, scope=name)
    conv2 = lambda net, name: slim.conv2d(net, 128, filter_size, padding=padding, scope=name)
    conv3 = lambda net, name: slim.conv2d(net, 256, filter_size, padding=padding, scope=name)
    conv4 = lambda net, name: slim.conv2d(net, 512, filter_size, padding=padding, scope=name)
    conv5 = conv4
    pooling_fns = {'avg': slim.avg_pool2d, 'max': slim.max_pool2d}
    pool =  lambda net, name: pooling_fns[pooling](net, [2, 2], scope=name)
    dropout = lambda net, name: slim.dropout(net, 0.5, is_training=False, scope=name)

    layers = OrderedDict()
    layers['conv1/conv1_1'] = conv1
    layers['conv1/conv1_2'] = conv1
    layers['pool1'] = pool
    layers['conv2/conv2_1'] = conv2
    layers['conv2/conv2_2'] = conv2
    layers['pool2'] = pool
    layers['conv3/conv3_1'] = conv3
    layers['conv3/conv3_2'] = conv3
    layers['conv3/conv3_3'] = conv3
    layers['conv3/conv3_4'] = conv3
    layers['pool3'] = pool
    layers['conv4/conv4_1'] = conv4
    layers['conv4/conv4_2'] = conv4
    layers['conv4/conv4_3'] = conv4
    layers['conv4/conv4_4'] = conv4
    layers['pool4'] = pool
    layers['conv5/conv5_1'] = conv5
    layers['conv5/conv5_2'] = conv5
    layers['conv5/conv5_3'] = conv5
    layers['conv5/conv5_4'] = conv5
    layers['pool5'] = pool
    layers['fc6'] = lambda net, name: slim.conv2d(net, 4096, [7, 7], padding='VALID', scope=name)
    layers['dropout6'] =  dropout
    layers['fc7'] = lambda net, name: slim.conv2d(net, 4096, [1, 1], padding='VALID', scope=name)
    layers['dropout7'] =  dropout
    layers['fc8'] = lambda net, name: slim.conv2d(net, 1000, [1, 1], padding='VALID', scope=name)

    with tf.variable_scope('vgg_19', reuse=reuse) as sc:
        if images.shape[-1] < 3:
            images = tf.tile(images, [1, 1, 1, 3])
        net = images
        if subtract_mean:
            net -= tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
        end_points = OrderedDict()
        with slim.arg_scope([slim.conv2d], trainable=False):
            for layer_name, layer_op in layers.items():
                last = final_endpoint == layer_name
                act_fn = tf.nn.relu if not last else None
                bias_in = tf.zeros_initializer() if not last else None
                with slim.arg_scope([slim.conv2d],activation_fn=act_fn,biases_initializer=bias_in):
                    net = layer_op(net, layer_name)
                end_points[layer_name] = net
                if last:
                    break
    return end_points

RF_SIZES = OrderedDict()
RF_SIZES['conv1/conv1_1'] = 3
RF_SIZES['conv1/conv1_2'] = 5
RF_SIZES['pool1'] = 6
RF_SIZES['conv2/conv2_1'] = 10
RF_SIZES['conv2/conv2_2'] = 14
RF_SIZES['pool2'] = 16
RF_SIZES['conv3/conv3_1'] = 24
RF_SIZES['conv3/conv3_2'] = 32
RF_SIZES['conv3/conv3_3'] = 40
RF_SIZES['conv3/conv3_4'] = 48
RF_SIZES['pool3'] = 52
RF_SIZES['conv4/conv4_1'] = 68
RF_SIZES['conv4/conv4_2'] = 84
RF_SIZES['conv4/conv4_3'] = 100
RF_SIZES['conv4/conv4_4'] = 116
RF_SIZES['pool4'] = 124
RF_SIZES['conv5/conv5_1'] = 156
RF_SIZES['conv5/conv5_2'] = 188
RF_SIZES['conv5/conv5_3'] = 220
RF_SIZES['conv5/conv5_4'] = 252
RF_SIZES['pool5'] = 268


#################### Gabor filter bank generation ##########################

class GaborSet:
    def __init__(self,
                 canvas_size,  # width x height
                 center_range, # [x_start, x_end, y_start, y_end]
                 sizes, # +/- 2 SD of envelope
                 spatial_frequencies,  # cycles / envelop SD, i.e. depends on size
                 contrasts,
                 orientations,
                 phases,
                 aspect_ratios,
                 relative_sf=True):   # scale SF by size (True) or use absolute units (False)
        self.canvas_size = canvas_size
        cr = center_range
        self.locations = np.array(
            [[x, y] for x in range(cr[0], cr[1]) 
                    for y in range(cr[2], cr[3])])
        self.sizes = sizes
        self.spatial_frequencies = spatial_frequencies
        self.contrasts = contrasts
        self.aspect_ratios = aspect_ratios
        if type(orientations) is not list:
            self.orientations = np.arange(orientations) * pi / orientations
        else:
            self.orientations = orientations
        if type(phases) is not list:
            self.phases = np.arange(phases) * (2*pi) / phases
        else:
            self.phases = phases
        self.num_params = [
            self.locations.shape[0],
            len(sizes),
            len(spatial_frequencies),
            len(contrasts),
            len(self.orientations),
            len(self.phases),
            len(self.aspect_ratios)
        ]
        self.relative_sf = relative_sf

    def params_from_idx(self, idx):
        c = np.unravel_index(idx, self.num_params)
        location = self.locations[c[0]]
        size = self.sizes[c[1]]
        spatial_frequency = self.spatial_frequencies[c[2]]
        if self.relative_sf:
            spatial_frequency /= size
        contrast = self.contrasts[c[3]]
        orientation = self.orientations[c[4]]
        phase = self.phases[c[5]]
        aspect_ratio = self.aspect_ratios[c[6]]
        return location, size, spatial_frequency, contrast, orientation, phase, aspect_ratio
        
    def params_dict_from_idx(self, idx):
        (location, size, spatial_frequency, 
            contrast, orientation, phase, aspect_ratio) = self.params_from_idx(idx)
        return {
            'location': location,
            'size': size,
            'spatial_frequency': spatial_frequency,
            'contrast': contrast,
            'orientation': orientation,
            'phase': phase,
            'aspect_ratio': aspect_ratio,
        }

    def gabor_from_idx(self, idx):
        return self.gabor(*self.params_from_idx(idx))

    def gabor(self, location, size, spatial_frequency, contrast, orientation, phase, aspect_ratio):
        x, y = np.meshgrid(np.arange(self.canvas_size[0]) - location[0],
                           np.arange(self.canvas_size[1]) - location[1])
        R = np.array([[np.cos(orientation), -np.sin(orientation)],
                      [np.sin(orientation),  np.cos(orientation)]])
        coords = np.stack([x.flatten(), y.flatten()])
        x, y = R.dot(coords).reshape((2, ) + x.shape)
        envelope = 0.5 * contrast * np.exp(-(x ** 2 + (aspect_ratio * y) ** 2) / (2 * (size/4)**2))
        
        grating = np.cos(spatial_frequency * x * (2*pi) + phase)
        return envelope * grating

    def image_batches(self, batch_size):
        num_stims = np.prod(self.num_params)
        for batch_start in np.arange(0, num_stims, batch_size):
            batch_end = np.minimum(batch_start + batch_size, num_stims)
            images = [self.gabor_from_idx(i)
                          for i in range(batch_start, batch_end)]
            yield np.array(images)

    def images(self):
        num_stims = np.prod(self.num_params)
        return np.array([self.gabor_from_idx(i) for i in range(num_stims)])
    