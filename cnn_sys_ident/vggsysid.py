'''
Vgg-based model for neural system identification

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
from collections import OrderedDict
from cnn_sys_ident.utils import *
from cnn_sys_ident.base import Model

PATH = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
VGG_CHECKPOINT_FILE = os.path.join(PATH, 'vgg_weights/vgg_normalized.ckpt')


def readout(inputs, num_neurons, smooth_reg_weight, sparse_reg_weight, group_sparsity_weight):
    s = inputs.shape
    reg = lambda w: smoothness_regularizer_2d_vgg(w, smooth_reg_weight) + \
    group_sparsity_regularizer_2d_vgg(w, group_sparsity_weight) +\
    l1_regularizer(w, sparse_reg_weight)  
    w_readout = tf.get_variable(
        'w_readout',
        shape=[s[1], s[2], s[3], num_neurons],
        initializer = tf.contrib.layers.xavier_initializer(),
        regularizer = reg)
    predictions = tf.tensordot(inputs, w_readout, [[1,2,3],[0,1,2]])
    s_ = predictions.shape
    biases = tf.get_variable('biases', shape=[num_neurons], initializer = tf.constant_initializer(value=0.0))
    predictions = predictions + biases
    return predictions

#####  Transfer Model #####

class VggTransfer(Model):
    
    def build(self,
              name_readout_layer,
              smooth_reg_weight, 
              sparse_reg_weight, 
              group_sparsity_weight,
              output_nonlin_smooth_weight = 1.0,
              b_norm = True):
        
        self.smooth_reg_weight = smooth_reg_weight
        self.sparse_reg_weight = sparse_reg_weight
        self.grout_sparsity_weight = group_sparsity_weight
        with self.graph.as_default():
            self.vgg = vgg19(self.images, subtract_mean=False, final_endpoint=name_readout_layer,padding='SAME')
            vgg_features = self.vgg[name_readout_layer]
            vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            self.saver_vgg = tf.train.Saver(var_list=vgg_vars)
            if b_norm:
                vgg_feats_bn = tf.layers.batch_normalization(vgg_features, training = self.is_training,\
                                                         momentum = 0.9, epsilon = 1e-4, name='vgg_bn', fused =True)
            else:
                vgg_feats_bn = vgg_features
                
            self.vgg_feats_bn = vgg_feats_bn
            vgg_feats_bn_act = tf.nn.relu(vgg_feats_bn) 
            self.vgg_feats_bn_act  = vgg_feats_bn_act
            predictions = readout(vgg_feats_bn_act, self.data.num_neurons, smooth_reg_weight, sparse_reg_weight, group_sparsity_weight)
            self.projection = predictions
            self.prediction = tf.identity(output_nonlinearity(predictions, self.data.num_neurons, vmin=-3, vmax=6,num_bins=50,\
                                                  alpha=output_nonlin_smooth_weight), name = 'predictions')            
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            self.total_loss = self.get_log_likelihood() + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            self.initialize()
            self.saver_vgg.restore(self.session, VGG_CHECKPOINT_FILE)

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.mse, self.prediction]