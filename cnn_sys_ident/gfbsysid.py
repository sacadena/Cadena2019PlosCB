'''
Gabor Filter Bank model (GFB) for neural system identification

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


    
class GaborFilterBank(Model):
    def __init__(self, *args, **kwargs):
        super(GaborFilterBank, self).__init__(*args, **kwargs)
        self.total_loss = 0

    def build(self,
            sizes, # +/- 2 SD of envelope
            spatial_frequencies,  # cycles / envelop SD, i.e. depends on size
            contrasts,
            orientations,
            phases,
            aspect_ratios = [1],
            stride = 1,
            padding = 'VALID',
            dilation = 1,
            nonlinearity = tf.nn.relu,
            sparsity_readout = 0.5):
        
        self.sizes        = sizes
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.nonlinearity = nonlinearity
        self.gaborsets    = []
        self.gaborsets_even = []
        self.gaborsets_odd  = []
        
        for s in sizes:
            #center_location = (s//2, s//2+1, s//2, s//2+1)
            #gab_set = GaborSet((s,s), center_location, [s], spatial_frequencies, contrasts, orientations, phases)            
            
            canv_size = min(np.round(s*1).astype(np.uint8), 40)
            center_location = (canv_size//2, canv_size//2+1, canv_size//2, canv_size//2+1)
            gab_set = GaborSet((canv_size, canv_size), center_location, [s], spatial_frequencies, contrasts, orientations, phases, aspect_ratios)            

            self.gaborsets.append(gab_set)
            gab_set_odd  = GaborSet((s,s),center_location, [s], spatial_frequencies, contrasts, orientations, [0], aspect_ratios)
            gab_set_even = GaborSet((s,s),center_location, [s], spatial_frequencies, contrasts, orientations, [pi/2], aspect_ratios)
            self.gaborsets_odd.append(gab_set_odd)
            self.gaborsets_even.append(gab_set_even)
        
        
        self.hnn_simple = []
        self.hnn_energy = []
        with self.graph.as_default():
            for si in range(len(sizes)):
                
                # Simple cell feature space
                gab_set = self.gaborsets[si]
                filters = gab_set.images() # get Gabor filters
                filters = filters[None,].transpose(2, 3, 0, 1)
                filters = tf.constant(filters, dtype=tf.float32, name = 'filters_simple')
                h  = tf.nn.conv2d(self.images, filters, [1] + [stride]*2 + [1], padding, dilations = [1] + [dilation]*2 + [1], name = 'conv_output')
                self.hnn_simple.append(nonlinearity(h, name = 'output_simple'))
                
                # Energy model feature space
                gab_set_even = self.gaborsets_even[si]
                filters_even = gab_set_even.images() # get Gabor filters 
                filters_even = filters_even[None,].transpose(2, 3, 0, 1)
                filters_even = tf.constant(filters_even, dtype=tf.float32,  name = 'filters_even')
                
                gab_set_odd = self.gaborsets_odd[si]
                filters_odd = gab_set_odd.images() # get Gabor filters 
                filters_odd = filters_odd[None,].transpose(2, 3, 0, 1)
                filters_odd = tf.constant(filters_odd, dtype=tf.float32, name = 'filters_odd')
                
                h_odd   = tf.nn.conv2d(self.images, filters_odd, [1] + [stride]*2 + [1], padding, dilations = [1] + [dilation]*2 + [1], name = 'conv_output_odd')
                h_even  = tf.nn.conv2d(self.images, filters_even, [1] + [stride]*2 + [1], padding, dilations = [1] + [dilation]*2 + [1], name = 'conv_output_even')
                self.hnn_energy.append(tf.sqrt(tf.square(h_odd) + tf.square(h_even), name = 'output_energy'))
             
            def flatten(list_tensor):
                flatten_tensor = []
                for si in range(len(list_tensor)):
                    flatten_tensor.append(tf.layers.flatten(list_tensor[si]))
                return tf.concat(flatten_tensor, axis = -1)
            
            self.simple_flat = flatten(self.hnn_simple)
            self.energy_flat = flatten(self.hnn_energy)
            
            self.feature_space = tf.concat([self.simple_flat, self.energy_flat], axis = -1)
            
            ## Readout
            
            regularizer = lambda w: l1_regularizer_flatten(w, sparsity_readout)  
            
            # Linear regression:
            self.w_readout = tf.get_variable('w_readout',
                                     shape=[self.feature_space.shape[-1], self.data.num_neurons],
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                                     regularizer = regularizer)
            self.h_out       = tf.tensordot(self.feature_space, self.w_readout, [[1],[0]])
            self.readout_out = tf.layers.batch_normalization(self.h_out)
            self.prediction  = tf.nn.elu(self.readout_out - 1.0) + 1.0
            
            self.compute_log_likelihoods(self.prediction, self.responses, self.realresp)
            self.total_loss = self.get_log_likelihood() + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)
            
            self.initialize()
    
    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.mse, self.prediction]
                
