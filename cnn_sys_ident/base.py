'''
Defines the base for a Model for neural system identification
with it's trainer and evaluation functions

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

class Model:

    def __init__(self, data=None, log_dir=None, log_hash=None, global_step=None, obs_noise_model='poisson'):
        self.data = data
        log_dir_ = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
        log_dir = os.path.join(log_dir_, 'train_logs', 'cnn_tmp' if log_dir is None else log_dir)
        if log_hash == None: log_hash = '%010x' % random.getrandbits(40)
        self.log_dir = os.path.join(log_dir, log_hash)
        self.log_hash = log_hash
        self.seed = int.from_bytes(log_hash[:4].encode('utf8'), 'big')
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.obs_noise_model = obs_noise_model
        self.best_loss = 1e100
        self.val_iter_loss = []

        # placeholders
        if data is None: return
        with tf.Graph().as_default() as self.graph:
            self.is_training = tf.placeholder(tf.bool, name = 'is_training')
            self.learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
            self.images = tf.placeholder(tf.float32, shape=[None, data.px_y, data.px_x, 1], name='images')
            self.responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons], name='responses')
            self.realresp  = tf.placeholder(tf.float32, shape=[None, data.num_neurons], name='realresp')


    def initialize(self):
        loss_summ = tf.summary.scalar('loss_function', self.total_loss)
        self.summaries = tf.summary.merge_all()
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_best = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.log_dir, max_queue=0, flush_secs=0.1)


    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
                self.writer.close()
        except:
            pass


    def close(self):
        self.session.close()


    def save(self, step=None):
        if step == None:
            step = self.global_step
        chkp_file = os.path.join(self.log_dir, 'model.ckpt')
        self.saver.save(self.session, chkp_file, global_step=step)


    def save_best(self):
        self.saver_best.save(self.session, os.path.join(self.log_dir, 'best.ckpt'))


    def load(self, step=None):
        if step == None:
            step = self.global_step
        else:
            self.global_step = step
        chkp_file = os.path.join(self.log_dir, 'model.ckpt-%d' % step)
        self.saver.restore(self.session, chkp_file)


    def load_best(self):
        self.saver_best.restore(self.session, os.path.join(self.log_dir, 'best.ckpt'))



    def train(self,
              max_iter=10000,
              learning_rate=0.005,
              batch_size=256,
              val_steps=100,
              save_steps=500,
              early_stopping_steps=5):
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            imgs_val, res_val, realresp_val = self.data.val()
            not_improved = 0
            for i in range(self.global_step + 1, self.global_step + max_iter + 1):

                # training step
                imgs_batch, res_batch, rresp_batch = self.data.minibatch(batch_size)
                self.global_step = i
                feed_dict = {self.images: imgs_batch,
                             self.responses: res_batch,
                             self.realresp : rresp_batch,
                             self.is_training: True,
                             self.learning_rate: learning_rate}
                self.session.run([self.train_step, update_ops], feed_dict)
                # validate/save periodically
                if not i % save_steps:
                    self.save(i)
                if not i % val_steps:
                    result = self.eval(images=imgs_val,
                                       responses=res_val,
                                       realresp = realresp_val,
                                       with_summaries=True,
                                       keep_record_loss =True,
                                       global_step=i,
                                       learning_rate=learning_rate)
                    if result[0] < self.best_loss:
                        self.best_loss = result[0]
                        self.save_best()
                        not_improved = 0
                    else:
                        not_improved += 1
                    if not_improved == early_stopping_steps:
                        self.global_step -= early_stopping_steps * val_steps
                        self.load_best()
                        not_improved = 0
                        break
                    yield (i, result[:-1])


    def eval(self, with_summaries=False, keep_record_loss=False, images=None, responses=None, realresp=None, global_step=None, learning_rate=None):
        if (images is None) or (responses is None):
            images, responses, realresp = self.data.test()
            nrep, nim, nneu = responses.shape
            images = np.tile(images, [nrep, 1, 1, 1])
            responses = responses.reshape([nim*nrep, nneu])
            realresp  = realresp.reshape([nim*nrep, nneu])
        ops = self.get_test_ops()
        feed_dict = {self.images: images,
                     self.responses: responses,
                     self.realresp : realresp,
                     self.is_training: False}
        if with_summaries:
            assert global_step != None, 'global_step must be set for summaries'
            assert learning_rate != None, 'learning_rate must be set for summaries'
            ops += [self.summaries]
            feed_dict[self.learning_rate] = learning_rate
        result = self.session.run(ops, feed_dict)
        if with_summaries: self.writer.add_summary(result[-1], global_step)
        if keep_record_loss: self.val_iter_loss.append(result[0]) 
        return result


    def compute_log_likelihoods(self, prediction, response, realresp):
        self.poisson = tf.reduce_mean(tf.reduce_sum((prediction - response * tf.log(prediction + 1e-9))\
                                                    * realresp, axis=0) / tf.reduce_sum(realresp,axis=0))
        
        self.mse = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - response) \
                                                * realresp, axis=0) / tf.reduce_sum(realresp,axis=0))
        


    def get_log_likelihood(self):
        if self.obs_noise_model == 'poisson':
            return self.poisson
        elif self.obs_noise_model == 'gaussian':
            return self.mse


    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.prediction]
    
    def performance_test(self):
        '''
        This function computes the explainable variance explained on the test set
        '''
        images, responses, real_responses = self.data.test()
        nrep, nim, nneu = responses.shape
        
        # get predictions
        predictions_test = self.prediction.eval(session=self.session, \
                                                     feed_dict={self.images: images, self.is_training: False})
        predictions_test = np.tile(predictions_test.T, nrep).T
        
        # replace inserted zeros in responses arrays with nans.
        responses_nan = self.data.nanarray(real_responses, responses)
        
        # mean squared error
        #mse = np.nanvar((predictions_test.reshape([nrep*nim, nneu] - responses_nan.reshape([nrep*nim, nneu]), axis=0)
        mse = np.nanmean((predictions_test - responses_nan.reshape([nrep*nim,nneu]))**2, axis=0)
        
        total_variance, explainable_var = [],[]
        for n in range(self.data.num_neurons):
            rep     = self.data.repetitions[n]  # use only original number of repetitions 
            resp_   = responses_nan[:rep,:,n]
            obs_var = np.nanmean((np.nanvar(resp_,axis=0, ddof=1)),axis=0) # obs variance
            tot_var = np.nanvar(resp_,axis=(0,1), ddof =1)                 # total variance
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)                      # explainable variance

        total_variance = np.array(total_variance)
        #mse[mse > total_variance] = total_variance[mse > total_variance]
        explainable_var = np.array(explainable_var)
        var_explained = total_variance - mse
        eve = var_explained / explainable_var  # explainable variance explained
        
        self.eve            = eve
        self.var_explained  = var_explained
        self.explainable_var= explainable_var
        self.MSE            = mse
        self.total_variance = total_variance  
        
    def performance_val(self):
        '''
        This function computes the explainable variance explained on the validation set
        '''
        images, responses, real_responses = self.data.val()
        predictions_val = self.prediction.eval(session=self.session, \
                                                     feed_dict={self.images: images, self.is_training: False})
        # replace inserted zeros in responses arrays with nans.
        responses_nan = self.data.nanarray(real_responses, responses)
        
        # mean aquared error
        #mse  = np.nanvar((predictions_val - responses_nan),axis=0)
        mse = np.nanmean((predictions_val - responses_nan)**2, axis=0)
        sz  = responses_nan.shape[0]
        resps_reshaped = responses_nan.reshape([self.data.num_reps, int(sz / self.data.num_reps), self.data.num_neurons])
        
        total_variance, explainable_var = [],[]
        for n in range(self.data.num_neurons):
            rep     = self.data.repetitions[n]    # use only original number of repetitions 
            resp_   = resps_reshaped[:rep,:,n]
            obs_var = np.nanmean((np.nanvar(resp_,axis=0, ddof=1)),axis=0) # obs variance
            tot_var = np.nanvar(resp_,axis=(0,1), ddof =1)                 # total variance
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)                      # explainable variance

        total_variance = np.array(total_variance)
        #mse[mse > total_variance] = total_variance[mse > total_variance]
        explainable_var = np.array(explainable_var)
        var_explained   = total_variance - mse
        eve = var_explained / explainable_var
        
        self.eve_val             = eve
        self.var_explained_val   = var_explained
        self.explainable_var_val = explainable_var
        self.MSE_val             = mse
        self.total_variance_val  = total_variance  
    
    
    def evaluate_avg_corr_val(self):
        '''
        Computes average correlation of the trained model on the validation set.
        '''
        from scipy.stats import pearsonr
        
        im, res, real_res = self.data.val()   # e.g. eval_data = data.val()
        pred = self.prediction.eval(session=self.session, feed_dict={self.images: im, self.is_training: False})
        
        # iterate over neurons
        corrs = []
        for i in range(self.data.num_neurons):
            
            # remove entries in both lists in which one is not real_res (i.e. keep only real_res)
            r = res[:,i]
            p = pred[:,i]
            b = real_res[:,i].astype(np.bool)
            r = np.compress(b, r)
            p = np.compress(b, p)
            corr = pearsonr(r, p)[0]

#             if np.isnan(corr):
#                 print("WARNING: corr for neuron", i, "is nan")

            corrs.append(corr)
        
        # if one of the inputs has zero variance, division by zero occurs. Therefore, pearsonr returns nan
        # remove nans
        clean_corrs = np.array([v for v in corrs if not np.isnan(v)])
        avg_corr    = np.mean(clean_corrs)
        return avg_corr