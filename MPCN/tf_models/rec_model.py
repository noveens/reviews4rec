#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import gzip
import json
from tqdm import tqdm
import random
from collections import Counter
import operator
import timeit
import time

import datetime
from keras.preprocessing import sequence

from .utilities import *
from keras.utils import np_utils
import numpy as np

from tylib.lib.att_op import *
from tylib.lib.seq_op import *
from tylib.lib.cnn import *
from tylib.lib.compose_op import *

from .mpcn import *

class RecModel:
    ''' Base rec model class.
        Implements latent factor model.
    '''
    def __init__(self, num_user, num_item, args):
        self.graph = tf.Graph()
        self.args = args
        self.imap = {} # index map of inputs
        self.inspect_op = []
        self.write_dict = {}
        # For interaction data only (disabled and removed from this repo)
        self.num_user = num_user
        self.num_item = num_item

        self.feat_prop = None
        if(self.args.init_type=='xavier'):
            self.initializer = tf.contrib.layers.xavier_initializer()
        elif(self.args.init_type=='normal'):
            self.initializer = tf.random_normal_initializer(0.0,
                                        self.args.init)
        elif(self.args.init_type=='uniform'):
            self.initializer = tf.random_uniform_initializer(
                                        maxval=self.args.init,
                                        minval=-self.args.init)

        self.cnn_initializer = tf.random_uniform_initializer(
                                        maxval=self.args.init,
                                        minval=-self.args.init)
        self.init = self.initializer
        self.build_graph()

    def _get_pair_feed_dict(self, data, mode='training', lr=None):
        """ This is for pairwise ranking and not relevant to this repo.
        """
        data = zip(*data)
        labels = data[-1]

        if(lr is None):
            lr = self.args.learn_rate

        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.learn_rate:lr,
            #self.dropout:self.args.dropout,
            #self.rnn_dropout:self.args.rnn_dropout,
            #self.emb_dropout:self.args.emb_dropout
        }
        if(mode=='training'):
            feed_dict[self.q3_inputs] = data[self.imap['q3_inputs']]
        #if(mode!='training'):
        #    feed_dict[self.dropout] = 1.0
        #    feed_dict[self.rnn_dropout] = 1.0
        #    feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
            if(mode=='training'):
                feed_dict[self.neg_features] = data[7]
        return feed_dict

    def _check_model_type(self):
        if('SOFT' in self.args.rnn_type):
            return 'point'
        elif('SIG_MSE' in self.args.rnn_type \
                or 'RAW_MSE' in self.args.rnn_type):
            return 'point'
        else:
            return 'pair'

    def get_feed_dict(self, data, mode='training', lr=None):
        mdl_type = self._check_model_type()
        if(mdl_type=='point'):
            return self._get_point_feed_dict(data, mode=mode, lr=lr)
        else:
            return self._get_pair_feed_dict(data, mode=mode, lr=lr)

    def _get_point_feed_dict(self, data, mode='training', lr=None):
        """ This is the pointwise feed-dict that is actually used.
        """
        data = zip(*data)
        labels = data[-1]
        soft_labels = np.array([[1 if t == i else 0
                            for i in range(self.args.num_class)] \
                            for t in labels])
        sig_labels = labels # rating

        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.learn_rate:lr,
            #self.dropout:self.args.dropout,
            #self.rnn_dropout:self.args.rnn_dropout,
            #self.emb_dropout:self.args.emb_dropout,
            self.soft_labels:soft_labels,
            self.sig_labels:sig_labels
        }
        #if(mode!='training'):
        #    feed_dict[self.dropout] = 1.0
        #    feed_dict[self.rnn_dropout] = 1.0
        #    feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
        return feed_dict

    def register_index_map(self, idx, target):
        self.imap[target] = idx


    def build_graph(self):
        ''' Builds Computational Graph
        '''
        len_shape = [None]

        print("Building placeholders with shape={}".format(len_shape))

        with self.graph.as_default():
            self.is_train = tf.get_variable("is_train",
                                            shape=[],
                                            dtype=tf.bool,
                                            trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)

            # user_id
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None], name='q1_inputs')
            # item_id
            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None], name='q1_inputs')

            '''
            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32,
                                                name='dropout')
                self.rnn_dropout = tf.placeholder(tf.float32,
                                                name='rnn_dropout')
                self.emb_dropout = tf.placeholder(tf.float32,
                                                name='emb_dropout')
            if(self.args.pretrained==1):
                self.emb_placeholder = tf.placeholder(tf.float32,
                            [self.vocab_size, self.args.emb_size])
            '''

            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')

            with tf.name_scope("soft_labels"):
                # softmax cross entropy (not used here)
                data_type = tf.int32
                self.soft_labels = tf.placeholder(data_type,
                             shape=[None, self.args.num_class],
                             name='softmax_labels')

            # self.y = tf.placeholder("float", [None], 'rating')
            with tf.name_scope("sig_labels"):
                # sigmoid cross entropy
                self.sig_labels = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name='sigmoid_labels')
                self.sig_target = tf.expand_dims(self.sig_labels, 1)

            self.batch_size = tf.shape(self.q1_inputs)[0]
            with tf.variable_scope('user_embedding_layer'):
                #self.P = tf.Variable(tf.random_normal([self.num_user, self.args.factor], stddev=0.01))
                self.P = tf.Variable(tf.random_uniform([self.num_user, self.args.factor], maxval=0.1))
            with tf.variable_scope('item_embedding_layer'):
                #self.Q = tf.Variable(tf.random_normal([self.num_item, self.args.factor], stddev=0.01))
                self.Q = tf.Variable(tf.random_uniform([self.num_item, self.args.factor], maxval=0.1))

            with tf.variable_scope('user_bias'):
                #self.B_U = tf.Variable(tf.random_normal([self.num_user], stddev=0.01))
                self.B_U = tf.Variable(tf.random_uniform([self.num_user], maxval=0.1))
            with tf.variable_scope('item_bias'):
                #self.B_I = tf.Variable(tf.random_normal([self.num_item], stddev=0.01))
                self.B_I = tf.Variable(tf.random_uniform([self.num_item], maxval=0.1))
            with tf.variable_scope('global_bias'):
                #self.B_G = tf.Variable(tf.random_normal([1], stddev=0.01))
                self.B_G = tf.Variable(tf.random_uniform([1], maxval=0.1))

            user_latent_factor = tf.nn.embedding_lookup(self.P, self.q1_inputs)
            item_latent_factor = tf.nn.embedding_lookup(self.Q, self.q2_inputs)
            user_bias = tf.nn.embedding_lookup(self.B_U, self.q1_inputs)
            item_bias = tf.nn.embedding_lookup(self.B_I, self.q2_inputs)

            self.output_pos = tf.reduce_sum(tf.multiply(user_latent_factor, item_latent_factor), 1) + user_bias + item_bias + self.B_G
            self.output_neg = None

            # Define loss and optimizer
            with tf.name_scope("train"):
                with tf.name_scope("cost_function"):
                    if("SOFT" in self.args.rnn_type):
                        target = self.soft_labels
                        if('POINT' in self.args.rnn_type):
                            target = tf.argmax(target, 1)
                            target = tf.expand_dims(target, 1)
                            target = tf.cast(target, tf.float32)
                            ce = tf.nn.sigmoid_cross_entropy_with_logits(
                                                logits=self.output_pos,
                                                labels=target)
                        else:
                            ce = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                    logits=self.output_pos,
                                                    labels=tf.stop_gradient(target))
                        self.cost = tf.reduce_mean(ce)
                    elif('RAW_MSE' in self.args.rnn_type):
                        sig = self.output_pos
                        target = tf.expand_dims(self.sig_labels, 1)
                        self.cost = tf.reduce_mean(
                                    tf.square(tf.subtract(target, sig)))
                    elif('LOG' in self.args.rnn_type):
                        # BPR loss for ranking
                        self.cost = tf.reduce_mean(
                                    -tf.log(tf.nn.sigmoid(
                                        self.output_pos-self.output_neg)))
                    else:
                        # Hinge loss for ranking
                        self.hinge_loss = tf.maximum(0.0,(
                                self.args.margin - self.output_pos \
                                + self.output_neg))

                        self.cost = tf.reduce_sum(self.hinge_loss)

                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            # only add reg on non-bias variables
                            #vars = tf.trainable_variables()
                            #lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                            #                    if 'bias' not in v.name ])
                            self.reg_loss = sum(map(tf.nn.l2_loss,[user_latent_factor, item_latent_factor])) * self.args.l2_reg
                            self.cost += self.reg_loss
                    tf.summary.scalar("cost_function", self.cost)
                global_step = tf.Variable(0, trainable=False)

                if(self.args.dev_lr>0):
                    lr = self.learn_rate
                else:
                    if(self.args.decay_steps>0):
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      global_step,
                                      self.args.decay_steps,
                                       self.args.decay_lr,
                                       staircase=self.args.decay_stairs)
                    elif(self.args.decay_lr>0 and self.args.decay_epoch>0):
                        decay_epoch = self.args.decay_epoch
                        lr = tf.train.exponential_decay(self.args.learn_rate,
                                      global_step,
                                      decay_epoch * self.args.batch_size,
                                       self.args.decay_lr, staircase=True)
                    else:
                        lr = self.args.learn_rate

                control_deps = []

                with tf.name_scope('optimizer'):
                    if(self.args.opt=='SGD'):
                        self.opt = tf.train.GradientDescentOptimizer(
                            learning_rate=lr)
                    elif(self.args.opt=='Adam'):
                        self.opt = tf.train.AdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='LazyAdam'):
                        self.opt = tf.contrib.opt.LazyAdamOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adadelta'):
                        self.opt = tf.train.AdadeltaOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='Adagrad'):
                        self.opt = tf.train.AdagradOptimizer(
                                        learning_rate=lr)
                    elif(self.args.opt=='RMS'):
                        self.opt = tf.train.RMSPropOptimizer(
                                    learning_rate=lr)
                    elif(self.args.opt=='Moment'):
                        self.opt = tf.train.MomentumOptimizer(lr, 0.9)

                    # Use SGD at the end for better local minima
                    self.opt2 = tf.train.GradientDescentOptimizer(
                            learning_rate=self.args.wiggle_lr)
                    tvars = tf.trainable_variables()
                    def _none_to_zero(grads, var_list):
                        return [grad if grad is not None else tf.zeros_like(var)
                              for var, grad in zip(var_list, grads)]
                    if(self.args.clip_norm>0):
                        grads, _ = tf.clip_by_global_norm(
                                        tf.gradients(self.cost, tvars),
                                        self.args.clip_norm)
                        with tf.name_scope('gradients'):
                            gradients = self.opt.compute_gradients(self.cost)
                            def ClipIfNotNone(grad):
                                if grad is None:
                                    return grad
                                grad = tf.clip_by_value(grad, -10, 10, name=None)
                                return tf.clip_by_norm(grad, self.args.clip_norm)
                            if(self.args.clip_norm>0):
                                clip_g = [(ClipIfNotNone(grad), var) for grad, var in gradients]
                            else:
                                clip_g = [(grad,var) for grad,var in gradients]

                        # Control dependency for center loss
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.apply_gradients(clip_g,
                                                global_step=global_step)
                            self.wiggle_op = self.opt2.apply_gradients(clip_g,
                                                global_step=global_step)
                    else:
                        with tf.control_dependencies(control_deps):
                            self.train_op = self.opt.minimize(self.cost)
                            self.wiggle_op = self.opt2.minimize(self.cost)

                self.grads = _none_to_zero(tf.gradients(self.cost,tvars), tvars)
                # grads_hist = [tf.summary.histogram("grads_{}".format(i), k) for i, k in enumerate(self.grads) if k is not None]
                self.merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
                # model_stats()

                 # for Inference
                self.predict_op = self.output_pos
                if('RAW_MSE' in self.args.rnn_type):
                    self.predict_op = tf.clip_by_value(self.predict_op, 1, 5)
                if('SOFT' in self.args.rnn_type):
                    if('POINT' in self.args.rnn_type):
                        predict_neg = 1 - self.predict_op
                        self.predict_op = tf.concat([predict_neg,
                                         self.predict_op], 1)
                    else:
                        self.predict_op = tf.nn.softmax(self.output_pos)
                    self.predictions = tf.argmax(self.predict_op, 1)
                    self.correct_prediction = tf.equal(tf.argmax(self.predict_op, 1),
                                                    tf.argmax(self.soft_labels, 1))
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,
                                                    "float"))
