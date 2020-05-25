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

class ExpModel:
    ''' Base model class.
    Multitask - rating prediction and experience ranking
    '''
    def __init__(self, vocab_size, args, char_vocab=0, pos_vocab=0,
                    mode='RATING+RANK', num_user=0, num_item=0):
        self.vocab_size = vocab_size
        self.char_vocab = char_vocab
        self.pos_vocab = pos_vocab
        self.graph = tf.Graph()
        self.args = args
        self.imap = {}
        self.inspect_op = []
        self.mode=mode
        self.write_dict = {}
        # For interaction data only (disabled and removed from this repo)
        self.num_user = num_user
        self.num_item = num_item
        print('Creating Model in [{}] mode'.format(self.mode))
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
        self.temp = []
        self.att1, self.att2 = [],[]
  
        # build graph
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
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout
        }
        if(mode=='training'):
            feed_dict[self.q3_inputs] = data[self.imap['q3_inputs']]
            feed_dict[self.q3_len]=data[self.imap['q3_len']]
        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
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
        labels = data[-1] # rating 
        soft_labels = np.array([[1 if t == i else 0
                            for i in range(self.args.num_class)] \
                            for t in labels]) # softmax label?
        sig_labels = labels

        if(lr is None):
            lr = self.args.learn_rate
        feed_dict = {
            self.q1_inputs:data[self.imap['q1_inputs']],
            self.q2_inputs:data[self.imap['q2_inputs']],
            self.q1_len:data[self.imap['q1_len']],
            self.q2_len:data[self.imap['q2_len']],
            self.learn_rate:lr,
            self.dropout:self.args.dropout,
            self.rnn_dropout:self.args.rnn_dropout,
            self.emb_dropout:self.args.emb_dropout,
            self.soft_labels:soft_labels,
            self.sig_labels:sig_labels
        }
        if('TNET' in self.args.rnn_type):
            # Use TransNet
            feed_dict[self.trans_inputs] = data[self.imap['trans_inputs']]
            feed_dict[self.trans_len] = data[self.imap['trans_len']]
        if('EXP' in self.args.rnn_type):
            # Add experience ranking
            feed_dict[self.user_idx] = data[self.imap['user_idx']]
            feed_dict[self.item_idx] = data[self.imap['item_idx']]
            feed_dict[self.pair_user_inputs2] = data[self.imap['pair_user_inputs2']]
            feed_dict[self.pair_user_len2] = data[self.imap['pair_user_len2']]
            feed_dict[self.exp_labels] = data[self.imap['exp_labels']]

        if(mode!='training'):
            feed_dict[self.dropout] = 1.0
            feed_dict[self.rnn_dropout] = 1.0
            feed_dict[self.emb_dropout] = 1.0
        if(self.args.features):
            feed_dict[self.pos_features] = data[6]
        return feed_dict

    def register_index_map(self, idx, target):
        self.imap[target] = idx

    # representation
    def _joint_representation(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                    q2_max, force_model=None, score=1,
                    reuse=None, features=None, extract_embed=False,
                    side='', c1_embed=None, c2_embed=None, p1_embed=None,
                    p2_embed=None, i1_embed=None, i2_embed=None, o1_embed=None,
                    o2_embed=None, o1_len=None, o2_len=None, q1_mask=None,
                    q2_mask=None):
        """ Learns a joint representation given q1 and q2.
        """

        print("Learning Repr [{}]".format(side))
        print(self.q1_embed)
        print(self.q2_embed)

        # Extra projection layer
        if('HP' in self.args.rnn_type):
            # Review level Highway layer
            use_mode='HIGH'
        else:
            use_mode='FC'

        # projection - word_dim -> h_dim
        if(self.args.translate_proj==1):
            q1_embed = projection_layer(
                    q1_embed,
                    self.args.rnn_size,
                    name='trans_proj',
                    activation=tf.nn.relu,
                    initializer=self.initializer,
                    dropout=self.args.dropout,
                    reuse=reuse,
                    use_mode=use_mode,
                    num_layers=self.args.num_proj,
                    return_weights=True,
                    is_train=self.is_train
                    )
            q2_embed = projection_layer(
                    q2_embed,
                    self.args.rnn_size,
                    name='trans_proj',
                    activation=tf.nn.relu,
                    initializer=self.initializer,
                    dropout=self.args.dropout,
                    reuse=True,
                    use_mode=use_mode,
                    num_layers=self.args.num_proj,
                    is_train=self.is_train
                    )
        else:
            self.proj_weights = self.embeddings

        if(self.args.all_dropout):
            q1_embed = tf.nn.dropout(q1_embed, self.dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.dropout)

        representation = None
        att1, att2 = None, None
        if(force_model is not None):
            rnn_type = force_model
        else:
            rnn_type = self.args.rnn_type
        rnn_size = self.args.rnn_size
        q1_output = self.learn_single_repr(q1_embed, q1_len, q1_max,
                                            rnn_type,
                                            reuse=reuse, pool=False,
                                            name='main', mask=q1_mask)
        q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
                                            rnn_type,
                                            reuse=True, pool=False,
                                            name='main', mask=q2_mask)
        print("==============================================")
        print('Single Repr:')
        print(q1_output)
        print(q2_output)
        print("===============================================")
        if('DUAL' in rnn_type):
            # D-ATT model
            q1_output = dual_attention(q1_output, self.args.rnn_size,
                                        initializer=self.initializer,
                                        reuse=reuse, dropout=self.dropout)
            q2_output = dual_attention(q2_output, self.args.rnn_size,
                                        initializer=self.initializer,
                                        reuse=True, dropout=self.dropout)
            if(side=='POS'):
                self.temp = []
        elif('MPCN' in rnn_type):
            # activate MPCN model
            q1_output, q2_output = multi_pointer_coattention_networks(
                                                self,
                                                q1_output, q2_output,
                                                q1_len, q2_len,
                                                o1_embed, o2_embed,
                                                o1_len, o2_len,
                                                rnn_type=self.args.rnn_type,
                                                reuse=reuse)
        else:
            if('MEAN' in rnn_type):
                # Standard Mean Over Time Baseline
                q1_len = tf.expand_dims(q1_len, 1)
                q2_len = tf.expand_dims(q2_len, 1)
                q1_output = mean_over_time(q1_output, q1_len)
                q2_output = mean_over_time(q2_output, q2_len)
            elif('SUM' in rnn_type):
                q1_output = tf.reduce_sum(q1_output, 1)
                q2_output = tf.reduce_sum(q2_output, 1)
            elif('MAX' in rnn_type):
                q1_output = tf.reduce_max(q1_output, 1)
                q2_output = tf.reduce_max(q2_output, 1)
            elif('LAST' in rnn_type):
                q1_output = last_relevant(q1_output, q1_len)
                q2_output = last_relevant(q2_output, q2_len)
            elif('MM' in rnn_type):
                # max mean pooling
                q1_len = tf.expand_dims(q1_len, 1)
                q2_len = tf.expand_dims(q2_len, 1)
                q1_mean = mean_over_time(q1_output, q1_len)
                q2_mean = mean_over_time(q2_output, q2_len)
                q1_max = tf.reduce_max(q1_output, 1)
                q2_max = tf.reduce_max(q2_output, 1)
                q1_output = tf.concat([q1_mean, q1_max], 1)
                q2_output = tf.concat([q2_mean, q2_max], 1)
        try:
            # For summary statistics
            self.max_norm = tf.reduce_max(tf.norm(q1_output,
                                        ord='euclidean',
                                        keep_dims=True, axis=1))
        except:
            self.max_norm = 0

        if(extract_embed):
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        q1_output = tf.nn.dropout(q1_output, self.dropout)
        q2_output = tf.nn.dropout(q2_output, self.dropout)

        # 
        if(self.mode=='HREC'):
            # Use Rec Style output
            if('TNET' not in self.args.rnn_type):
                output = self._rec_output(q1_output, q2_output,
                                        reuse=reuse,
                                        side=side)
            elif("TNET" in self.args.rnn_type):
                 # Learn Repr with CNN
                input_vec = tf.concat([q1_output, q2_output], 1)
                dim = q1_output.get_shape().as_list()[1]
                trans_output = ffn(input_vec, dim,
                          self.initializer, name='transform',
                          reuse=reuse,
                          num_layers=2,
                          dropout=None, activation=tf.nn.tanh)
                trans_cnn = self.learn_single_repr(self.trans_embed,
                                                 self.trans_len,
                                                 self.args.smax * 2,
                                                 rnn_type,
                                                 reuse=True, pool=False,
                                                 name='main')
                trans_cnn = tf.reduce_max(trans_cnn, 1)
                self.trans_loss = tf.nn.l2_loss(trans_output - trans_cnn)
                # Alternative predict op using transform
                output = self._rec_output(trans_output, None,
                                            reuse=reuse,
                                            side=side,
                                            name='target')

        representation = output
        return output, representation, att1, att2

    def learn_single_repr(self, q1_embed, q1_len, q1_max, rnn_type,
                        reuse=None, pool=False, name="", mask=None):
        """ This is the single sequence encoder function.
        rnn_type controls what type of encoder is used.
        Supports neural bag-of-words (NBOW) and CNN encoder
        """
        if('NBOW' in rnn_type):
            q1_output = tf.reduce_sum(q1_embed, 1)
            if(pool):
                return q1_embed, q1_output
        elif('CNN' in rnn_type):
            q1_output = build_raw_cnn(q1_embed, self.args.rnn_size,
                filter_sizes=3,
                initializer=self.initializer,
                dropout=self.rnn_dropout, reuse=reuse, name=name) # reuse and name?
            if(pool):
                q1_output = tf.reduce_max(q1_output, 1)
                return q1_embed, q1_output
                #return q1_output, q1_output
        else: # if rnn_type is some kind of rnn, do nothing?
            q1_output = q1_embed

        return q1_output

    def _rec_output(self, q1_output, q2_output, reuse=None, side="",
                        name=''):
        """ This function supports the final layer outputs of
        recommender models.

        Four options: 'DOT','MLP','MF' and 'FM'
        (should be self-explanatory)
        """
        print("Rec Output")
        print(q1_output)
        dim = q1_output.get_shape().as_list()[1]
        with tf.variable_scope('rec_out', reuse=reuse) as scope:
            if('DOT' in self.args.rnn_type):
                output = q1_output * q2_output
                output = tf.reduce_sum(output, 1, keep_dims=True)
            elif('MLP' in self.args.rnn_type):
                output = tf.concat([q1_output, q2_output,
                                q1_output * q2_output], 1)
                output = ffn(output, self.args.hdim,
                            self.initializer,
                            name='ffn', reuse=None,
                            dropout=self.dropout,
                            activation=tf.nn.relu, num_layers=2)
                output = linear(output, 1, self.initializer)
            elif('MF' in self.args.rnn_type):
                output = q1_output * q2_output
                h = tf.get_variable(
                            "hidden", [dim, 1],
                            initializer=self.initializer,
                            )
                output = tf.matmul(output, h)
            elif('FM' in self.args.rnn_type):
                if(q2_output is None):
                    input_vec = q1_output
                else:
                    input_vec = tf.concat([q1_output, q2_output], 1)
                input_vec = tf.nn.dropout(input_vec, self.dropout)
                output, _ = build_fm(input_vec, k=self.args.factor,
                                    reuse=reuse,
                                    name=name,
                                    initializer=self.initializer,
                                    reshape=False)

            if('SIG' in self.args.rnn_type):
                output = tf.nn.sigmoid(output)
            return output

    def prepare_hierarchical_input(self):
        """ Supports hierarchical data input
        Converts word level -> sentence level
        """
        # tylib/lib/seq_op/clip_sentence
        # q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
        # q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
        # q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)

        # Build word-level masks
        self.q1_mask = tf.cast(self.q1_inputs, tf.bool)
        self.q2_mask = tf.cast(self.q2_inputs, tf.bool)
        self.q3_mask = tf.cast(self.q3_inputs, tf.bool)

        def make_hmasks(inputs, smax):
            # Hierarchical Masks
            # Inputs are bsz x (dmax * smax)
            inputs = tf.reshape(inputs,[-1, smax]) # -> (bsz * dmax) x smax
            masked_inputs = tf.cast(inputs, tf.bool)
            return masked_inputs

        # Build review-level masks
        self.q1_hmask = make_hmasks(self.q1_inputs, self.args.smax)
        self.q2_hmask = make_hmasks(self.q2_inputs, self.args.smax)
        self.q3_hmask = make_hmasks(self.q3_inputs, self.args.smax)

        with tf.device('/cpu:0'):
            q1_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                self.q1_inputs)
            q2_embed =  tf.nn.embedding_lookup(self.embeddings,
                                                self.q2_inputs)
            q3_embed = tf.nn.embedding_lookup(self.embeddings,
                                                self.q3_inputs)

        print("=============================================")
        # This is found in nn.py in tylib
        print("Hierarchical Flattening")
        q1_embed, q1_len = hierarchical_flatten(q1_embed,
                                            self.q1_len,
                                            self.args.smax)
        q2_embed, q2_len = hierarchical_flatten(q2_embed,
                                            self.q2_len,
                                            self.args.smax)
        q3_embed, q3_len = hierarchical_flatten(q3_embed,
                                            self.q3_len,
                                            self.args.smax)
        # After flatten -> (bsz * dmax) x smax x dim

        # o_emb is q_emb before the learn_single_repr layer
        self.o1_embed = q1_embed
        self.o2_embed = q2_embed
        self.o3_embed = q3_embed
        self.o1_len = q1_len
        self.o2_len = q2_len
        self.o3_len = q3_len
        _, q1_embed = self.learn_single_repr(q1_embed, q1_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=None, pool=True,
                                            name='sent', mask=self.q1_hmask)
        _, q2_embed = self.learn_single_repr(q2_embed, q2_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=self.q2_hmask)
        _, q3_embed = self.learn_single_repr(q3_embed, q3_len, self.args.smax,
                                            self.args.base_encoder,
                                            reuse=True, pool=True,
                                            name='sent', mask=self.q3_hmask)
        # According to the paper, each review is represented as a sum of its constituent word embeddings
        # Therefore, q_emb is summed over seq_len dimension -> (bsz * dmax) x dim 
        _dim = q1_embed.get_shape().as_list()[1]
        q1_embed = tf.reshape(q1_embed, [-1, self.args.dmax, _dim])
        q2_embed = tf.reshape(q2_embed, [-1, self.args.dmax, _dim])
        q3_embed = tf.reshape(q3_embed, [-1, self.args.dmax, _dim])
        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed
        # set value
        self.qmax = self.args.dmax
        self.a1max = self.args.dmax
        self.a2max = self.args.dmax
        # Doesn't support any of these yet
        self.c1_cnn, self.c2_cnn, self.c3_cnn = None, None, None
        self.p1_pos, self.p2_pos, self.p3_pos = None, None, None
        if('TNET' in self.args.rnn_type):
            t_inputs, _ = clip_sentence(self.trans_inputs, self.trans_len)
            self.trans_embed = tf.nn.embedding_lookup(self.embeddings,
                                                        t_inputs)
        print("=================================================")

    # prepare flat input
    def prepare_inputs(self):
        """ Prepares Input
        """
        #q1_inputs, self.qmax = clip_sentence(self.q1_inputs, self.q1_len)
        #q2_inputs, self.a1max = clip_sentence(self.q2_inputs, self.q2_len)
        #q3_inputs, self.a2max = clip_sentence(self.q3_inputs, self.q3_len)
        q1_inputs = self.q1_inputs
        q2_inputs = self.q2_inputs
        q3_inputs = self.q3_inputs
        self.qmax = self.args.dmax * self.args.smax
        self.a1max = self.args.dmax * self.args.smax
        self.a2max = self.args.dmax * self.args.smax

        self.q1_mask = tf.cast(q1_inputs, tf.bool)
        self.q2_mask = tf.cast(q2_inputs, tf.bool)
        self.q3_mask = tf.cast(q3_inputs, tf.bool)

        with tf.device('/cpu:0'):
            q1_embed = tf.nn.embedding_lookup(self.embeddings,
                                                    q1_inputs)
            q2_embed = tf.nn.embedding_lookup(self.embeddings,
                                                    q2_inputs)
            q3_embed = tf.nn.embedding_lookup(self.embeddings,
                                                    q3_inputs)

        if(self.args.all_dropout):
            # By default, this is disabled
            q1_embed = tf.nn.dropout(q1_embed, self.emb_dropout)
            q2_embed = tf.nn.dropout(q2_embed, self.emb_dropout)
            q3_embed = tf.nn.dropout(q3_embed, self.emb_dropout)

        # Ignore these. :)
        self.c1_cnn, self.c2_cnn, self.c3_cnn = None, None, None
        self.p1_pos, self.p2_pos, self.p3_pos = None, None, None

        if('TNET' in self.args.rnn_type):
            t_inputs, _ = clip_sentence(self.trans_inputs, self.trans_len)
            self.trans_embed = tf.nn.embedding_lookup(self.embeddings,
                                                        t_inputs)

        if('EXP' in self.args.rnn_type):
            #p_user_inputs1, self.pair_user_q1max = clip_sentence(self.pair_user_inputs1, self.pair_user_len1)
            #p_user_inputs2, self.pair_user_q2max = clip_sentence(self.pair_user_inputs2, self.pair_user_len2)

            self.batch_size = tf.shape(self.q1_inputs)[0]

            # B x N x T -> B*N x T            
            p_user_inputs2 = tf.reshape(self.pair_user_inputs2, [self.batch_size*self.args.num_neg, -1])
            self.pair_user_len2 = tf.reshape(self.pair_user_len2, [self.batch_size*self.args.num_neg, -1])

            self.pair_user_q2max = self.args.dmax * self.args.smax            
            pair_user_q2_embed = tf.nn.embedding_lookup(self.embeddings,
                                                        p_user_inputs2)
            self.pair_user_q2_embed = tf.nn.dropout(pair_user_q2_embed, self.emb_dropout)
            self.pair_user_q2_mask = tf.cast(p_user_inputs2, tf.bool)

        self.q1_embed = q1_embed
        self.q2_embed = q2_embed
        self.q3_embed = q3_embed

    def build_graph(self):
        ''' Builds Computational Graph
        '''
        if(self.mode=='HREC' and self.args.base_encoder!='Flat'): # hierarchical model and not flat base_encoder - then len is 2d
            len_shape = [None, None]
        else:
            len_shape = [None]

        print("Building placeholders with shape={}".format(len_shape))

        with self.graph.as_default():
            self.is_train = tf.get_variable("is_train",
                                            shape=[],
                                            dtype=tf.bool,
                                            trainable=False)
            self.true = tf.constant(True, dtype=tf.bool)
            self.false = tf.constant(False, dtype=tf.bool)
            with tf.name_scope('q1_input'):
                self.q1_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.qmax], # if qmax changes, tensor shape will also change 
                                                    name='q1_inputs')
            with tf.name_scope('q2_input'):
                self.q2_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q2_inputs')
            with tf.name_scope('q3_input'): # could not use
                # supports pairwise mode.
                self.q3_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.amax],
                                                    name='q3_inputs')


            self.pair_user_q2max = self.args.qmax
            # if('EXP' in self.args.rnn_type):
            # user_id
            with tf.name_scope('user_idx'):
                self.user_idx = tf.placeholder(tf.int32, shape=[None], name='user_idx')
            # item_id
            with tf.name_scope('item_idx'):
                self.item_idx = tf.placeholder(tf.int32, shape=[None], name='item_idx')

            # add pair input
            with tf.name_scope('pair_user_input2'):
                self.pair_user_inputs2 = tf.placeholder(tf.int32, shape=[None, self.args.num_neg,
                                                    self.pair_user_q2max],
                                                    name='pair_user_inputs2')
            with tf.name_scope('pair_user_lengths2'):
                self.pair_user_len2 = tf.placeholder(tf.int32, shape=[None, None])

            with tf.name_scope('exp_labels'):
                self.exp_labels = tf.placeholder(tf.float32, shape=[None, self.args.num_neg], name='exp_labels')

            with tf.name_scope('dropout'):
                self.dropout = tf.placeholder(tf.float32,
                                                name='dropout')
                self.rnn_dropout = tf.placeholder(tf.float32,
                                                name='rnn_dropout')
                self.emb_dropout = tf.placeholder(tf.float32,
                                                name='emb_dropout')
            with tf.name_scope('q1_lengths'):
                self.q1_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('q2_lengths'):
                self.q2_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('q3_lengths'):
                self.q3_len = tf.placeholder(tf.int32, shape=len_shape)
            with tf.name_scope('learn_rate'):
                self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
            if(self.args.pretrained==1):
                self.emb_placeholder = tf.placeholder(tf.float32,
                            [self.vocab_size, self.args.emb_size])

            with tf.name_scope("soft_labels"):
                # softmax cross entropy (not used here)
                data_type = tf.int32
                self.soft_labels = tf.placeholder(data_type,
                             shape=[None, self.args.num_class],
                             name='softmax_labels')

            with tf.name_scope("sig_labels"):
                # sigmoid cross entropy
                self.sig_labels = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name='sigmoid_labels')
                self.sig_target = tf.expand_dims(self.sig_labels, 1)

            self.batch_size = tf.shape(self.q1_inputs)[0]

            with tf.variable_scope('embedding_layer'):
                if(self.args.pretrained==1):
                    self.embeddings = tf.Variable(tf.constant(
                                        0.0, shape=[self.vocab_size,
                                            self.args.emb_size]), \
                                        trainable=self.args.trainable,
                                         name="embeddings")
                    self.embeddings_init = self.embeddings.assign(
                                        self.emb_placeholder)
                else:
                    self.embeddings = tf.get_variable('embedding',
                                        [self.vocab_size,
                                        self.args.emb_size],
                                        initializer=self.initializer)

            self.i1_embed, self.i2_embed, self.i3_embed = None, None, None

            if('TNET' in self.args.rnn_type):
                self.trans_inputs = tf.placeholder(tf.int32, shape=[None,
                                                    self.args.smax * 2],
                                                    name='trans_inputs')
                self.trans_len = tf.placeholder(tf.int32, shape=[None])

            # prepare inputs
            if(self.mode=='HREC' and self.args.base_encoder!='Flat'):
                # Hierarchical mode
                self.prepare_hierarchical_input() # build o1, o2, o3?

                q1_len = tf.cast(tf.count_nonzero(self.q1_len, axis=1),
                                    tf.int32)
                q2_len = tf.cast(tf.count_nonzero(self.q2_len, axis=1),
                                    tf.int32)
                q3_len = tf.cast(tf.count_nonzero(self.q3_len, axis=1),
                                    tf.int32)
            else:
                print("Flat Mode..")
                self.prepare_inputs()
                q1_len = self.q1_len
                q2_len = self.q2_len
                q3_len = self.q3_len
                # o_emb are hierarchical embeddings. therefore in the flat mode, they do not exist!!!
                self.o1_embed = None
                self.o2_embed = None
                self.o3_embed = None
                self.o1_len = None
                self.o2_len = None
                self.o3_len = None

            # build model
            # experience-aware latent factor model
            # first version - only consider text to predict experience

            rnn_type = self.args.rnn_type

            # user text -> experience emb
            # reuse and do pooling: B x T x h -> B x h
            _, q1_output = self.learn_single_repr(self.q1_embed, q1_len, self.qmax,
                                            rnn_type,
                                            reuse=None, 
                                            pool=True,
                                            name='main', mask=self.q1_mask)
            #q2_output = self.learn_single_repr(q2_embed, q2_len, q2_max,
            #                                rnn_type,
            #                                reuse=True, pool=False,
            #                                name='main', mask=q2_mask)

            _, pair_user_q2_output = self.learn_single_repr(self.pair_user_q2_embed, self.pair_user_len2, self.pair_user_q2max,
                                            rnn_type,
                                            reuse=True, 
                                            pool=True,
                                            name='main', mask=self.pair_user_q2_mask)

            print("---Single Representation---")

            # rating
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

            user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_idx)
            item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_idx)
            user_bias = tf.nn.embedding_lookup(self.B_U, self.user_idx)
            item_bias = tf.nn.embedding_lookup(self.B_I, self.item_idx)

            # experience 
            self.exp_output_pos = q1_output
            self.exp_output_pos = ffn(q1_output, self.args.hdim,
                            self.initializer,
                            name='ffn', reuse=None,
                            dropout=self.dropout,
                            activation=tf.nn.relu, num_layers=1)
            self.exp_output_pos = linear(self.exp_output_pos, 1, self.initializer, name='exp_pos_proj')
 
            self.exp_output_neg = pair_user_q2_output
            self.exp_output_neg = ffn(pair_user_q2_output, self.args.hdim,
                            self.initializer,
                            name='ffn', reuse=True,
                            dropout=self.dropout,
                            activation=tf.nn.relu, num_layers=1)
            self.exp_output_neg = linear(self.exp_output_neg, 1, self.initializer, name='exp_pos_proj', reuse=True)


            # use exp_output_pos to learn experience emb, then concatenate with latent factor model parameters
            exp_B_G = tf.concat([self.exp_output_pos, tf.tile(tf.expand_dims(self.B_G, 1), [self.batch_size,1])], 1)
            exp_B_G = linear(exp_B_G, 1, self.initializer, name='global_proj')
            exp_user_bias = tf.concat([self.exp_output_pos, tf.expand_dims(user_bias, 1)], 1)
            exp_user_bias = linear(exp_user_bias, 1, self.initializer, name='user_proj')
            exp_item_bias = tf.concat([self.exp_output_pos, tf.expand_dims(item_bias, 1)], 1)
            exp_item_bias = linear(exp_item_bias, 1, self.initializer, name='item_proj')
            exp_user_latent_factor = tf.concat([self.exp_output_pos, user_latent_factor], 1)
            exp_user_latent_factor = linear(exp_user_latent_factor, self.args.factor, self.initializer, name='user_latent_proj')
            exp_item_latent_factor = tf.concat([self.exp_output_pos, item_latent_factor], 1)
            exp_item_latent_factor = linear(exp_item_latent_factor, self.args.factor, self.initializer, name='item_latent_proj')

            self.output_pos = (tf.reduce_sum(tf.multiply(exp_user_latent_factor, exp_item_latent_factor), 1) 
                              + exp_user_bias + exp_item_bias + exp_B_G)


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
                        self.r_cost = tf.reduce_mean(
                                    tf.square(tf.subtract(target, sig)))
                        # add experience ranking loss
                        if('EXP' in self.args.rnn_type):
                            lambda1 = 20

                            # exp_output_pos also need to tile: B x 1 -> B x N -> (B*N) x 1            
                            self.exp_output_pos = tf.tile(self.exp_output_pos, [1, self.args.num_neg])
                            self.exp_output_pos = tf.reshape(self.exp_output_pos, [-1, 1])
                            self.exp_labels2 = tf.reshape(self.exp_labels, [-1, 1])

                            print(self.exp_output_pos.shape)
                            print(self.exp_output_neg.shape)
                            print(self.exp_labels2.shape)

                            self.exp_cost = tf.reduce_mean(
                                        -tf.log(tf.nn.sigmoid(
                                            (self.exp_output_pos-self.exp_output_neg) * self.exp_labels2)))
                            self.cost = self.r_cost + self.args.lambda1 * self.exp_cost
 
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

                    # add reg
                    with tf.name_scope('regularization'):
                        if(self.args.l2_reg>0):
                            vars = tf.trainable_variables()
                            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars \
                                                if '_bias' not in v.name ]) # vars other than user_bias, item_bias, global_bias
                            lossL2 *= self.args.l2_reg
                            self.cost += lossL2

                    # add cost function
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
                self.predict_op2 = self.exp_output_pos

                if('RAW_MSE' in self.args.rnn_type):
                    self.predict_op = tf.clip_by_value(self.predict_op, 1, 5)
                # for classification task
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

                self.exp_output_pos = tf.squeeze(self.exp_output_pos, 1)
                self.exp_output_neg = tf.squeeze(self.exp_output_neg, 1)
                self.exp_labels2 = tf.squeeze(self.exp_labels2, 1)


