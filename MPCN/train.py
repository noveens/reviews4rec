from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import csv
import argparse
from keras.preprocessing import sequence
from datetime import datetime
import numpy as np
import random
np.random.seed(1337)
random.seed(1337)
import os
from tqdm import tqdm
from utilities import *
#from metrics import *
import time
import tensorflow as tf
import sys
from sklearn.utils import shuffle
from collections import Counter
import cPickle as pickle
from keras.utils import np_utils
import visdom
import string
import re
import math
import operator
from utilities import *
from collections import defaultdict
import sys
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from tf_models.model import Model
from tf_models.exp_model import ExpModel
from tf_models.rec_model import RecModel
from tylib.exp.experiment import Experiment
from tylib.exp.exp_ops import *
from tylib.exp.metrics import *
from parser import *
from sklearn.metrics import mean_absolute_error

from collections import defaultdict 

# from rec_config import *

reload(sys)
sys.setdefaultencoding('UTF8')

def batchify(data, i, bsz, max_sample):
    start = int(i * bsz)
    end = int(i * bsz) + bsz
    if(end>max_sample):
        end = max_sample
    data = data[start:end]
    return data

class CFExperiment(Experiment):
    """ Main experiment class for collaborative filtering.

    Check tylib/exp/experiment.py for base class.
    """

    def __init__(self, inject_params=None):
        print("Starting Rec Experiment")
        super(CFExperiment, self).__init__()
        # self.uuid = datetime.now().strftime("%d:%m:%H:%M:%S")
        self.uuid = "log"
        self.parser = build_parser()
        self.no_text_mode = False
        self.char_index = {}    # Not supported here
        self.pos_index = {}     # Not supported here

        self.args = self.parser.parse_args()

        if('BREC' in self.args.rnn_type):
            self.no_text_mode=True
            # this supports standard interaction-only recommender models
            # currently disabled.
            print("Found Baseline Model. Setting to No-Text Mode")

        # get config for each dataset. max_val / min_val are used for normalize scores -> [0,5]?
        #self.max_val, self.min_val, self.args.data_link = get_rec_config(
        #                                        self.args.dataset)
        self.max_val, self.min_val, self.args.data_link = 5.0, 1.0, "." 

        # self.show_metrics = ['MSE','RMSE','MAE']
        ### MINE
        self.show_metrics = ['MSE']
        self.eval_primary = 'MSE'
        # For hierarchical setting
        self.args.qmax = self.args.smax * self.args.dmax
        self.args.amax = self.args.smax * self.args.dmax


        ############## MINE
        print("Setting up environment..")
        data_path = self.args.dataset + 'env.gz'
        # if(self.args.data_link!=""):
            # print("[Starting Data Link..]")
            # this is used to connect to a legacy repo
            # data_path = '{}/datasets/{}/env.gz'.format(self.args.data_link,
                                            # self.args.dataset)
        # else:
            # data_path  = './datasets/{}/env.gz'.format(
                # self.args.dataset)

        self.model_wrapper()

        self.env = dictFromFileUnicode(data_path)
        self.model_name = self.args.rnn_type
        self._setup()

        if(inject_params is not None):
            for param, val in inject_params.items():
                setattr(self.args, param, val)
                self.write_to_file("[Injection] {} to {}".format(
                                                param, val))
        self._load_sets()

        try:
            self.num_users = len(self.env['user_index'])
            self.num_items = len(self.env['item_index'])
        except:
            self.num_users = self.env['num_users']
            self.num_items = self.env['num_items']
        if('movie' in self.args.dataset):
            self.num_users +=1
            self.num_items +=1

        if(self.no_text_mode):
            print("Users={} Items={}".format(self.num_users, self.num_items))
            self.mdl = RecModel(self.num_users, self.num_items,
                                 self.args)
        else:
            if('EXP' in self.args.rnn_type):
                self.mdl = ExpModel(self.vocab, self.args,
                            char_vocab=len(self.char_index),
                            pos_vocab=len(self.pos_index),
                            num_item=self.num_items,
                            num_user=self.num_users)
            else:
                self.mdl = Model(self.vocab, self.args,
                            char_vocab=len(self.char_index),
                            pos_vocab=len(self.pos_index),
                            mode='HREC', num_item=self.num_items, # HREC?
                            num_user=self.num_users)

        self._print_model_stats()
        self.hyp_str = self.model_name + '_' + self.uuid
        self._setup_tf(load_embeddings=not self.no_text_mode) # no_text_mode False -> load 

 
        # define sequences
        if(self.no_text_mode==False): # load text repr for user/item
            self.user_repr = self.repr_convert(self.env['user_text'])
            self.item_repr = self.repr_convert(self.env['item_text'])
            if('TNET' in self.args.rnn_type):
                print("repr Convert for TNET") # 
                self.user_repr2 = self.repr_convert(self.env['user_text2'])

    def model_wrapper(self):
        """ Converts model name to consituent components.
        """
        original = self.args.rnn_type
        if(self.args.rnn_type=='DeepCoNN'):
            self.args.rnn_type = 'RAW_MSE_MAX_CNN_FM'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='TRANSNET'):
            self.args.rnn_type = 'RAW_MSE_MAX_CNN_FM_TNET'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='DATT'):
            self.args.rnn_type ='RAW_MSE_DUAL_DOT'
            self.args.base_encoder = 'Flat'
        elif(self.args.rnn_type=='MPCN'):
            self.args.rnn_type = 'RAW_MSE_MPCN_FN_FM'
            self.args.base_encoder = 'NBOW'

        print("Conversion to {} | base:{}".format(
                                self.args.rnn_type,
                                self.args.base_encoder))

    # accelerate!
    def repr_convert(self,repr_dict):
        lengths = []
        lengths2 = []
        for key, value in tqdm(repr_dict.items(), desc='repr convert'):
            _tmp = [[int(y) for y in x.split()] for x in value]
            repr_dict[key] = _tmp # list of reviews 
            lengths.append(len(_tmp)) # append number of reviews
            lengths2 += [len(x) for x in _tmp] # extend length of each review
        show_stats('num review', lengths)
        show_stats('avg review', lengths2)
        return repr_dict

    def _label_scaler(self, labels):

        def movie_scaler(x):
            # Some adapter to deal with legacy format
            return ((x * 5) - 1) / (4)

        if('movie' in self.args.dataset):
            print("Movie Scaler.")
            print(np.min(labels))
            print(np.max(labels))
            labels = [movie_scaler(x) for x in labels]

        return labels

    def _prepare_base_set(self, data):
        print("Preparing Base Set")
        lbls = [x[2] for x in data]
        min_labels = np.min(lbls)
        if(min_labels<0):
            data = [x for x in data if x[2]>=0]
        else:
            data = [x for x in data if x[2]>0]
        user = [x[0] for x in data]
        item = [x[1] for x in data]
        labels = [x[2] for x in data]
        # labels = self._label_scaler(labels)
        self._majority_baseline(labels)
        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q2_inputs')
        output = [user, item]
        def normalize_labels(x, max_val, min_val):
            return (x - min_val) / (max_val - min_val)

        max_val = np.max(labels)
        min_val = np.min(labels)

        need_scaling = ['yelp17','netflixPrize']
        if(self.args.dataset in need_scaling and 'RAW' not in self.args.rnn_type):
            print("Scaling dataset..")
            labels = [normalize_labels(x, self.max_val, self.min_val) \
                            for x in labels]
        print(np.max(labels))
        print(np.min(labels))
        output.append(labels)
        output = zip(*output)
        return output

    def prepare_set(self, data):
        if(self.no_text_mode):
            return self._prepare_base_set(data)
        else:
            return self._prepare_text_set(data)

    def _majority_baseline(self, labels):
        print("============================================")
        print("Running Majority Baseline...")
        _stat_pred = [abs(math.floor(x)) for x in labels]
        count = Counter(_stat_pred)
        print(count)
        max_class = count.most_common(5)[0][0]
        _majority = [float(max_class) for i in range(len(labels))]
        print('MSE={}'.format(mean_squared_error(_majority, labels)))
        print("============================================")

    # prepare text based model
    def _prepare_text_set(self, data):
        # prepares dataset with negative sampling
        print("Preparing Text Set...")
        self.char_pad_token = [0 for i in range(self.args.char_max)]

        def word2id(word):
            try:
                return self.word_index[word]
            except:
                return 1

        def sent2char(sent, pad_max):
            def word2char(word):
                word = [self.char_index[x] for x in word]
                word = pad_to_max(word, self.args.char_max)
                return word
            sent_chars = [word2char(x) for x in sent]
            pad_token = [0 for i in range(self.args.char_max)]
            sent_chars = pad_to_max(sent_chars, pad_max,
                            pad_token=pad_token)
            return sent_chars

        def text2ids(txt):
            txt = [x for x in txt if len(x)>0]
            if(self.args.use_lower):
                txt = [x.lower() for x in txt]
            if(len(txt)==0):
                return [0]
            _txt = [word2id(x) for x in txt]
            return _txt

        def char_ids(txt, sent_max):
            txt = [x for x in txt if len(x)>0]
            _txt = [[self.char_index[y] for y in x] for x in txt]
            _txt = [pad_to_max(x, self.args.char_max) for x in _txt]
            _txt = pad_to_max(_txt, sent_max,
                    pad_token=self.char_pad_token)
            return _txt

        def sent2words(sent):
            sent = sent.rstrip('\n').split(' ')
            return [word2id(x) for x in sent]
            # return sent

        def entity2review(x, reviews):
            r = reviews[str(x)] # Get reviews
            return r

        def split_review(data):
            data = [[int(y) for y in x.split()] for x in data]
            return data

        # data = dict_to_list(data)

        # load data
        # user, item, rating, time, time_after, index
        # 0,    1,    2,      3,    4,          5
        user = [x[0] for x in data]
        items = [x[1] for x in data]
        labels = [x[2] for x in data]
        ts_deltas = [x[4] for x in data] # normalized timestamp delta - need to convert to float
        seq_idxs = [x[-1] for x in data] # index of each review within that user's review sequence

        # randomly sample pairs of (user_id1, user_id2)
        # make sure experience level increase from old to new
        user_seqs = defaultdict(list)
        for x in user:
            # print(len(user_seqs[x]))
            user_seqs[x].append(len(user_seqs[x]))

        # sample
        pair_seq_idxs1, pair_seq_idxs2 = [], []
        # exp_labels = []
        for x, x1 in zip(user, seq_idxs): # user x at x1 position of its review sequence
           x2 = x1
           while x2 == x1:
               x2 = random.sample(user_seqs[x], k=1)[0]
           #pair_seq_idxs1.append(x1)
           #pair_seq_idxs2.append(x2)
           if x1 > x2: # x1 is new 
               pair_seq_idxs1.append(x1)
               pair_seq_idxs2.append(x2)
               #exp_labels.append(1)
           else:     
               pair_seq_idxs1.append(x2)
               pair_seq_idxs2.append(x1)
               #exp_labels.append(0)

        # Raw user-item ids
        user_idx = user
        item_idx = items

        # (user, item) text
        user = [entity2review(x, self.user_repr) \
                            for x in tqdm(user, desc='user2rev')]
        items = [entity2review(x, self.item_repr) \
                            for x in tqdm(items, desc='item2rev')]

        raw_user = user
        raw_items = items

        user_len = [len(x) for x in user]
        item_len = [len(x) for x in items]

        if(self.args.base_encoder!='Flat'):
            print("Preparing [Hierarchical Mode]")
            # MPCN uses hierarchical inputs
            # tylib.exp.prep_hierarchical_data_list
            if self.args.base_dynamic: # only use reviews earlier than current interaction - danymically change for each interaction of an user        
                user, user_len = prep_hierarchical_data_list(user, # len(user) equals total iteractions
                                                    self.args.smax,
                                                    self.args.dmax, 
                                                    recency=self.args.recency, dynamic=True, end_idx=seq_idxs) # seq_idx - which position is that interaction along all interactions for that user
            else: # use dmax earliest reviews for all interactions of an user
                user, user_len = prep_hierarchical_data_list(user,
                                                    self.args.smax,
                                                    self.args.dmax, 
                                                    recency=self.args.recency)
            items, item_len = prep_hierarchical_data_list(items,
                                                self.args.smax,
                                                self.args.dmax,
                                                recency=self.args.recency)
        else:
            print("Preparing [Flat Mode]")
            # Flat mode are for DeepCoNN or D-ATT models
            print("Prepare user text...")
            if self.args.base_dynamic: 
                user, user_len = prep_flat_data_list(user,
                                                    self.args.smax,
                                                    self.args.dmax,
                                                    add_delimiter=2,
                                                    dynamic=True,
                                                    end_idx=seq_idxs
                                                    )

            else:
                user, user_len = prep_flat_data_list(user,
                                                    self.args.smax,
                                                    self.args.dmax,
                                                    add_delimiter=2
                                                    )
            print("Prepare item text...")
            items, item_len = prep_flat_data_list(items,
                                                self.args.smax,
                                                self.args.dmax,
                                                add_delimiter=2)
            # print(user_len)

        output = [user, user_len, items, item_len]
        self.mdl.register_index_map(0, 'q1_inputs')
        self.mdl.register_index_map(1, 'q1_len')
        self.mdl.register_index_map(2, 'q2_inputs')
        self.mdl.register_index_map(3, 'q2_len')

        if('TNET' in self.args.rnn_type):
            # TransNet specific review-loss
            user2 = [entity2review(x, self.user_repr2) \
                                for x in tqdm(user_idx,
                                desc='tnet_user2rev')]
            user2, user2len = prep_flat_data_list(user2, self.args.smax,
                                                2, add_delimiter=True)
            self.mdl.register_index_map(len(output), 'trans_inputs')
            output += [user2]
            self.mdl.register_index_map(len(output), 'trans_len')
            output += [user2len]

        if('EXP' in self.args.rnn_type): 
            # user_id and item_id
            self.mdl.register_index_map(len(output), 'user_idx')
            output += [user_idx]
            self.mdl.register_index_map(len(output), 'item_idx')
            output += [item_idx]
            # (pair_user1, pair_user2) according different seq_idx (i.e. time)
            print("Prepare pos user text...")
            pair_user1, pair_user_len1 = prep_flat_data_list(raw_user, self.args.smax,
                                                    self.args.dmax,
                                                    add_delimiter=2,
                                                    dynamic=True,
                                                    end_idx=pair_seq_idxs1
                                                    )
            print("Prepare neg user text...")
            pair_user2, pair_user_len2 = prep_flat_data_list(raw_user, self.args.smax,
                                                    self.args.dmax,
                                                    add_delimiter=2,
                                                    dynamic=True,
                                                    end_idx=pair_seq_idxs2
                                                    )
            self.mdl.register_index_map(len(output), 'pair_user_inputs1')
            output += [pair_user1]
            self.mdl.register_index_map(len(output), 'pair_user_len1')
            output += [pair_user_len1]
            self.mdl.register_index_map(len(output), 'pair_user_inputs2')
            output += [pair_user2]
            self.mdl.register_index_map(len(output), 'pair_user_len2')
            output += [pair_user_len2]
            #output.append(exp_labels)

        def normalize_labels(x, max_val, min_val):
            return (x - min_val) / (max_val - min_val)

        need_scaling = ['yelp17','netflixPrize']

        if(self.args.dataset in need_scaling and 'RAW' not in self.args.rnn_type):
            print("Scaling dataset..")
            labels = [normalize_labels(x, self.max_val, self.min_val) \
                            for x in labels]
        #print(labels)
        output.append(labels)
        output = zip(*output)
        print("=====================================")
        print('[Prep {}]'.format(len(output)))
        print("Max={} Min={}".format(self.max_val, self.min_val))
        print("=====================================")
        return output

    def _load_sets(self):
        # Load train, test and dev sets
        # fp = './datasets/fold{}/'.format(self.args.fold)
        self.train_set = self.env['train']
        self.dev_set = self.env['dev']
        if(self.args.dev==0):
            self.train_set += self.dev_set
        self.test_set = self.env['test']
        self.neg_set = self.env['negs']
        
        ######## MINE
        # make sure test only see reviews on train
        # for idx in range(len(self.test_set)):
            # self.test_set[idx][4] = self.test_set[idx][4] - 1 

        if('CHAR' in self.args.rnn_type):
            self.char_index = self.env['char_index']

        if(self.no_text_mode==False):
            self.word_index = self.env['word_index']
            self.index_word = {k:v for v, k in self.word_index.items()}
            self.vocab = len(self.word_index)
            print(self.env.keys())
            self.predict_dict = None
            self.test_predict_dict = defaultdict(int)
            print("vocab={}".format(self.vocab))
            '''
            if(self.args.features and 'word2dfs' in self.env):
                word2df = self.env['word2dfs']
                id2df = {}
                for key, value in word2df.items():
                    _id = self.word_index[key]
                    id2df[_id] = value
                self.word2df = id2df
                print("Loaded word2dfs")
            else:
                self.word2df = None 
            '''
        self.write_to_file("Train={} Dev={} Test={}".format(
                                len(self.train_set),
                                len(self.dev_set),
                                len(self.test_set)))

    def evaluate(self, data, bsz, epoch, name="", set_type=""):

        # acc = 0
        num_batches = int(len(data) / bsz)
        all_preds = []
        raw_preds = []
        ff_feats = []
        all_qout = []

        predict_op = self.mdl.predict_op
        actual_labels = [x[-1] for x in data]
        for i in tqdm(range(num_batches+1)):
            batch = batchify(data, i, bsz, max_sample=len(data))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
            loss, preds = self.sess.run([self.mdl.cost,
                            predict_op], feed_dict)
            try:
                all_preds += [x[0] for x in preds]
            except:
                all_preds += [x for x in preds]

        if('SIG_MSE' in self.args.rnn_type):
            """ Rescaling [0,1] is not supported
            """
            # print(all_preds)
            def rescale(x):
                return (x * (self.max_val - self.min_val)) + self.min_val
            all_preds = [rescale(x) for x in all_preds]
            actual_labels = [rescale(x) for x in actual_labels]

        ## MINE
        # _stat_al = [math.ceil(x) for x in actual_labels]
        # _stat_pred = [math.ceil(x) for x in all_preds]
        # print(Counter(_stat_pred))
        # print(Counter(_stat_al))

        def clip_labels(x):
            if(x>5):
                return 5
            elif(x<1):
                return 1
            else:
                return x

        all_preds = [clip_labels(x) for x in all_preds]
        # acc_preds = [round(x) for x in all_preds]
        # acc = accuracy_score(actual_labels, acc_preds)
        mse = mean_squared_error(actual_labels, all_preds)
        # actual_labels = [int(x) for x in actual_labels]
        # all_preds = [int(x) for x in all_preds]
        # f1 = f1_score(actual_labels, all_preds, average='macro')
        # mae = mean_absolute_error(actual_labels, all_preds)
        # record evaluation result
        self._register_eval_score(epoch, set_type, 'MSE', mse)
        # self._register_eval_score(epoch, set_type, 'MAE', mae)
        self._register_eval_score(epoch, set_type, 'RMSE', mse ** 0.5)
        # self._register_eval_score(epoch, set_type, 'ACC', acc)
        # self._register_eval_score(epoch, set_type, 'F1', f1)
        return mse, all_preds

    def eval_ranking(self):
        bsz = self.args.batch_size

        num_batches = int(len(self.neg_set) / bsz)
        all_preds = []
        raw_preds = []
        ff_feats = []
        all_qout = []

        predict_op = self.mdl.predict_op
        actual_labels = [x[-1] for x in self.neg_set]
        for i in tqdm(range(num_batches+1)):
            batch = batchify(self.neg_set, i, bsz, max_sample=len(self.neg_set))
            if(len(batch)==0):
                continue
            feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
            loss, preds = self.sess.run([self.mdl.cost,
                            predict_op], feed_dict)
            try:
                all_preds += [x[0] for x in preds]
            except:
                all_preds += [x for x in preds]

        hr = 0.0
        total = 0.0

        assert len(all_preds) % 6 == 0
        for i in range(0, len(all_preds), 6):
            consider = all_preds[i : i + 6]
            ranking = np.argsort(consider)

            if ranking[-1] == 0: hr += 1.0
            total += 1.0

        final = round(100.0 * hr / total, 2)

        return final

        # mse = mean_squared_error(actual_labels, all_preds)
        # actual_labels = [int(x) for x in actual_labels]
        # all_preds = [int(x) for x in all_preds]
        # f1 = f1_score(actual_labels, all_preds, average='macro')
        # mae = mean_absolute_error(actual_labels, all_preds)
        # record evaluation result
        # self._register_eval_score(epoch, set_type, 'MSE', mse)
        # self._register_eval_score(epoch, set_type, 'MAE', mae)
        # self._register_eval_score(epoch, set_type, 'RMSE', mse ** 0.5)
        # self._register_eval_score(epoch, set_type, 'ACC', acc)
        # self._register_eval_score(epoch, set_type, 'F1', f1)
        # return mse, all_preds

    def save_preds(self, all_preds, name):
        ret = "\n".join(map(str, all_preds))

        f = open("logs/" + self.args.dataset + "/" + name + "_preds.txt", "w")
        f.write(ret)
        f.close()

        return

    def train(self):
        """ Main training loop
        """
        print(self.args)
        scores = []
        best_score = -1
        best_dev = -1
        best_epoch = -1
        counter = 0
        epoch_scores = {}
        self.eval_list = []
        # process data
        data = self.prepare_set(self.train_set)
        self.test_set = self.prepare_set(self.test_set)
        self.dev_set = self.prepare_set(self.dev_set)
        self.neg_set = self.prepare_set(self.neg_set)

        print("Training Interactions={}".format(len(data)))
        self.sess.run(tf.assign(self.mdl.is_train,self.mdl.true))

        for epoch in range(1, self.args.epochs+1):

            all_att_dict = {}
            pos_val, neg_val = [],[]
            t0 = time.clock()
            self.write_to_file("=====================================")
            losses = []
            # random.shuffle(data)
            num_batches = int(len(data) / self.args.batch_size)
            norms = []
            # all_acc = 0

            train_preds = []

            for i in tqdm(range(0, num_batches+1)):
                batch = batchify(data, i, self.args.batch_size,
                                max_sample=len(data))

                if(len(batch)==0):
                    continue

                feed_dict = self.mdl.get_feed_dict(batch)
                train_op = self.mdl.train_op
                run_options = tf.RunOptions(timeout_in_ms=10000)

                _, loss  = self.sess.run([train_op,
                                        self.mdl.cost],
                                        feed_dict)

                if('TNET' in self.args.rnn_type):
                    # TransNet secondary review-loss
                    loss2 = self.sess.run([self.mdl.trans_loss], feed_dict)

                if(self.args.show_affinity==1):
                    afm = self.sess.run([self.mdl.afm], feed_dict)
                    show_afm(afm)

                # all_acc += (loss * len(batch))
                if(self.args.tensorboard):
                    self.train_writer.add_summary(summary, counter)
                counter +=1

                losses.append(loss)

                # To save preds later
                feed_dict = self.mdl.get_feed_dict(batch, mode='testing')
                _, preds = self.sess.run([self.mdl.cost, self.mdl.predict_op], feed_dict)
                try: train_preds += [x[0] for x in preds]
                except: train_preds += [x for x in preds]

            t1 = time.clock()
            # self.write_to_file("[{}] [Epoch {}] [{}] loss={} acc={}".format(
            #                     self.args.dataset, epoch, self.model_name,
            #                     np.mean(losses), all_acc / len(data)))
            self.write_to_file("[{}] [Epoch {}] [{}] loss={}".format(
                                self.args.dataset, epoch, self.model_name,
                                np.mean(losses)))
            self.write_to_file("GPU={} | | d={}".format(
                                            self.args.gpu,
                                            self.args.emb_size))

            if(epoch % self.args.eval==0):
                self.sess.run(tf.assign(self.mdl.is_train, self.mdl.false))
                
                _, dev_preds = self.evaluate(self.dev_set, self.args.batch_size, epoch, set_type='Dev')
                # self._show_metrics(epoch, self.eval_dev, self.show_metrics, name='Dev')
                # best_epoch1, cur_dev = self._select_test_by_dev(
                    # epoch,
                    # self.eval_dev,
                    # {},
                    # no_test=True,
                    # lower_is_better=True
                # )

                _, test_preds = self.evaluate(self.test_set, self.args.batch_size, epoch, set_type='Test')
                self._show_metrics(epoch, self.eval_test, self.show_metrics, name='Test')

                final = self.eval_ranking()
                self.write_to_file("HR@1={}\n".format(final))

                stop, max_e, best_epoch = self._select_test_by_dev(
                    epoch,
                    self.eval_dev,
                    self.eval_test,
                    lower_is_better=True
                )

                if epoch == best_epoch:
                    # Save output into files
                    self.save_preds(train_preds, "train")
                    self.save_preds(test_preds, "test")
                    self.save_preds(dev_preds, "val")

                if(epoch-best_epoch>self.args.early_stop and self.args.early_stop>0):
                    print("Ended at early stop")
                    return

if __name__ == '__main__':
    exp = CFExperiment(inject_params=None)
    exp.train()
    print("End of code!")
