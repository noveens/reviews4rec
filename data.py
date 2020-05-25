import os
import copy
import torch
import surprise
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable

from utils import *

class DataLoader():
    def __init__(
            self, hyper_params, data, user_reviews, item_reviews, negs,
            this_index_user_item = None, test_reviews = None,
            train_loader = None
        ):
        self.data = np.array(data)
        self.hyper_params = hyper_params
        self.user_reviews = user_reviews
        self.item_reviews = item_reviews
        self.this_index_user_item = this_index_user_item
        self.test_reviews = test_reviews
        self.negs = negs

        if train_loader is None:
            self.count_train_counts()
            self.calculate_reviewed_map()
        else:
            self.user_count, self.item_count = train_loader.user_count, train_loader.item_count
            self.u_to_i_map = train_loader.u_to_i_map
            self.i_to_u_map = train_loader.i_to_u_map

        if hyper_params['model_type'] in [ 'bias_only', 'MF', 'MF_dot', 'NeuMF' ]: self.iter = self.iter_simple
        else: self.iter = self.iter_review

    def calculate_reviewed_map(self):
        self.u_to_i_map = {}
        self.i_to_u_map = {}

        for u in self.user_reviews:
            self.u_to_i_map[u] = []
            for index in range(len(self.user_reviews[u])): self.u_to_i_map[u].append(0)

            if u not in self.this_index_user_item: continue

            for item in self.this_index_user_item[u]:
                index = self.this_index_user_item[u][item][0]
                self.u_to_i_map[u][index] = item

        inv_map = {}
        for item in self.item_reviews: inv_map[item] = {}
        for user in self.this_index_user_item:
            for item in self.this_index_user_item[user]:

                inv_map[item][user] = self.this_index_user_item[user][item]

        for i in self.item_reviews:
            self.i_to_u_map[i] = []
            for index in range(len(self.item_reviews[i])): self.i_to_u_map[i].append(0)

            for user in inv_map[i]:
                index = self.this_index_user_item[user][i][1]
                self.i_to_u_map[i][index] = user

    def __len__(self):
        num_b  = len(self.data) // int(self.hyper_params['batch_size'])
        num_b += int(len(self.data) % int(self.hyper_params['batch_size']) > 0)
        return num_b

    def count_train_counts(self):
        self.user_count, self.item_count = {}, {}

        for rating_tuple in self.data:
            if rating_tuple[0] not in self.user_count: self.user_count[rating_tuple[0]] = 0
            if rating_tuple[1] not in self.item_count: self.item_count[rating_tuple[1]] = 0

            self.user_count[rating_tuple[0]] += 1
            self.item_count[rating_tuple[1]] += 1

    def get_count_user(self, user):
        if user in self.user_count: return self.user_count[user]
        return 0

    def get_count_item(self, item):
        if item in self.item_count: return self.item_count[item]
        return 0

    def get_surprise_format_data(self):
        # write dataset to a temp file
        delim = ";"
        write_str = ""
        for rating_tuple in tqdm(self.data):
            write_str += str(int(rating_tuple[0])) + delim
            write_str += str(int(rating_tuple[1])) + delim
            write_str += str(rating_tuple[2]) + '\n'

        f = open("temp_file.txt", "w")
        f.write(write_str)
        f.close()

        rating_scale = (1, 5)
        if self.hyper_params['dataset'] == 'ratebeer': rating_scale = (1, 20)

        reader = surprise.Reader(line_format = "user item rating", sep = delim, rating_scale = rating_scale)
        dataset = surprise.Dataset.load_from_file(file_path = "temp_file.txt", reader = reader)
        trainset = dataset.build_full_trainset()

        # Delete temp file
        os.remove("temp_file.txt")

        return trainset

    # Used in surprise models' evaluation
    def iter_non_torch(self, eval = False, train_counts = False):
        user, item, y_ratings, counts_user, counts_item = [], [], [], [], []
        batch_num = 0

        for rating_tuple in tqdm(self.data):

            user.append(int(rating_tuple[0]))
            item.append(int(rating_tuple[1]))
            y_ratings.append(rating_tuple[2])
            counts_user.append(self.get_count_user(rating_tuple[0]))
            counts_item.append(self.get_count_item(rating_tuple[1]))

            if len(user) % int(self.hyper_params['batch_size']) == 0:
                if train_counts == False:
                    yield [ user, item ], y_ratings

                else:
                    yield [ 
                        user, item, 
                        counts_user, 
                        counts_item
                    ], y_ratings                    

                batch_num += 1
                user, item, y_ratings, counts_user, counts_item = [], [], [], [], []

        if len(user) > 0:
            yield [ 
                user, 
                item, 
            ], y_ratings

    def pad_only(self, reviews_all, negs):
        # Input shape: [ bsz x variable_num_reviews x variable_num_words ]

        num_reviews = self.hyper_params['narre_num_reviews']
        num_words = self.hyper_params['narre_num_words']

        flag = False
        if negs == False:
            reviews_all = [ reviews_all ]
            flag = True

        for reviews in reviews_all:
            for batch_num in range(len(reviews)):
                for review_num in range(len(reviews[batch_num])):
                    for _ in range(num_words - len(reviews[batch_num][review_num])):
                        reviews[batch_num][review_num].append(0)

                    reviews[batch_num][review_num] = reviews[batch_num][review_num][:num_words]

                for _ in range(num_reviews - len(reviews[batch_num])):
                    reviews[batch_num].append([ 0 for _ in range(num_words) ])

                reviews[batch_num] = reviews[batch_num][:num_reviews]

        if flag == True: reviews_all = reviews_all[0]

        return reviews_all

    def pad_and_join(self, reviews, negs = False):
        if self.hyper_params['model_type'] in [ 'NARRE' ]: return self.pad_only(reviews, negs)

        # Input shape: [ bsz x variable_num_reviews x variable_num_words ]
        ret = []

        if len(reviews) == 0: return reviews

        flag = False
        if negs == False:
            reviews = [ reviews ]
            flag = True

        for b1 in reviews:

            ret_temp = []

            for b in b1:
                concat = []

                for review in b:
                    concat += review

                # Pad this to fixed length
                while len(concat) < self.hyper_params['input_length']:
                    concat.append(0)

                # If greater
                concat = concat[:self.hyper_params['input_length']]

                ret_temp.append(concat)

            ret.append(ret_temp)

        if flag == True: ret = ret[0]

        return ret
        
    def remove_overlap(self, u_r, i_r, user_id, item_id):
        this_reviews = []

        users_who_gave, items_reviewed = [], []

        if self.this_index_user_item is not None:
            indices = self.this_index_user_item[user_id][item_id]

            u_r_new, i_r_new = [], []

            this_reviews.append(u_r[indices[0]])

            for i in range(len(u_r)): 
                if i != indices[0]: 
                    u_r_new.append(u_r[i])
                    items_reviewed.append(self.u_to_i_map[user_id][i])
                else:
                    assert self.u_to_i_map[user_id][i] == item_id

            for i in range(len(i_r)): 
                if i != indices[1]: 
                    i_r_new.append(i_r[i])
                    users_who_gave.append(self.i_to_u_map[item_id][i])
                else:
                    assert self.i_to_u_map[item_id][i] == user_id

            u_r, i_r = u_r_new, i_r_new

        else:
            items_reviewed = self.u_to_i_map[user_id]
            users_who_gave = self.i_to_u_map[item_id]

            if self.test_reviews is not None:
                this_reviews.append(self.test_reviews[user_id][item_id])
            else: this_reviews.append([0])

        return u_r, i_r, this_reviews, users_who_gave, items_reviewed

    # Used in pytorch models where review is used. Eg. DeepCoNN, NARRE etc.
    def iter_review(self, eval = False, simple = False):
        user, item, user_reviews, item_reviews, y_ratings, this_reviews = [], [], [], [], [], []
        users_who_gave, items_reviewed = [], []
        batch_num = 0

        for rating_tuple in tqdm(self.data):

            user.append(int(rating_tuple[0]))
            item.append(int(rating_tuple[1]))
            y_ratings.append(rating_tuple[2])

            u_r = self.user_reviews[int(rating_tuple[0])]
            i_r = self.item_reviews[int(rating_tuple[1])]
            u_r, i_r, this_reviews_, users_who_gave_, items_reviewed_ = self.remove_overlap(u_r, i_r, int(rating_tuple[0]), int(rating_tuple[1]))
            
            user_reviews.append(u_r)
            item_reviews.append(i_r)
            this_reviews.append(this_reviews_)
            users_who_gave.append(users_who_gave_)
            items_reviewed.append(items_reviewed_)

            if len(user) % int(self.hyper_params['batch_size']) == 0:
                
                for b in range(len(users_who_gave)):
                    while len(users_who_gave[b]) < 10: users_who_gave[b].append(self.hyper_params['total_users'] + 1)
                    while len(items_reviewed[b]) < 10: items_reviewed[b].append(self.hyper_params['total_items'] + 1)

                    users_who_gave[b] = users_who_gave[b][:10]
                    items_reviewed[b] = items_reviewed[b][:10]

                if simple == True:
                    yield [
                        self.pad_and_join(this_reviews), 
                        users_who_gave,
                        items_reviewed,
                        self.pad_and_join(user_reviews), 
                        self.pad_and_join(item_reviews),
                        user, 
                        item, 
                    ], y_ratings

                else:
                    yield [
                        Variable(LongTensor(self.pad_and_join(this_reviews))), 
                        Variable(LongTensor(users_who_gave)),
                        Variable(LongTensor(items_reviewed)),
                        Variable(LongTensor(self.pad_and_join(user_reviews))), 
                        Variable(LongTensor(self.pad_and_join(item_reviews))),  
                        Variable(LongTensor(user)), 
                        Variable(LongTensor(item)), 
                    ], Variable(FloatTensor(y_ratings))

                batch_num += 1
                user, item, user_reviews, item_reviews, this_reviews, y_ratings = [], [], [], [], [], []
                users_who_gave, items_reviewed = [], []

        if len(user) > 0:

            for b in range(len(users_who_gave)):
                while len(users_who_gave[b]) < 10: users_who_gave[b].append(self.hyper_params['total_users'] + 1)
                while len(items_reviewed[b]) < 10: items_reviewed[b].append(self.hyper_params['total_items'] + 1)

                users_who_gave[b] = users_who_gave[b][:10]
                items_reviewed[b] = items_reviewed[b][:10]

            if simple == True:
                yield [
                    self.pad_and_join(this_reviews), 
                    users_who_gave,
                    items_reviewed,
                    self.pad_and_join(user_reviews), 
                    self.pad_and_join(item_reviews),
                    user, 
                    item, 
                ], y_ratings

            else:
                yield [
                    Variable(LongTensor(self.pad_and_join(this_reviews))), 
                    Variable(LongTensor(users_who_gave)),
                    Variable(LongTensor(items_reviewed)),
                    Variable(LongTensor(self.pad_and_join(user_reviews))), 
                    Variable(LongTensor(self.pad_and_join(item_reviews))),  
                    Variable(LongTensor(user)), 
                    Variable(LongTensor(item)), 
                ], Variable(FloatTensor(y_ratings))

    # Used in pytorch models where review is not used. Eg. Bias Only, MF etc.
    def iter_simple(self, eval = False):
        user, item, user_reviews, item_reviews, y_ratings = [], [], [], [], []
        batch_num = 0

        for rating_tuple in tqdm(self.data):

            user.append(rating_tuple[0])
            item.append(rating_tuple[1])
            y_ratings.append(rating_tuple[2])

            if len(user) % int(self.hyper_params['batch_size']) == 0:
                yield [ 
                    None,
                    None,
                    None,
                    None,
                    None,
                    Variable(LongTensor(user)), 
                    Variable(LongTensor(item)), 
                ], Variable(FloatTensor(y_ratings))

                batch_num += 1
                user, item, user_reviews, item_reviews, y_ratings = [], [], [], [], []

        if len(user) > 0:
            yield [ 
                None,
                None,
                None,
                None,
                None,
                Variable(LongTensor(user)), 
                Variable(LongTensor(item)), 
            ], Variable(FloatTensor(y_ratings))

    # Used in HR@1 computation
    def iter_negs(self, review):
        user, item, y_ratings, user_reviews, item_reviews, this_reviews = [], [], [], [], [], []
        users_who_gave, items_reviewed = [], []

        for u in tqdm(self.negs):

            i = self.negs[u][0][0]

            temp_u, temp_i = [], []
            temp_user_reviews, temp_item_reviews, temp_this_reviews = [], [], []
            temp_users_who_gave, temp_items_reviewed = [], []

            for i2 in [i] + self.negs[u][1]:

                temp_u.append(u)
                temp_i.append(i2)

                if review == True:

                    u_r = self.user_reviews[u]
                    i_r = self.item_reviews[i2]

                    u_r, i_r, this_reviews_, users_who_gave_, items_reviewed_ = self.remove_overlap(u_r, i_r, u, i)
            
                    temp_user_reviews.append(u_r)
                    temp_item_reviews.append(i_r)
                    temp_this_reviews.append(this_reviews_)
                    temp_users_who_gave.append(users_who_gave_)
                    temp_items_reviewed.append(items_reviewed_)

            user.append(temp_u)
            item.append(temp_i)
            y_ratings.append(0.0) # Doesn't matter, only for ranking

            if review == True:
                for b in range(len(temp_users_who_gave)):
                    while len(temp_users_who_gave[b]) < 10: temp_users_who_gave[b].append(self.hyper_params['total_users'] + 1)
                    while len(temp_items_reviewed[b]) < 10: temp_items_reviewed[b].append(self.hyper_params['total_items'] + 1)

                    temp_users_who_gave[b] = temp_users_who_gave[b][:10]
                    temp_items_reviewed[b] = temp_items_reviewed[b][:10]

                user_reviews.append(temp_user_reviews)
                item_reviews.append(temp_item_reviews)
                this_reviews.append(temp_this_reviews)
                users_who_gave.append(temp_users_who_gave)
                items_reviewed.append(temp_items_reviewed)

            if len(user) % int(self.hyper_params['batch_size']) == 0:

                yield [ 
                    Variable(LongTensor(self.pad_and_join(this_reviews, negs = True))), 
                    Variable(LongTensor(users_who_gave)),
                    Variable(LongTensor(items_reviewed)),
                    Variable(LongTensor(self.pad_and_join(user_reviews, negs = True))), 
                    Variable(LongTensor(self.pad_and_join(item_reviews, negs = True))), 
                    Variable(LongTensor(user)), 
                    Variable(LongTensor(item)), 
                ], Variable(FloatTensor(y_ratings))

                user, item, y_ratings, user_reviews, item_reviews, this_reviews = [], [], [], [], [], []
                users_who_gave, items_reviewed = [], []

        if len(user) > 0:
            yield [ 
                Variable(LongTensor(self.pad_and_join(this_reviews, negs = True))), 
                Variable(LongTensor(users_who_gave)),
                Variable(LongTensor(items_reviewed)),
                Variable(LongTensor(self.pad_and_join(user_reviews, negs = True))), 
                Variable(LongTensor(self.pad_and_join(item_reviews, negs = True))),  
                Variable(LongTensor(user)), 
                Variable(LongTensor(item)), 
            ], Variable(FloatTensor(y_ratings))

def load_data(hyper_params, load_negs = True):
    print("Loading data...")

    train = load_obj(hyper_params['data_dir'] + 'train')
    test = load_obj(hyper_params['data_dir'] + 'test')
    val = load_obj(hyper_params['data_dir'] + 'val')

    user_reviews, item_reviews = None, None
    user_reviews = load_obj(hyper_params['data_dir'] + 'user_reviews')
    item_reviews = load_obj(hyper_params['data_dir'] + 'item_reviews')
    
    negs = None
    if load_negs:
        negs = load_obj(hyper_params['data_dir'] + 'negs')
    
    this_index_user_item = load_obj(hyper_params['data_dir'] + 'this_index_user_item')
    test_reviews = load_obj(hyper_params['data_dir'] + 'test_reviews')
    num_users, num_items, num_words = load_obj(hyper_params['data_dir'] + 'num_users_items')

    hyper_params['total_users'] = num_users
    hyper_params['total_items'] = num_items
    hyper_params['total_words'] = num_words

    train_loader = DataLoader(
        hyper_params, train, user_reviews, item_reviews, negs, 
        this_index_user_item = this_index_user_item
    )

    return train_loader, \
           DataLoader(hyper_params, test, user_reviews, item_reviews, negs, 
                      test_reviews = test_reviews, train_loader = train_loader), \
           DataLoader(hyper_params, val, user_reviews, item_reviews, negs, 
                      test_reviews = test_reviews, train_loader = train_loader), \
           hyper_params
