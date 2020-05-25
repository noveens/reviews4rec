from gensim.models import Word2Vec
from tqdm import tqdm
import pickle, json
import numpy as np
import gensim
import random
import copy
import sys
import re
import gc
import os

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_json(name):
    final = []
    with open(name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            final.append(json.loads(line))
    return final

def tokenize_all(data):
    for vote in tqdm(data):
        vote[3] = tokenize(vote[3])
    return data

def tokenize(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower().split()

def get_map_recursive(data, key, k_core):
    counts = {}

    for review in data:
        if review[key] not in counts: counts[review[key]] = 0
        counts[review[key]] += 1

    ret = {}
    now = 0

    for review in data:
        if review[key] not in ret and counts[review[key]] >= k_core:
            ret[review[key]] = now
            now += 1

    return ret

def get_map(data, key_user, key_item, k_core = 0):

    done = 0; prev_u = -1; prev_i = -1; map_temp_u = None; map_temp_i = None; data_new = data
    
    while done < 500:
        map_temp_u = get_map_recursive(data_new, key_user, k_core)
        map_temp_i = get_map_recursive(data_new, key_item, k_core)

        if prev_u == len(map_temp_u) and prev_i == len(map_temp_i): break

        prev_u = len(map_temp_u); prev_i = len(map_temp_i); done += 1

        # Update data
        data_new2 = []
        for review in data_new:
            if review[key_user] in map_temp_u and review[key_item] in map_temp_i:
                data_new2.append(review)
        data_new = data_new2

    return map_temp_u, map_temp_i

def get_word_map(data):
    counts = {}

    for vote in data:
        review = vote[3]

        for word in review:
            if word not in counts: counts[word] = 0
            counts[word] += 1

    total, min_needed = len(counts), 0
    if total > 0: min_needed = np.sort(list(counts.values()))[::-1][min(total - 1, 50000)]

    ret, word_num_to_word = {}, {}
    now = 1 # 0 is UNK

    for vote in data:
        review = vote[3]

        for word in review:
            if word in ret: continue

            if counts[word] >= min_needed:
                ret[word] = now
                word_num_to_word[now] = word
                now += 1
            else: ret[word] = 0

    print("Num words kept =", now)

    return ret, now
      
def load_beer(file):
    data = []

    f = open(file, "rb")
    lines = f.readlines()
    f.close()

    temp = {}

    for line in tqdm(lines):
        line = str(line.strip().decode('latin-1'))
        if len(line) == 0: 
            data.append(temp)
            temp = {}
            continue

        if line[:11] == "beer/beerId": temp['asin'] = line.split(":")[-1]
        elif line[:18] == "review/profileName": temp['reviewerID'] = line.split(":")[-1][1:]
        elif line[:14] == "review/overall": temp['overall'] = float(line.split(":")[-1].split("/")[0])
        elif line[:11] == "review/text": temp['reviewText'] = line.split(":")[-1][1:]

    return data

dataset = sys.argv[1]
orig_file = sys.argv[2]

for k_core in map(int, sys.argv[3].split(",")):
    print("Dataset:", dataset, ", cores:", k_core, "\n\n")

    print("Loading data..")
    if dataset in [ 'ratebeer' ]: all_data = load_beer(orig_file)
    else: all_data = load_obj_json(orig_file)

    print("User and item maps..")
    user_map, item_map = get_map(all_data, 'reviewerID', 'asin', k_core = k_core)

    print("Creating final data..")
    final_first = []

    for review in all_data:
        if review['reviewerID'] not in user_map: continue
        if review['asin'] not in item_map: continue

        final_first.append([
            user_map[review['reviewerID']],
            item_map[review['asin']],
            float(review['overall']),
            review['reviewText']
        ])

    del all_data
    gc.collect()

    # Split intro train/test/val sets
    np.random.shuffle(final_first)
    train_split = int(0.8 * len(final_first))

    print("Tokenizing text..")
    final_first = tokenize_all(final_first)

    for percent_to_keep in map(int, sys.argv[4].split(",")):

        # Create a copy of final_first
        final = copy.deepcopy(final_first)

        reviews_kept, total_reviews = 0.0, 0.0
        for i in range(len(final[:train_split])):
            if random.random() > float(percent_to_keep) / 100.0: final[i][3] = []
            else: reviews_kept += 1.0
            total_reviews += 1.0

        print("% reviews kept:", round(100.0 * reviews_kept / total_reviews, 2))

        print("Word map..")
        word_map, total_words = get_word_map(final[:train_split]) # Keep all words

        for vote_num in range(len(final)):
            for i in range(len(final[vote_num][3])):
                if final[vote_num][3][i] in word_map:
                    final[vote_num][3][i] = word_map[final[vote_num][3][i]]
                else:
                    final[vote_num][3][i] = 0

        # Train set
        train = final[:train_split]

        # Get user/item reviews on train set
        user_reviews, item_reviews, this_index_user_item, num_reviews = {}, {}, {}, 0
        for user in range(0, len(user_map)): user_reviews[user] = []
        for item in range(0, len(item_map)): item_reviews[item] = []

        all_reviews_train_word2vec = []

        for review in train:
            if review[0] not in this_index_user_item: this_index_user_item[review[0]] = {}
            this_index_user_item[review[0]][review[1]] = [ len(user_reviews[review[0]]), len(item_reviews[review[1]]) ]

            user_reviews[review[0]].append(review[3])
            item_reviews[review[1]].append(review[3])
            num_reviews += 1

            all_reviews_train_word2vec.append([ str(i) for i in review[3] ])

        # Strip reviews
        train = [ [ i[0], i[1], i[2] ] for i in train ]

        remaining = final[train_split:]
        split_point = int(0.5 * len(remaining))

        test_reviews = {}
        test, val = [], []
        for review in remaining[:split_point]: 
            if review[0] not in test_reviews: test_reviews[review[0]] = {}
            test_reviews[review[0]][review[1]] = review[3]

            test.append([ review[0], review[1], review[2] ])
        
        for review in remaining[split_point:]: 
            if review[0] not in test_reviews: test_reviews[review[0]] = {}
            test_reviews[review[0]][review[1]] = review[3]

            val.append([ review[0], review[1], review[2] ])

        # Train word2vec
        print("Training Word2Vec..")

        all_reviews_train_word2vec.append([ '0' ])
        model = Word2Vec(all_reviews_train_word2vec, min_count=1, size=64, workers=-1, window=1, sg=1, negative=64, iter=20).wv
        word2vec = [ np.random.uniform(0.0, 1.0, 64).tolist() ]
        for word_num in range(1, total_words):
            if str(word_num) in model.wv:
                word2vec.append(model.wv[str(word_num)])
            else:
                word2vec.append(np.random.uniform(0.0, 1.0, 64).tolist())
        del all_reviews_train_word2vec

        num_words = 0
        if len(list(word_map.values())) > 0: num_words = max(list(word_map.values()))

        print()
        print('STATISTICS ', '-' * 30)
        print("Num Words:", num_words)
        print("Num Users:", len(user_map))
        print("Num Items:", len(item_map))
        print("Num Reviews:", len(train) + len(test) + len(val))
        print("Num Train:", len(train))
        print("Num Test:", len(test))
        print("Num Val:", len(val))
        print('-' * 42)
        print()

        print("Saving data..")
        base_path = sys.argv[5] + dataset + '/'
        base_path += str(k_core) + '_core/'
        if percent_to_keep != 100: base_path += str(percent_to_keep) + '_percent/'

        os.makedirs(base_path, exist_ok = True)

        save_obj(train, base_path + 'train')
        save_obj(test, base_path + 'test')
        save_obj(val, base_path + 'val')
        save_obj([ len(user_map), len(item_map), num_words ], base_path + 'num_users_items')
        save_obj(user_reviews, base_path + 'user_reviews')
        save_obj(item_reviews, base_path + 'item_reviews')
        save_obj(test_reviews, base_path + 'test_reviews')
        save_obj(this_index_user_item, base_path + 'this_index_user_item')
        save_obj(word2vec, base_path + 'word2vec')

        # Save user count and item train counts
        user_count, item_count = {}, {}
        for rating_tuple in train:
            if rating_tuple[0] not in user_count: user_count[rating_tuple[0]] = 0
            if rating_tuple[1] not in item_count: item_count[rating_tuple[1]] = 0

            user_count[rating_tuple[0]] += 1
            item_count[rating_tuple[1]] += 1

        save_obj(user_count, base_path + 'user_count')
        save_obj(item_count, base_path + 'item_count')

        del train, test, val, user_reviews, item_reviews, final
        gc.collect()
