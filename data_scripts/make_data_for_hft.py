import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, "../")
from utils import load_obj

def get_string(data):
    ret = ""

    for review in tqdm(data):
        ret += str(int(review[0])) + " "
        ret += str(int(review[1])) + " "
        ret += str(float(review[2])) + " "
        ret += "0 " # Time
        ret += str(int(review[3])) + " " # In which train/test/val set
        ret += str(len(review[4])) + " "
        for word in review[4]: ret += str(word) + " "
        ret += '\n'

    return ret

dataset = sys.argv[1]

for k_core in map(int, sys.argv[2].split(",")):
    for percent_reviews in map(int, sys.argv[3].split(",")):

        print("\n\nMAKING HFT DATA FOR:", k_core, percent_reviews)

        base_path = sys.argv[4] + dataset + '/'
        base_path += str(k_core) + '_core/'
        if percent_reviews != 100: base_path += str(percent_reviews) + '_percent/'

        # Load data
        train = load_obj(base_path + 'train')
        test = load_obj(base_path + 'test')
        val = load_obj(base_path + 'val')
        user_reviews = load_obj(base_path + 'user_reviews')
        test_reviews = load_obj(base_path + 'test_reviews')
        this_index_user_item = load_obj(base_path + 'this_index_user_item')
        negs = load_obj(base_path + "negs")

        final = []
        for vote in train:
            temp = [ vote[0], vote[1], vote[2], 0 ] #

            this_index_user = this_index_user_item[vote[0]][vote[1]][0]
            review = user_reviews[vote[0]][this_index_user]

            temp.append(review)
            final.append(temp)

        num_print = 0
        for vote in test:
            review = []
            temp = [ vote[0], vote[1], vote[2], 1, review ]
            final.append(temp)
            num_print += 1

        for vote in val:
            review = []
            temp = [ vote[0], vote[1], vote[2], 2, review ]
            final.append(temp)

        for user in negs:
            for item in negs[user][0] + negs[user][1]:
                review = []
                temp = [ user, item, 5.0, 3, review ]
                final.append(temp)

        print("Saving all data..")
        base_path = sys.argv[4] + dataset + '/' + str(k_core) + '_core/'
        if percent_reviews != 100: base_path += str(percent_reviews) + '_percent/'

        all_str = get_string(final)

        f = open(base_path + "hft_all.txt", "w")
        f.write(all_str)
        f.close()
