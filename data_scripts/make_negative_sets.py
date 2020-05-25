import numpy as np
import os, sys
import pickle 
from tqdm import tqdm

sys.path.insert(0, "../")
from data import load_data

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

dataset = sys.argv[1]

for k_core in map(int, sys.argv[2].split(",")):

        hyper_params = {
            'dataset': dataset,
            'k_core': k_core,
            'percent_reviews_to_keep': 100,

            'model_type': 'bias_only',
        }

        hyper_params['data_dir']  = sys.argv[4]
        hyper_params['data_dir'] += hyper_params['dataset'] + "/" 
        hyper_params['data_dir'] += str(hyper_params['k_core']) + "_core/"
        if hyper_params['percent_reviews_to_keep'] != 100: 
            hyper_params['data_dir'] += str(hyper_params['percent_reviews_to_keep']) + "_percent/"

        try:
            print("\n\nMAKING NEGATIVES FOR:", k_core, dataset)
            train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params, load_negs = False)

            num_negs = 5

            negs = {}
            total_negs = 0

            user_pos = {}
            user_neg = {}
            for rating_tuple in tqdm(test_reader.data):
                u, i = int(rating_tuple[0]), int(rating_tuple[1])

                if u not in user_pos: 
                    user_pos[u] = []
                    user_neg[u] = []

                if float(rating_tuple[2]) >= 4.9: user_pos[u].append(i)
                else: user_neg[u].append(i)

            for user in tqdm(user_pos):
                all_pos = user_pos[user]
                all_neg = user_neg[user]

                if len(all_pos) == 0 or len(list(set(all_neg))) < num_negs: continue

                pos = [ all_pos[np.random.randint(len(all_pos), size = 1)[0]] ]

                neg = set()
                while len(neg) < num_negs:
                    check = all_neg[np.random.randint(len(all_neg), size = 1)[0]]
                    neg.add(check)
                neg = list(neg)

                negs[user] = [ pos, neg ]

                total_negs += 1

            print(dataset, "; Total negative transactions:", total_negs)

            print("Started saving data")
            for percent_reviews_to_keep in map(int, sys.argv[3].split(",")):
                init_path = sys.argv[4] + hyper_params['dataset'] + '/' + str(hyper_params['k_core']) + '_core/'
                if percent_reviews_to_keep != 100: init_path += str(percent_reviews_to_keep) + '_percent/'
                save_obj(negs, init_path + 'negs')

        except Exception as e:
            print("ERROR MAKING NEGATIVES:", k_core, dataset)
            print(e)
            pass
