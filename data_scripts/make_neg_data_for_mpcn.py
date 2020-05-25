import sys
import gzip
import json
import pickle
from tqdm import tqdm

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def dictFromFileUnicode(path):
    with gzip.open(path, 'r') as f:
        return json.loads(f.read())

def dictToFile(dict, path):
    print("Writing to {}".format(path))
    with gzip.open(path, 'w') as f: f.write(json.dumps(dict).encode())

dataset = sys.argv[1]
for k_core in map(int, sys.argv[2].split(",")):
  for percent_reviews_to_keep in map(int, sys.argv[3].split(",")):
    print("Loading data...")
    init_path = sys.argv[4] + dataset + '/' + str(k_core) + '_core/'
    if percent_reviews_to_keep != 100: init_path += str(percent_reviews_to_keep) + '_percent/'

    env = dictFromFileUnicode(init_path + 'env.gz')

    # Negs stuff
    final = []
    negs = load_obj(init_path + 'negs')
    for user in tqdm(negs):
      for item in negs[user][0] + negs[user][1]:
        final.append([
          user,
          item,
          5.0,
          0.0,
          0.0,
          -1
        ])
    env['negs'] = final

    dictToFile(env, init_path + 'env.gz')
