import sys
import gzip
import json
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

dataset = sys.argv[1]
for k_core in map(int, sys.argv[2].split(",")):
  for percent_reviews_to_keep in map(int, sys.argv[3].split(",")):
    print("Loading data...")
    init_path = sys.argv[4] + dataset + '/' + str(k_core) + '_core/'
    if percent_reviews_to_keep != 100: init_path += str(percent_reviews_to_keep) + '_percent/'

    train = load_obj(init_path + 'train')
    test = load_obj(init_path + 'test')
    val = load_obj(init_path + 'val')

    def convert(data):
      for i in range(len(data)):
        this = data[i]

        data[i] = [
          this[0],
          this[1],
          this[2],
          0.0,
          0.0,
          -1
        ]

      return data

    train = convert(train)
    test = convert(test)
    val = convert(val)

    user_reviews = load_obj(init_path + 'user_reviews')
    item_reviews = load_obj(init_path + 'item_reviews')
    test_reviews = load_obj(init_path + 'test_reviews')
    num_users, num_items, num_words = load_obj(init_path + 'num_users_items')

    for user in user_reviews:
      user_reviews[user] = [ ' '.join(list(map(str, review))) for review in user_reviews[user] ]

    for item in item_reviews:
      item_reviews[item] = [ ' '.join(list(map(str, review))) for review in item_reviews[item] ]

    user_text2 = {}
    for user in test_reviews:
      user_text2[user] = {}

      for item in test_reviews[user]:
        user_text2[user][item] = ' '.join(list(map(str, test_reviews[user][item])))

    word_index = {}
    for i in range(num_words):
      word_index[str(i)] = i

    env = {
      'train':          train,
      'dev':            val,
      'test':           test,

      'user_text':      user_reviews,
      'item_text':      item_reviews,
      'user_text2':     user_text2,
      'word_index':     word_index,

      'num_users': num_users,
      'num_items': num_items,

      'user_negative':  [],
    }

    def dictToFile(dict, path):
        print("Writing to {}".format(path))
        with gzip.open(path, 'w') as f: f.write(json.dumps(dict).encode())

    dictToFile(env, init_path + 'env.gz')
