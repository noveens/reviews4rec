import os
import sys
import h5py

sys.path.insert(0, "../")
from data import load_data

def func(hyper_params, reader, name):
    base_path = sys.argv[5] + hyper_params['dataset'] + '/'
    base_path += str(hyper_params['k_core']) + '_core/'
    if hyper_params['percent_reviews_to_keep'] != 100: base_path += str(hyper_params['percent_reviews_to_keep']) + '_percent/'

    os.makedirs(base_path, exist_ok = True)

    shape = None
    if hyper_params['model_type'] == "NARRE":
        shape = [ len(reader.data), hyper_params['narre_num_reviews'], hyper_params['narre_num_words'] ]
    else:
        shape = [ len(reader.data), hyper_params['input_length'] ] # DeepCoNN, TransNet

    with h5py.File(base_path + name + '.hdf5', 'w') as file:
        dset = {}
        dset['a'] = file.create_dataset("a", shape, dtype = 'i8', maxshape = shape, compression="gzip")

        dset['b'] = file.create_dataset("b", [ len(reader.data), 10 ], dtype = 'i8', maxshape = [ len(reader.data), 10 ], compression="gzip")
        dset['c'] = file.create_dataset("c", [ len(reader.data), 10 ], dtype = 'i8', maxshape = [ len(reader.data), 10 ], compression="gzip")
        
        dset['d'] = file.create_dataset("d", shape, dtype = 'i8', maxshape = shape, compression="gzip")
        dset['e'] = file.create_dataset("e", shape, dtype = 'i8', maxshape = shape, compression="gzip")
        dset['f'] = file.create_dataset("f", [ len(reader.data) ], dtype = 'i8', maxshape = [ len(reader.data) ], compression="gzip")
        dset['g'] = file.create_dataset("g", [ len(reader.data) ], dtype = 'i8', maxshape = [ len(reader.data) ], compression="gzip")
        dset['h'] = file.create_dataset("h", [ len(reader.data) ], dtype = 'f8', maxshape = [ len(reader.data) ], compression="gzip")

        at = 0
        for (this_reviews, users_who_gave, items_reviewed, user_reviews, item_reviews, user, item), rating in reader.iter_review(simple = True):
            dset['a'][at : at+hyper_params['batch_size']] = this_reviews
            dset['b'][at : at+hyper_params['batch_size']] = users_who_gave
            dset['c'][at : at+hyper_params['batch_size']] = items_reviewed
            dset['d'][at : at+hyper_params['batch_size']] = user_reviews
            dset['e'][at : at+hyper_params['batch_size']] = item_reviews
            dset['f'][at : at+hyper_params['batch_size']] = user
            dset['g'][at : at+hyper_params['batch_size']] = item
            dset['h'][at : at+hyper_params['batch_size']] = rating
            at += hyper_params['batch_size']

dataset = sys.argv[1]

for k_core in map(int, sys.argv[2].split(",")):
    for percent in map(int, sys.argv[3].split(",")):

        hyper_params = {
            'dataset': dataset,
            'k_core': k_core,
            'percent_reviews_to_keep': percent,
            'input_length': 1000,
            'model_type': sys.argv[4],

            'narre_num_reviews': 10,
            'narre_num_words': 100,
        }

        hyper_params['data_dir']  = sys.argv[6]
        hyper_params['data_dir'] += hyper_params['dataset'] + "/" 
        hyper_params['data_dir'] += str(hyper_params['k_core']) + "_core/"
        if hyper_params['percent_reviews_to_keep'] != 100: 
            hyper_params['data_dir'] += str(hyper_params['percent_reviews_to_keep']) + "_percent/"

        try:
            print("\nRUNNING FOR:", dataset, ";", k_core, "core ;", percent, "\% reviews") 

            train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)

            # Adjust `b` according to your system's available memory 
            # A lower `b` will use less memory but take more time to complete
            b = len(train_reader.data) // 5

            train_reader.hyper_params['batch_size'] = b
            test_reader.hyper_params['batch_size'] = b
            val_reader.hyper_params['batch_size'] = b

            print("Train set:")
            func(hyper_params, train_reader, 'train')
            print("Test set:")
            func(hyper_params, test_reader, 'test')
            print("Val set:")
            func(hyper_params, val_reader, 'val')

        except Exception as e:
            print(e)
            print("NOT FOUND:", dataset, k_core, percent) 
            pass
