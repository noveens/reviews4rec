import os
import time
import datetime as dt
from tqdm import tqdm

INF = 10000.0

def train(model, criterion, optimizer, reader, hyper_params):
    import torch

    model.train()
    
    # Initializing metrics since we will calculate MSE on the train set on the fly
    metrics = {}
    metrics['MSE'] = 0.0
    if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]: 
        metrics['MSE_target'], metrics['MSE_transform'] = 0.0, 0.0
    
    # Initializations
    total_x, total_batches = 0.0, 0.0
    
    # Train for one epoch, batch-by-batch
    for data, y in reader.iter():
        
        # Empty the gradients
        model.zero_grad()
        if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]: 
            for o in optimizer: o.zero_grad()
        else: optimizer.zero_grad()
    
        # Forward pass
        all_output = model(data)

        # Backward pass
        if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
            optimizer_source, optimizer_source_fm, optimizer_target, optimizer_all = optimizer

            loss_target = criterion(all_output[1], y)
            loss_target.backward(retain_graph = True)
            optimizer_target.step()

            loss_transform = all_output[2]
            loss_transform.backward(retain_graph = True)
            optimizer_source.step()

            loss_source = criterion(all_output[0], y, return_mean = False)
            metrics['MSE'] += float(torch.sum(loss_source.data))
            loss_source = torch.mean(loss_source)
            loss_source.backward()
            optimizer_source_fm.step()

            metrics['MSE_target'] += float(loss_target.data)
            metrics['MSE_transform'] += float(loss_transform.data)
        
        else:
            loss = criterion(all_output, y, return_mean = False)
            metrics['MSE'] += float(torch.sum(loss.data))
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

        try: total_x += float(int(all_output.shape[0])) # For every model
        except: total_x += float(int(all_output[0].shape[0])) # For TransNet
        total_batches += 1

    metrics['MSE'] = round(metrics['MSE'] / float(total_x), 4)
    if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
        metrics['MSE_target'] = round(metrics['MSE_target'] / float(total_batches), 4)
        metrics['MSE_transform'] = round(metrics['MSE_transform'] / float(total_batches), 4)

    return metrics

def train_complete(hyper_params, Model, train_reader, val_reader, user_count, item_count, model, review = True):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

    from loss import MSELoss
    from eval import evaluate, eval_ranking
    from utils import file_write, is_cuda_available, load_obj, log_end_epoch, init_transnet_optim

    file_write(hyper_params['log_file'], "\n\nSimulation run on: " + str(dt.datetime.now()) + "\n\n")
    file_write(hyper_params['log_file'], "Data reading complete!")
    file_write(hyper_params['log_file'], "Number of train batches: {:4d}".format(len(train_reader)))
    file_write(hyper_params['log_file'], "Number of validation batches: {:4d}".format(len(val_reader)))

    criterion = MSELoss(hyper_params)

    if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
        optimizer = init_transnet_optim(hyper_params, model)

    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
        )

    file_write(hyper_params['log_file'], str(model))
    file_write(hyper_params['log_file'], "\nModel Built!\nStarting Training...\n")

    try:
        best_MSE = float(INF)

        for epoch in range(1, hyper_params['epochs'] + 1):
            epoch_start_time = time.time()
            
            # Training for one epoch
            metrics = train(
                model, criterion, optimizer, train_reader, hyper_params
            )
            metrics['dataset'] = hyper_params['dataset']
            # log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on = '(TRAIN)')

            # Calulating the metrics on the validation set
            metrics, _, _ = evaluate(
                model, criterion, val_reader, hyper_params, 
                user_count, item_count, review = review
            )
            metrics['dataset'] = hyper_params['dataset']
            log_end_epoch(hyper_params, metrics, epoch, time.time() - epoch_start_time, metrics_on = '(VAL)')
            
            # Save best model on validation set
            if metrics['MSE'] < best_MSE:
                print("Saving model...")
                torch.save(model.state_dict(), hyper_params['model_path'])
                best_MSE = metrics['MSE']
            
    except KeyboardInterrupt: print('Exiting from training early')

    # Load best model and return it for evaluation on test-set
    model = Model(hyper_params)
    if is_cuda_available: model = model.cuda()
    model.load_state_dict(torch.load(hyper_params['model_path']))
    model.eval()

    return model

def main_MPCN(hyper_params, gpu_id = None):
    from utils import log_end_epoch, load_obj, load_user_item_counts

    # Try getting GPU ID to train MPCN on
    if gpu_id is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ: 
            gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            gpu_id = 0

    # Run MPCN (needs a python 2 environment)
    start_time = time.time()
    command  = "bash run_MPCN_in_p2.sh " + hyper_params['data_dir'] 
    command += " " + str(gpu_id) + " " + str(hyper_params['latent_size'])
    os.system(command)

    # This is where MPCN training and evaluation log file would be
    log_path  = "logs/" + hyper_params['data_dir']
    log_path += "RAW_MSE_MPCN_FN_FM/log/logs.txt"
    
    # Reading the log file to extract best MSE and HR@1
    f = open(log_path, 'r')
    lines = f.readlines()
    f.close()

    best_mse, hr = float(INF), None
    for line_num, line in enumerate(lines): 
        if line[:10] == "[Test] MSE": 
            mse = float(line.strip().split("=")[-1])
            if mse < best_mse:
                best_mse = mse
                hr = float(lines[line_num + 1][5:].strip())

    metrics = { 
        'MSE': round(best_mse, 4),
        'HR@1': hr,
        'dataset': hyper_params['dataset'],
    }

    log_end_epoch(hyper_params, metrics, 'FINAL', (time.time() - start_time))

    # Loading test data and user, item counts
    test_data = load_obj(hyper_params['data_dir'] + 'test')
    user_count, item_count = load_user_item_counts(hyper_params)

    # Load saved test predictions
    f = open("logs/" + hyper_params['data_dir'] + "test_preds.txt", "r")
    lines = f.readlines()
    f.close()
    test_results = [ float(i.strip()) for i in lines ]
    assert len(test_results) == len(test_data)

    # Getting the `user_count_mse_map` and `item_count_mse_map`
    user_count_mse_map, item_count_mse_map = {}, {}
    for i in range(len(test_data)):

        user, item = int(test_data[i][0]), int(test_data[i][1])
        pred, rating = float(test_results[i]), float(test_data[i][2])
        
        user_c, item_c = 0, 0
        if user in user_count: user_c = user_count[user]
        if item in item_count: item_c = item_count[item]

        mse = (rating - pred) ** 2

        if user_c not in user_count_mse_map: user_count_mse_map[user_c] = []
        if item_c not in item_count_mse_map: item_count_mse_map[item_c] = []

        user_count_mse_map[user_c].append(mse)
        item_count_mse_map[item_c].append(mse)

    return metrics, user_count_mse_map, item_count_mse_map

def main_HFT(hyper_params, gpu_id = None):
    from utils import log_end_epoch

    start_time = time.time()

    # Compile the HFT Model
    prev = ""
    if 'LD_LIBRARY_PATH' in os.environ: prev = os.environ['LD_LIBRARY_PATH']
    os.environ['LD_LIBRARY_PATH'] = prev + ':HFT/liblbfgs-1.10/lib/.libs/'
    os.chdir("HFT/")
    os.system("make")
    os.chdir("../")
            
    run_command = "HFT/train data/" + hyper_params['dataset'] + '/' + str(hyper_params['k_core']) + "_core/"
    if hyper_params['percent_reviews_to_keep'] != 100: 
        run_command += str(hyper_params['percent_reviews_to_keep']) + '_percent/'
    run_command += "hft_all.txt"

    # latent_reg, lambda, K, model-path, prediction-path
    run_command += ' ' + str(hyper_params['latent_reg']) + ' ' 
    run_command += str(hyper_params['lamda']) + ' ' 
    run_command += str(hyper_params['latent_size']) + ' a b'

    if os.system(run_command): print("Exiting...")
    
    metrics = {}
    metrics['dataset'] = hyper_params['dataset']
    f = open("saved_metrics.txt", "r")
    lines = f.readlines()
    f.close()
    metrics['HR@1'] = float(lines[-1].strip())
    metrics['MSE'] = float(lines[-2].strip())

    log_end_epoch(hyper_params, metrics, 'final', (time.time() - start_time))

    # Making `user_count_mse_map` and `item_count_mse_map`
    user_count_mse_map, item_count_mse_map = {}, {}

    f = open("user_count_mse_map.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip().split()
        user_count_mse_map[int(line[0])] = []
        for err in line[1:]:
            user_count_mse_map[int(line[0])].append(float(err))

    f = open("item_count_mse_map.txt", "r")
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.strip().split()
        item_count_mse_map[int(line[0])] = []
        for err in line[1:]:
            item_count_mse_map[int(line[0])].append(float(err))

    return metrics, user_count_mse_map, item_count_mse_map

def main_surprise(hyper_params, gpu_id = None):
    from data import load_data
    from utils import load_user_item_counts, log_end_epoch
    from surprise_models import Model

    # User and item (train-set) counts for making `user_count_mse_map`, `item_count_mse_map`
    user_count, item_count = load_user_item_counts(hyper_params)

    train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    rating_matrix = train_reader.get_surprise_format_data()
    model = Model(hyper_params, user_count, item_count)

    start_time = time.time()
    metrics, user_count_mse_map, item_count_mse_map = model(
        rating_matrix, test_reader
    )
    log_end_epoch(hyper_params, metrics, 'final', (time.time() - start_time))

    return metrics, user_count_mse_map, item_count_mse_map

def main_NeuMF(hyper_params, gpu_id = None):
    from pytorch_models.NeuMF import GMF, MLP, NeuMF
    from data import load_data
    from eval import evaluate, eval_ranking
    from utils import load_user_item_counts, is_cuda_available
    from utils import xavier_init, log_end_epoch
    from loss import MSELoss
    import torch

    user_count, item_count = load_user_item_counts(hyper_params)
    train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    start_time = time.time()

    initial_path = hyper_params['model_path']

    # Pre-Training the GMF Model
    hyper_params['model_path'] = initial_path + "_gmf"
    gmf_model = GMF(hyper_params)
    if is_cuda_available: gmf_model = gmf_model.cuda()
    xavier_init(gmf_model)
    gmf_model = train_complete(
        hyper_params, GMF, train_reader, val_reader, user_count, item_count, gmf_model
    )

    # Pre-Training the MLP Model
    hyper_params['model_path'] = initial_path + "_mlp"
    mlp_model = MLP(hyper_params)
    if is_cuda_available: mlp_model = mlp_model.cuda()
    xavier_init(mlp_model)
    mlp_model = train_complete(
        hyper_params, MLP, train_reader, val_reader, user_count, item_count, mlp_model
    )

    # Training the final NeuMF Model
    hyper_params['model_path'] = initial_path
    model = NeuMF(hyper_params)
    if is_cuda_available: model = model.cuda()
    model.init(gmf_model, mlp_model)
    model = train_complete(hyper_params, NeuMF, train_reader, val_reader, user_count, item_count, model)

    # Evaluating the final model for MSE on test-set
    criterion = MSELoss(hyper_params)
    metrics, user_count_mse_map, item_count_mse_map = evaluate(
        model, criterion, test_reader, hyper_params, user_count, item_count, review = False
    )

    # Evaluating the final model for HR@1 on test-set
    metrics.update(eval_ranking(model, test_reader, hyper_params, review = False))

    log_end_epoch(hyper_params, metrics, 'final', (time.time() - start_time), metrics_on = '(TEST)')
    
    return metrics, user_count_mse_map, item_count_mse_map

def main_pytorch(hyper_params, gpu_id = None):
    from data import load_data
    from eval import evaluate, eval_ranking
    from utils import load_obj, is_cuda_available
    from utils import load_user_item_counts, xavier_init, log_end_epoch
    from loss import MSELoss

    if hyper_params['model_type'] in [ 'deepconn', 'deepconn++' ]: from pytorch_models.DeepCoNN import DeepCoNN as Model
    elif hyper_params['model_type'] in [ 'transnet', 'transnet++' ]: from pytorch_models.TransNet import TransNet as Model
    elif hyper_params['model_type'] in [ 'NARRE' ]: from pytorch_models.NARRE import NARRE as Model
    elif hyper_params['model_type'] in [ 'bias_only', 'MF', 'MF_dot' ]: from pytorch_models.MF import MF as Model

    import torch

    # Load the data readers
    user_count, item_count = load_user_item_counts(hyper_params)
    if hyper_params['model_type'] not in [ 'bias_only', 'MF', 'MF_dot', 'NeuMF' ]:
        review_based_model = True
        try:
            from data_fast import load_data_fast
            train_reader, test_reader, val_reader, hyper_params = load_data_fast(hyper_params)
            print("Loaded preprocessed epoch files. Should be faster training...")
        except Exception as e:
            print("Tried loading preprocessed epoch files, but failed.")
            print("Please consider running `prep_all_data.sh` to make quick data for DeepCoNN/TransNet/NARRE.")
            print("This will save large amounts of run time.")
            print("Loading standard (slower) data..")
            train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)
    else:
        review_based_model = False
        train_reader, test_reader, val_reader, hyper_params = load_data(hyper_params)

    # Initialize the model
    model = Model(hyper_params)
    if is_cuda_available: model = model.cuda()
    xavier_init(model)
    
    # Train the model
    start_time = time.time()
    model = train_complete(
        hyper_params, Model, train_reader, 
        val_reader, user_count, item_count, model, review = review_based_model
    )

    # Calculating MSE on test-set
    criterion = MSELoss(hyper_params)
    metrics, user_count_mse_map, item_count_mse_map = evaluate(
        model, criterion, test_reader, hyper_params, 
        user_count, item_count, review = review_based_model
    )

    # Calculating HR@1 on test-set
    _, test_reader2, _, _ = load_data(hyper_params) # Needs default slow reader
    metrics.update(eval_ranking(model, test_reader2, hyper_params, review = review_based_model))

    log_end_epoch(hyper_params, metrics, 'final', time.time() - start_time, metrics_on = '(TEST)')
    
    return metrics, user_count_mse_map, item_count_mse_map

def main(hyper_params, gpu_id = None): 
    import os
    import torch
    import subprocess

    # Setting GPU ID for running entire code ## Very Very Imp.
    if gpu_id is not None: torch.cuda.set_device(int(gpu_id))

    if hyper_params['model_type'] in [ 'SVD', 'kNN', 'NMF', 'SVD++', 'baseline' ]: method = main_surprise
    elif hyper_params['model_type'] == 'HFT': method = main_HFT
    elif hyper_params['model_type'] == 'MPCN': method = main_MPCN
    elif hyper_params['model_type'] == 'NeuMF': method = main_NeuMF
    else: method = main_pytorch

    metrics, user_count_mse_map, item_count_mse_map = method(hyper_params, gpu_id = gpu_id)

    '''
    NOTE: In addition to metrics, we also provide the following for research purposes:
    - `user_count_mse_map`: 
        Python dict with key of type <int> and values as <list>
        where, 
            - Key: Test user's train-set frequency
            - Value: list containing MSE's for all test users with same train-frequency
    - `item_count_mse_map`: 
        Python dict with key of type <int> and values as <list>
        where, 
            - Key: Test item's train-set frequency
            - Value: list containing MSE's for all test items with same train-frequency
    '''

    return metrics

if __name__ == '__main__':
    from hyper_params import hyper_params
    main(hyper_params)
