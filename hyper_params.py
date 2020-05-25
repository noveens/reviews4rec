import os

def get_common_path(hyper_params):
    method, fm = hyper_params['model_type'], True
    if method == 'deepconn++': method, fm = 'deepconn', False

    common_path  = str(method)
    common_path += '_' + str(hyper_params['dataset'])
    common_path += '_' + str(hyper_params['k_core']) + '_core_'

    if hyper_params['model_type'] in [ 'MF', 'MF_dot', 'NeuMF' ]:
        common_path += '_latent_size_' + str(hyper_params['latent_size'])

    elif hyper_params['model_type'] == 'HFT': 
        common_path += '_latent_size_' + str(hyper_params['latent_size'])
        common_path += '_percent_reviews_' + str(hyper_params['percent_reviews_to_keep'])

    elif hyper_params['model_type'] in [ 'deepconn', 'deepconn++' ]:
        common_path += '_word_embed_size_' + str(hyper_params['word_embed_size'])
        common_path += '_latent_size_' + str(hyper_params['latent_size'])
        common_path += '_percent_reviews_' + str(hyper_params['percent_reviews_to_keep'])
        common_path += '_fm_' + str(fm)

    elif hyper_params['model_type'] == 'NARRE':
        common_path += '_num_reviews_' + str(hyper_params['narre_num_reviews'])
        common_path += '_num_words_' + str(hyper_params['narre_num_words'])
        common_path += '_word_embed_size_' + str(hyper_params['word_embed_size'])
        common_path += '_latent_size_' + str(hyper_params['latent_size'])
        common_path += '_only_reviews_' + str(hyper_params['only_reviews'])
        common_path += '_percent_reviews_' + str(hyper_params['percent_reviews_to_keep'])

    elif hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
        common_path += '_word_embed_size_' + str(hyper_params['word_embed_size'])
        common_path += '_latent_size_' + str(hyper_params['latent_size'])
        common_path += '_percent_reviews_' + str(hyper_params['percent_reviews_to_keep'])
        common_path += '_fm_' + str(fm)

    elif hyper_params['model_type'] == 'MPCN':
        common_path += '_latent_size_' + str(hyper_params['latent_size'])
        common_path += '_percent_reviews_' + str(hyper_params['percent_reviews_to_keep'])
        return common_path

    common_path += '_wd_' + str(hyper_params['weight_decay'])
    common_path += '_lr_' + str(hyper_params['lr'])
    common_path += '_dropout_' + str(hyper_params['dropout'])
    common_path += '_input_len_' + str(hyper_params['input_length'])

    return common_path

hyper_params = {
    'dataset': 'InstantVideo', # Which dataset to run? 
    # Pass the same human-friendly dataset name as passed in `prep_all_data.sh`

    'k_core': 5, # Data setting?
    'percent_reviews_to_keep': 100, # How many percent of total-reviews to keep?

    'weight_decay': float(1e-6), # WD for pytorch models
    'lr': 0.002, # LR for ADAM
    'epochs': 2, # Epochs to train
    'batch_size': 128, # Batch size
    'shuffle_data_every_epoch': False, # Shuffle train-data every epoch?

    'latent_size': 10, # Latent size in all algos
    'word_embed_size': 64, # Word embedding size
    'input_length': 1000, # Length of user/item review document
    'dropout': 0.6, # 0.3/4 works good for 0-core, 0.6/8 for 5-core

    'model_type': 'bias_only', 
    #### Options: 
    # Non-textual: [ 'bias_only', 'MF', 'MF_dot', 'NeuMF' ]
    # Non-textual: [ 'SVD', 'kNN', 'NMF', 'SVD++', 'baseline' ] (From surprise library)
    # Reviews-as-reg: [ 'HFT' ]
    # Reviews-as-feat: [ 'deepconn', 'deepconn++', 'NARRE', transnet', 'transnet++', 'MPCN' ]
    
    'lamda': 0.1, # HFT Lamda (from text)
    'latent_reg': 0.0, # HFT MF Regularizer

    'narre_num_reviews': 10,
    'narre_num_words': 100,
}

common_path = get_common_path(hyper_params)
hyper_params['common_path'] = common_path
hyper_params['log_file'] = 'saved_logs/' + common_path
hyper_params['model_path'] = 'saved_models/' + common_path

os.makedirs('saved_logs/', exist_ok = True)
os.makedirs('saved_models/', exist_ok = True)

# Setting the data dir
hyper_params['data_dir']  = "data/"
hyper_params['data_dir'] += hyper_params['dataset'] + "/" 
hyper_params['data_dir'] += str(hyper_params['k_core']) + "_core/"
if hyper_params['percent_reviews_to_keep'] != 100: 
    hyper_params['data_dir'] += str(hyper_params['percent_reviews_to_keep']) + "_percent/"