import json
import torch
import pickle

is_cuda_available = torch.cuda.is_available()

if is_cuda_available: 
    print("Using CUDA...\n")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor
else:
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor
    
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 2)

def save_obj_json(obj, name):
    with open(name + '.json', 'w') as f:
        json.dump(obj, f)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_obj_json(name):
    with open(name + '.json', 'r') as f:
        return json.load(f)

def load_user_item_counts(hyper_params):
    user_count = load_obj(hyper_params['data_dir'] + 'user_count')
    item_count = load_obj(hyper_params['data_dir'] + 'item_count')
    return user_count, item_count

def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def clear_log_file(log_file):
    f = open(log_file, 'w')
    f.write('')
    f.close()

def pretty_print(h):
    print("{")
    for key in h:
        print(' ' * 4 + str(key) + ': ' + h[key])
    print('}\n')

def log_end_epoch(hyper_params, metrics, epoch, time_elpased, metrics_on = '(VAL)'):
    string2 = ""
    for m in metrics: string2 += " | " + m + ' = ' + str(metrics[m])
    string2 += ' ' + metrics_on

    ss  = '-' * 89
    ss += '\n| end of epoch {} | time: {:5.2f}s'.format(epoch, time_elpased)
    ss += string2
    ss += '\n'
    ss += '-' * 89
    file_write(hyper_params['log_file'], ss)

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

def init_transnet_optim(hyper_params, model):
    optimizer_source = torch.optim.Adam(
        model.source.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
    )

    if hyper_params['model_type'] == 'transnet++':
        optimizer_source_fm = torch.optim.Adam(
            list(model.source_fm.parameters()) + [ model.user_embedding.weight, model.item_embedding.weight ], 
            lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
        )
    else:
        optimizer_source_fm = torch.optim.Adam(
            model.source_fm.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
        )

    optimizer_target = torch.optim.Adam(
        model.target.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
    )
    optimizer_all = torch.optim.Adam(
        model.parameters(), lr=hyper_params['lr'], weight_decay=hyper_params['weight_decay']
    )

    return [ optimizer_source, optimizer_source_fm, optimizer_target, optimizer_all ]