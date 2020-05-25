import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import surprise
import copy
import time

from utils import *
import h5py
import gc
import os

class DataLoader():
    def __init__(self, hyper_params, file_name):
        self.hyper_params = hyper_params
        self.bsz = int(hyper_params['batch_size'])
        self.file_name = file_name
        self.init_path = "/".join(hyper_params['data_dir'].split("/")[1:])

        # Modify these parameters according to your system's memory (RAM)
        # Higher `num_times_to_load` will result in fast training but higher RAM
        # If `num_times_to_load` > 1, then `num_memoise` + 1 partitions will be in RAM
        self.num_times_to_load = 1
        self.num_memoise = 0
        
        self.memoised_data = []

        if hyper_params['model_type'] in [ 'NARRE' ]: self.root_path = "quick_data_narre/"
        else: self.root_path = "quick_data_deepconn/"

        with h5py.File(self.root_path + self.init_path + self.file_name, 'r') as f:
            self.total = len(f['a'])

            self.a = None
            if self.num_times_to_load == 1:
                self.a = f['a'][:]
                self.b = f['b'][:]
                self.c = f['c'][:]
                self.d = f['d'][:]
                self.e = f['e'][:]
                self.f2 = f['f'][:]
                self.g = f['g'][:]
                self.h = f['h'][:]

    def __len__(self):
        return int(self.total // self.bsz) + int(self.total % self.bsz > 0)

    def iter(self, eval = False, torch = True):
        num_entries_to_load = self.total // self.num_times_to_load

        start = []
        end = []
        s = 0

        for i in range(self.num_times_to_load):
            start.append(s)
            end.append(s + num_entries_to_load)
            s = end[-1] + 1
        end[-1] = self.total

        for i in range(self.num_times_to_load):

            delete_loaded_data = True

            if self.a is not None: 
                a, b, c, d, e, f2, g, h = self.a, self.b, self.c, self.d, self.e, self.f2, self.g, self.h
                delete_loaded_data = False
            else:
                prev_time = time.time()
                
                # Check if memoised
                if i < self.num_memoise and len(self.memoised_data) == self.num_memoise:
                    a, b, c, d, e, f2, g, h = self.memoised_data[i]
                    delete_loaded_data = False
                else:
                    mem_before = int(os.popen("free -gh").readlines()[1].strip().split()[2][:-1])
                    
                    with h5py.File(self.root_path + self.init_path + self.file_name, 'r') as f:
                        a = f['a'][start[i] : end[i]]
                        b = f['b'][start[i] : end[i]]
                        c = f['c'][start[i] : end[i]]
                        d = f['d'][start[i] : end[i]]
                        e = f['e'][start[i] : end[i]]
                        f2 = f['f'][start[i] : end[i]]
                        g = f['g'][start[i] : end[i]]
                        h = f['h'][start[i] : end[i]]

                    mem_after = int(os.popen("free -gh").readlines()[1].strip().split()[2][:-1])
                    print("Memory taken to load", i+1, "/", self.num_times_to_load, "part:", mem_after - mem_before, "G")
                    
                    # Memoise
                    if i < self.num_memoise: 
                        self.memoised_data.append([ a, b, c, d, e, f2, g, h ])
                        delete_loaded_data = False

                print("Time taken to load", i+1, "/", self.num_times_to_load, "part:", round(time.time() - prev_time, 1), "s")

            for index in tqdm(range(0, len(a), self.bsz)):
                if torch == True:
                    yield [
                        Variable(LongTensor(a[index : index + self.bsz])), 
                        Variable(LongTensor(b[index : index + self.bsz])), 
                        Variable(LongTensor(c[index : index + self.bsz])),  
                        Variable(LongTensor(d[index : index + self.bsz])), 
                        Variable(LongTensor(e[index : index + self.bsz])), 
                        Variable(LongTensor(f2[index : index + self.bsz])), 
                        Variable(LongTensor(g[index : index + self.bsz])), 
                    ], Variable(FloatTensor(h[index : index + self.bsz]))
                else:
                    yield [
                        a[index : index + self.bsz], 
                        b[index : index + self.bsz], 
                        c[index : index + self.bsz],  
                        d[index : index + self.bsz], 
                        e[index : index + self.bsz], 
                        f2[index : index + self.bsz], 
                        g[index : index + self.bsz], 
                    ], h[index : index + self.bsz]

            if delete_loaded_data:
                del a, b, c, d, e, f2, g, h
                gc.collect()

def load_data_fast(hyper_params):
    print("Loading data...")

    num_users, num_items, num_words = load_obj(hyper_params['data_dir'] + 'num_users_items')

    hyper_params['total_users'] = num_users
    hyper_params['total_items'] = num_items
    hyper_params['total_words'] = num_words

    train_loader = DataLoader(hyper_params, 'train.hdf5')
    test_loader = DataLoader(hyper_params, 'test.hdf5')
    val_loader = DataLoader(hyper_params, 'val.hdf5')

    return train_loader, test_loader, val_loader, hyper_params
