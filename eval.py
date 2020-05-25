import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

def evaluate(model, criterion, reader, hyper_params, user_count, item_count, review):
    metrics = {}
    total_loss = FloatTensor([ 0.0 ])
    total_batches = 0.0
    mse_right, conv_loss = 0.0, 0.0
    total_temp, total_temp2 = 0.0, 0.0

    user_count_mse_map = {}
    item_count_mse_map = {}

    model.eval()

    with torch.no_grad():
        at = 0

        for data, y in reader.iter(eval = True):
            _, _, _, _, _, user, item = data
            
            output = model(data)

            if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
                mse = criterion(output[0], y, return_mean = False).data
                mse_right += criterion(output[1], y).data
                conv_loss += output[2].data
            else:
                mse = criterion(output, y, return_mean = False).data

            total_temp += torch.sum(mse)
            try: total_temp2 += float(int(output.shape[0]))
            except: total_temp2 += float(int(output[0].shape[0])) # Transnets

            for batch in range(int(y.shape[0])):
                user_id = int(user[batch])
                item_id = int(item[batch])

                if user_id not in user_count: user_count[user_id] = 0
                if item_id not in item_count: item_count[item_id] = 0

                if user_count[user_id] not in user_count_mse_map: user_count_mse_map[ user_count[user_id] ] = []
                if item_count[item_id] not in item_count_mse_map: item_count_mse_map[ item_count[item_id] ] = []

                user_count_mse_map[ user_count[user_id] ].append(float(mse[batch]))
                item_count_mse_map[ item_count[item_id] ].append(float(mse[batch]))

            total_batches += 1.0

        metrics['MSE'] = round(float(total_temp) / total_temp2, 4)
        if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]:
            metrics['MSE_right'] = round(float(mse_right) / total_batches, 4)
            metrics['MSE_transform'] = round(float(conv_loss) / total_batches, 4)

    return metrics, user_count_mse_map, item_count_mse_map

def eval_ranking(model, reader, hyper_params, review = False):  
    top_indices = torch.zeros(len(reader.data), 1).long()
    at = 0

    with torch.no_grad():
        for data, y in reader.iter_negs(review):
            output = model(data)

            if hyper_params['model_type'] in [ 'transnet', 'transnet++' ]: output = output[0]

            for batch in range(int(y.shape[0])):
                # Calculating raning metrics
                vals, inds = torch.topk(output[batch], k = 1, sorted = True)
                top_indices[at, :] = inds
                at += 1

    metrics = {}
    ks = [1]
    for k in ks:
        hr, total = 0.0, 0.0

        for i in range(at):
            pred = top_indices[i].cpu().numpy()[:k]
            if 0 in pred: hr += 1.0
            total += 1.0

        metrics['HR@' + str(k)] = round(100.0 * hr / total, 2)

    return metrics