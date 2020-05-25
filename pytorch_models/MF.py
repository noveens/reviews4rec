import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pytorch_models.common_pytorch_models import TorchFM # No global bias

class MF(nn.Module):
    def __init__(self, hyper_params):
        super(MF, self).__init__()
        self.hyper_params = hyper_params

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 1) ]))
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 1) ]))
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]))

        if hyper_params['model_type'] in [ 'MF', 'MF_dot' ]:
            latent_size = hyper_params['latent_size']
            
            self.user_embedding = nn.Embedding(hyper_params['total_users'] + 1, latent_size)
            self.item_embedding = nn.Embedding(hyper_params['total_items'] + 1, latent_size)
            
            self.dropout = nn.Dropout(hyper_params['dropout'])

        if hyper_params['model_type'] == 'MF':
            self.projection = nn.Sequential(
                nn.Dropout(hyper_params['dropout']),
                nn.Linear(2 * latent_size, latent_size),
                nn.ReLU(),
                nn.Linear(latent_size, latent_size)
            )

            self.final = TorchFM(2 * hyper_params['latent_size'], hyper_params['latent_size'])

            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU()

    def forward(self, data):
        _, _, _, _, _, user_id, item_id = data

        self.in_shape = user_id.shape

        # For the FM
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        if self.hyper_params['model_type'] == 'bias_only': 
            return user_bias + item_bias + self.global_bias

        # Embed Latent space
        user = self.dropout(self.user_embedding(user_id.view(-1)))                         # [bsz x 32]
        item = self.dropout(self.item_embedding(item_id.view(-1)))                         # [bsz x 32]

        # Dot product
        if self.hyper_params['model_type'] == 'MF_dot':
            rating = torch.sum(user * item, dim = -1).view(user_id.shape)
            return user_bias + item_bias + self.global_bias + rating

        mf_vector = user * item
        cat = torch.cat([ user, item ], dim = -1)
        mlp_vector = self.projection(cat)

        # Concatenate and get single score
        cat = torch.cat([ mlp_vector, mf_vector ], dim = -1)
        rating = self.final(cat)[:, 0].view(user_id.shape) # [bsz]

        return user_bias + item_bias + self.global_bias + rating
