import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pytorch_models.common_pytorch_models import TorchFM # No global bias

class GMF(nn.Module):
    def __init__(self, hyper_params):
        super(GMF, self).__init__()
        self.hyper_params = hyper_params

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 1) ]))
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 1) ]))
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]))

        self.user_embedding = nn.Embedding(hyper_params['total_users'] + 1, hyper_params['latent_size'])
        self.item_embedding = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['latent_size'])
        self.final = nn.Linear(hyper_params['latent_size'], 1)
        self.dropout = nn.Dropout(hyper_params['dropout'])

    def forward(self, data):
        _, _, _, _, _, user_id, item_id = data

        # For the FM
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        # Embed Latent space
        user = self.dropout(self.user_embedding(user_id.view(-1)))                         # [bsz x 32]
        item = self.dropout(self.item_embedding(item_id.view(-1)))                         # [bsz x 32]
        joint = user * item
        rating = self.final(joint)[:, 0].view(user_id.shape) # [bsz]
        return user_bias + item_bias + self.global_bias + rating

class MLP(nn.Module):
    def __init__(self, hyper_params):
        super(MLP, self).__init__()
        self.hyper_params = hyper_params

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 1) ]))
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 1) ]))
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]))

        self.user_embedding = nn.Embedding(hyper_params['total_users'] + 1, hyper_params['latent_size'])
        self.item_embedding = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['latent_size'])
        self.project = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(2 * hyper_params['latent_size'], hyper_params['latent_size']),
            nn.ReLU(),
            nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        )
        self.final = nn.Linear(hyper_params['latent_size'], 1)
        self.dropout = nn.Dropout(hyper_params['dropout'])

    def forward(self, data):
        _, _, _, _, _, user_id, item_id = data

        # For the FM
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        # Embed Latent space
        user = self.dropout(self.user_embedding(user_id.view(-1)))                         # [bsz x 32]
        item = self.dropout(self.item_embedding(item_id.view(-1)))                         # [bsz x 32]
        joint = torch.cat([ user, item ], dim = -1)
        joint = self.project(joint)
        rating = self.final(joint)[:, 0].view(user_id.shape) # [bsz]
        return user_bias + item_bias + self.global_bias + rating

class NeuMF(nn.Module):
    def __init__(self, hyper_params):
        super(NeuMF, self).__init__()
        self.hyper_params = hyper_params

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 1) ]))
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 1) ]))
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]))
    
        self.gmf_user_embedding = nn.Embedding(hyper_params['total_users'] + 1, hyper_params['latent_size'])
        self.gmf_item_embedding = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['latent_size'])

        self.mlp_user_embedding = nn.Embedding(hyper_params['total_users'] + 1, hyper_params['latent_size'])
        self.mlp_item_embedding = nn.Embedding(hyper_params['total_items'] + 1, hyper_params['latent_size'])

        self.project = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(2 * hyper_params['latent_size'], hyper_params['latent_size']),
            nn.ReLU(),
            nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        )
        self.final = nn.Linear(2 * hyper_params['latent_size'], 1)
        self.dropout = nn.Dropout(hyper_params['dropout'])

    def init(self, gmf_model, mlp_model):
        with torch.no_grad():
            self.gmf_user_embedding.weight.data = gmf_model.user_embedding.weight.data
            self.gmf_item_embedding.weight.data = gmf_model.item_embedding.weight.data

            self.mlp_user_embedding.weight.data = mlp_model.user_embedding.weight.data
            self.mlp_item_embedding.weight.data = mlp_model.item_embedding.weight.data

            for i in range(len(self.project)): 
                try:
                    self.project[i].weight.data = mlp_model.project[i].weight.data
                    self.project[i].bias.data = mlp_model.project[i].bias.data
                except: pass

            self.final.weight.data = torch.cat([ gmf_model.final.weight.data, mlp_model.final.weight.data ], dim = -1)
            self.final.bias.data = 0.5 * (gmf_model.final.bias.data + mlp_model.final.bias.data)

            self.user_bias.data = 0.5 * (gmf_model.user_bias.data + mlp_model.user_bias.data)
            self.item_bias.data = 0.5 * (gmf_model.item_bias.data + mlp_model.item_bias.data)

    def forward(self, data):
        _, _, _, _, _, user_id, item_id = data

        # For the FM
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        # GMF Part
        user = self.dropout(self.gmf_user_embedding(user_id.view(-1))) # [bsz x 32]
        item = self.dropout(self.gmf_item_embedding(item_id.view(-1))) # [bsz x 32]
        gmf_joint = user * item

        # MLP Part
        user = self.dropout(self.mlp_user_embedding(user_id.view(-1))) # [bsz x 32]
        item = self.dropout(self.mlp_item_embedding(item_id.view(-1))) # [bsz x 32]
        mlp_joint = torch.cat([ user, item ], dim = -1)
        mlp_joint = self.project(mlp_joint)

        # NeuMF
        final = torch.cat([ gmf_joint, mlp_joint ], dim = -1)
        rating = self.final(final)[:, 0].view(user_id.shape) # [bsz]

        return user_bias + item_bias + self.global_bias + rating
