import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pytorch_models.common_pytorch_models import TextCNN, TorchFM # TorchFM doesn't include global bias

class Source(nn.Module):
    def __init__(self, hyper_params):
        super(Source, self).__init__()
        self.hyper_params = hyper_params

        self.user_conv = TextCNN(hyper_params)
        self.item_conv = TextCNN(hyper_params)

        self.project = nn.Sequential(
            nn.Linear(2 * hyper_params['latent_size'], hyper_params['latent_size']),
            nn.ReLU(),
            nn.Linear(hyper_params['latent_size'], hyper_params['latent_size'])
        )

        self.dropout = nn.Dropout(hyper_params['dropout'])

    def forward(self, user, item):
        # Extract features
        user = self.user_conv(user)                                    # [bsz x 32]
        item = self.item_conv(item)                                    # [bsz x 32]

        # Concatenate and get single score
        cat = torch.cat([ user, item ], dim = -1)

        temp = self.project(cat)
        temp = self.dropout(temp)
        self.ir = temp

        return None

class Target(nn.Module):
    def __init__(self, hyper_params):
        super(Target, self).__init__()
        self.hyper_params = hyper_params
        
        word_vectors = load_obj(hyper_params['data_dir'] + '/word2vec')
        self.word2vec = nn.Embedding.from_pretrained(FloatTensor(word_vectors))
        self.word2vec.requires_grad = False # Not trainable

        self.conv = TextCNN(hyper_params)
        self.dropout = nn.Dropout(hyper_params['dropout'])
        self.fm = TorchFM(hyper_params['latent_size'], 8)

    def embed(self, review):
        return self.word2vec(review)

    def forward(self, this):
        # Extract features
        this = self.conv(this)                                    # [bsz x 32]
        temp = self.dropout(this)
        self.ir = temp
        
        return self.fm(self.ir)

class TransNet(nn.Module):
    def __init__(self, hyper_params):
        super(TransNet, self).__init__()
        self.hyper_params = hyper_params
        
        self.target = Target(hyper_params)
        xavier_init(self.target)

        self.source = Source(hyper_params)
        xavier_init(self.source)

        if hyper_params['model_type'] == 'transnet++':
            self.user_embedding = nn.Embedding(hyper_params['total_users'] + 2, 5)
            self.item_embedding = nn.Embedding(hyper_params['total_items'] + 2, 5)
            self.source_fm = TorchFM(10 + hyper_params['latent_size'], 8)
        else:
            self.source_fm = TorchFM(hyper_params['latent_size'], 8)

        self.dropout = nn.Dropout(hyper_params['dropout'])

    def forward(self, data):
        this_reviews, _, _, user_reviews, item_reviews, user_id, item_id = data

        final_shape = (user_id.shape[0])
        first_dim = user_id.shape[0]
        if len(user_id.shape) > 1:
            final_shape = (user_id.shape[0], user_id.shape[1])
            first_dim = user_id.shape[0] * user_id.shape[1]

        # For handling negatives
        this_reviews = this_reviews.view(first_dim, -1)
        user_reviews = user_reviews.view(first_dim, -1)
        item_reviews = item_reviews.view(first_dim, -1)
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        # Embed words
        user = self.target.embed(user_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        item = self.target.embed(item_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        this = self.target.embed(this_reviews)               # [bsz x (num_reviews*num_words) x word_embedding]
        
        # Forward
        self.source(user, item)

        if self.hyper_params['model_type'] == 'transnet++':
            user_id = self.dropout(self.user_embedding(user_id))
            item_id = self.dropout(self.item_embedding(item_id))
            final = torch.cat([ user_id, item_id, self.source.ir ], dim = -1)
        else:
            final = self.source.ir
        
        source_out = self.source_fm(final)

        target_out = self.target(this)

        return [
            (source_out[:, 0]).view(final_shape),
            (target_out[:, 0]).view(final_shape),
            torch.mean(torch.sum(torch.pow(self.source.ir - self.target.ir, 2), -1))
        ]
