import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from pytorch_models.common_pytorch_models import TextCNN, TorchFM # TorchFM doesn't include global bias

class NARRE(nn.Module):
    def __init__(self, hyper_params):
        super(NARRE, self).__init__()
        self.hyper_params = hyper_params

        word_vectors = load_obj(hyper_params['data_dir'] + '/word2vec')
        self.word2vec = nn.Embedding.from_pretrained(FloatTensor(word_vectors))
        self.word2vec.requires_grad = False # Not trainable

        self.user_embedding = nn.Embedding(hyper_params['total_users'] + 2, hyper_params['latent_size'])
        self.item_embedding = nn.Embedding(hyper_params['total_items'] + 2, hyper_params['latent_size'])

        self.user_conv = TextCNN(hyper_params)
        self.item_conv = TextCNN(hyper_params)

        self.attention_scorer_user = nn.Sequential(
            nn.Linear(2 * self.hyper_params['latent_size'], self.hyper_params['latent_size']),
            nn.ReLU(),
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(self.hyper_params['latent_size'], 1)
        )

        self.attention_scorer_item = nn.Sequential(
            nn.Linear(2 * self.hyper_params['latent_size'], self.hyper_params['latent_size']),
            nn.ReLU(),
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(self.hyper_params['latent_size'], 1)
        )

        self.final = nn.Sequential(
            nn.Dropout(hyper_params['dropout']),
            nn.Linear(self.hyper_params['latent_size'], self.hyper_params['latent_size']),
            nn.ReLU(),
            nn.Linear(self.hyper_params['latent_size'], 1)
        )

        self.user_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_users'] + 2) ]), requires_grad=True)
        self.item_bias = nn.Parameter(FloatTensor([ 0.1 for _ in range(hyper_params['total_items'] + 2) ]), requires_grad=True)
        self.global_bias = nn.Parameter(FloatTensor([ 4.0 ]), requires_grad = True)

        self.dropout = nn.Dropout(hyper_params['dropout'])
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def attention(self, x, other_x = None, scorer = None):

        # Attention input
        cat_input = torch.cat([ x, other_x ], dim = -1)

        # Get attention scores
        attention_scores = scorer(cat_input)[:, :, 0]
        attention_scores = F.softmax(attention_scores, dim = -1)

        # Multiply
        temp_output = attention_scores.unsqueeze(-1) * x
        return torch.sum(temp_output, dim = 1)

    def forward(self, data):
        _, users_who_reviewed, reviewed_items, user_reviews, item_reviews, user_id, item_id = data

        final_shape = (user_id.shape[0])
        first_dim = user_id.shape[0]
        if len(user_id.shape) > 1:
            final_shape = (user_id.shape[0], user_id.shape[1])
            first_dim = user_id.shape[0] * user_id.shape[1]

        # For handling negatives
        users_who_reviewed = users_who_reviewed.view(first_dim, -1)
        reviewed_items = reviewed_items.view(first_dim, -1)
        user_reviews = user_reviews.view(first_dim, user_reviews.shape[-2], user_reviews.shape[-1])
        item_reviews = item_reviews.view(first_dim, item_reviews.shape[-2], item_reviews.shape[-1])
        user_id = user_id.view(-1)
        item_id = item_id.view(-1)

        in_shape  = user_reviews.shape                            # [bsz x num_reviews x num_words]
        in_shape2 = item_reviews.shape

        # Get user and item biases
        user_bias = self.user_bias.gather(0, user_id.view(-1)).view(user_id.shape)
        item_bias = self.item_bias.gather(0, item_id.view(-1)).view(item_id.shape)

        # View
        user_reviews = user_reviews.view(in_shape[0], in_shape[1] * in_shape[2])
        item_reviews = item_reviews.view(in_shape2[0], in_shape2[1] * in_shape2[2])

        # Embed words
        user = self.word2vec(user_reviews)                        # [bsz x (num_reviews*num_words) x 300]
        item = self.word2vec(item_reviews)                        # [bsz x (num_reviews*num_words) x 300]

        # Separate reviews
        user = user.view(in_shape[0] * in_shape[1], in_shape[2], -1)      # [(bsz * num_reviews) x num_words x word_embedding]
        item = item.view(in_shape2[0] * in_shape2[1], in_shape2[2], -1)   # [(bsz * num_reviews) x num_words x word_embedding]

        # Extract features
        user = self.user_conv(user)                                    # [(bsz * num_reviews) x 32]
        item = self.item_conv(item)                                    # [(bsz * num_reviews) x 32]

        # View
        user = user.view(in_shape[0], in_shape[1], -1)                 # [bsz x num_reviews x 32]
        item = item.view(in_shape2[0], in_shape2[1], -1)               # [bsz x num_reviews x 32]

        reviewed_items_embedded = self.item_embedding(reviewed_items)
        user = self.attention(user, reviewed_items_embedded, self.attention_scorer_user)  # [bsz x 32]
        users_who_reviewed_embedded = self.user_embedding(users_who_reviewed)
        item = self.attention(item, users_who_reviewed_embedded, self.attention_scorer_item)  # [bsz x 32]
        
        user_id = self.dropout(self.user_embedding(user_id))                         # [bsz x 32]
        item_id = self.dropout(self.item_embedding(item_id))                         # [bsz x 32]
        
        user = user + user_id
        item = item + item_id

        # Element-wise multiply and get single score
        cat = user * item
        rating = self.final(cat)[:, 0] # [bsz]
        return (rating + user_bias + item_bias + self.global_bias).view(final_shape)
