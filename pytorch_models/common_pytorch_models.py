import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, hyper_params, window_sizes = [3]):
        super(TextCNN, self).__init__()
        self.hyper_params = hyper_params

        self.num_filters = 100
        self.kernel_size = 5

        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.num_filters, [window_size, self.hyper_params['word_embed_size']], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(self.num_filters * len(window_sizes), hyper_params['latent_size'])
        self.dropout = nn.Dropout(hyper_params['dropout'])

    def forward(self, x):
        in_shape = x.shape                     # [bsz x (num_reviews*num_words) x word_embedding]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(x, 1)              # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))               # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)         # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)                   # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)              # [B, F * window]
        x = self.dropout(self.fc(x))           # [B, class]

        return x

class TorchFM(nn.Module):
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)
        
    def forward(self, x):
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) #S_1^2
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out
