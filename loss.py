import torch

class MSELoss(torch.nn.Module):
    def __init__(self, hyper_params):
        super(MSELoss, self).__init__()

    def forward(self, output, y, return_mean = True):
        mse = torch.pow(output - y, 2)
                
        if return_mean: return torch.mean(mse)
        return mse
