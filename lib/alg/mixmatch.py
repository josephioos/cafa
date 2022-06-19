import torch
import torch.nn as nn
import torch.nn.functional as F

class MixMatch(nn.Module):
    def __init__(self, temperature, n_augment, alpha):
        super().__init__()
        self.T = temperature
        self.K = n_augment
        self.beta_distirb = torch.distributions.beta.Beta(alpha, alpha)

    def sharpen(self, y):
        y = y.pow(1/self.T)
        return y / y.sum(1,keepdim=True)

    def forward(self, x, y, model, mask):
        # mixup
        index = torch.randperm(x.shape[0])
        shuffled_x, shuffled_y = x[index], y[index]
        lam = self.beta_distirb.sample().item()
        lam = max(lam, 1-lam)
        mixed_x = lam * x + (1-lam) * shuffled_x
        mixed_y = lam * y.softmax(1) + (1-lam) * shuffled_y.softmax(1)
        # mean squared error
        loss = F.mse_loss(model(mixed_x), mixed_y)
        return loss
