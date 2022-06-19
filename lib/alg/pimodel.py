import torch
import torch.nn as nn
import torch.nn.functional as F

class PiModel(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, y, model, mask):
        _, y_hat = model(x)
        return (F.mse_loss(y_hat.softmax(1), y.detach(), reduction="none").mean(1) * mask).mean()
