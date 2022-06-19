import torch
import torch.nn as nn
import torch.nn.functional as F

class PL(nn.Module):
    def __init__(self, threshold, tot_class=6):
        super().__init__()
        self.th = threshold
        self.tot_class = tot_class

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)
        onehot_label = self.__make_one_hot(y_probs.max(1)[1], self.tot_class).float()
        gt_mask = (y_probs > self.th).float()
        gt_mask = gt_mask.max(1)[0] # reduce_any
        lt_mask = 1 - gt_mask # logical not
        p_target = gt_mask[:,None] * self.tot_class * onehot_label + lt_mask[:,None] * y_probs
        _, output = model(x)
        loss = (-(p_target.detach() * F.log_softmax(output, 1)).sum(1)*mask).mean()
        return loss

    def __make_one_hot(self, y, n_classes=6):
        return torch.eye(n_classes)[y].to(y.device)
