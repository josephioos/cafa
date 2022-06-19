import torch
import os
from easydl import *
from torch.functional import norm

def compute_statistics(feature, onehotlabel, eps=1.e-6):
    '''
        compute the variance of the input feature, 
        return the mean variance w.r.t. the feature dimension.
        feature:      (n, 1, d)
        onehotlabel:  (n, k, 1)
        mean_var:     (1, k, 1)
    '''
    mean = torch.sum(onehotlabel * feature, dim=0, keepdim=True)
    f2 = (onehotlabel * feature * feature).sum(0, keepdim=True)
    mean2 = mean * mean
    fmean = (onehotlabel * mean * feature).sum(0, keepdim=True)
    var = f2 - 2 * fmean + mean2 + eps
    mean_var = torch.mean(var, dim=2)
    return mean_var


def make_one_hot(y, tot_class=6):
    '''
        convert integer label to onehot encoding. 
    '''
    return torch.eye(tot_class)[y].to(y.device)


# during validation phase.            
def save_var(feature, target, alg):
    """ compute the mean variance of labeled data and unlabeled data, repectively. """
    feature = feature.unsqueeze(1)
    target_onehot = make_one_hot(target).unsqueeze(2)
    mean_var = compute_statistics(feature, target_onehot).squeeze()

    fname = 'meanvar_'
    for i in range(6):
        with open(os.path.join('result', 'meanvar', fname+str(i)+'_'+alg+'.txt'), 'a') as f:
            f.write(str(mean_var[i].item()) + '\n')
            
def TempScale(p, t):
    return p / t

##################################
# adapted from Universal Domain Adaptation.
# https://github.com/thuml/Universal-Domain-Adaptation

def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)


def get_label_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    min_val = pred_shift.min()
    max_val = pred_shift.max()
    pred_shift = (pred_shift - min_val) / (max_val - min_val)
    pred_shift = reverse_sigmoid(pred_shift)
    pred_shift = pred_shift / class_temperature
    pred_shift = nn.Sigmoid()(pred_shift)

    domain_logit = reverse_sigmoid(domain_out)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit)
    
    weight = domain_out - pred_shift
    weight = weight.detach()
    return weight


def get_unlabel_share_weight(domain_out, pred_shift, domain_temperature=1.0, class_temperature=1.0):
    weight = get_label_share_weight(domain_out, pred_shift, domain_temperature, class_temperature)
    return -weight

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / max(torch.mean(x), 1e-6)
    return x.detach()

def feature_scaling(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def pseudo_label_calibration(pslab, weight):
    weight = weight.transpose(1, 0).expand(pslab.shape[0], -1)
    weight = normalize_weight(weight)
    pslab = torch.exp(pslab)
    pslab = pslab * weight
    pslab = pslab / torch.sum(pslab, 1, keepdim=True)
    return pslab







